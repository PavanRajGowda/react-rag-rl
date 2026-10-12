"""Lazy GPU dependencies; Transformers generation on a T4 in FP16."""

import time

from baseline_agent.prompt import messages
from baseline_agent.protocol import action_complete, select_history

ASSISTANT_PREFILL = "<reasoning>"


class QwenGenerator:
    def __init__(self, model_path, config):
        import torch
        from transformers import (
            AutoTokenizer,
            Qwen3_5ForConditionalGeneration,
            set_seed,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this baseline")
        set_seed(config.seed)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        print(f"Loaded Qwen on {torch.cuda.get_device_name(0)} in FP16", flush=True)

    def make_prefix(self, question):
        self.base_messages = messages(question)
        return self.render_prompt([], "replace")

    def render_prompt(self, events, context_mode):
        conversation = list(self.base_messages)
        for event in select_history(events, context_mode):
            if event["kind"] == "model":
                conversation.append({"role": "assistant", "content": event["text"]})
            else:
                conversation.append(
                    {
                        "role": "user",
                        "content": event["text"]
                        + "\nContinue the original question. If the evidence is sufficient, calculate and emit the final numerical answer in answer tags. Otherwise request evidence using search with company, year and query child tags. Never use search to say that no search is needed. Replace all example placeholders with actual values.",
                    }
                )
        # The requested checkpoint is an ordinary chat model, not a
        # Search-R1-trained continuation model. Start a fresh assistant chat
        # turn after each observation rather than inviting immediate EOS.
        return (
            self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + ASSISTANT_PREFILL
        )

    def generate(self, prompt):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

        encoded = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        length = encoded["input_ids"].shape[1]
        if length + self.config.max_new_tokens > self.config.max_context_tokens:
            raise ValueError(
                f"Prompt {length} + generation reserve {self.config.max_new_tokens} exceeds context budget {self.config.max_context_tokens}; nothing was truncated"
            )
        tokenizer = self.tokenizer

        class StopAtAction(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(
                    input_ids[0, length:],
                    skip_special_tokens=False,
                )
                return action_complete(ASSISTANT_PREFILL + text)

        started = time.perf_counter()
        print(ASSISTANT_PREFILL, end="", flush=True)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.05,
                stopping_criteria=StoppingCriteriaList([StopAtAction()]),
                streamer=TextStreamer(
                    self.tokenizer, skip_prompt=True, skip_special_tokens=True
                ),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0, length:]
        text = ASSISTANT_PREFILL + self.tokenizer.decode(
            generated, skip_special_tokens=True
        )
        return text, {
            "input_tokens": length,
            "assistant_prefill": ASSISTANT_PREFILL,
            "generated_tokens": len(generated),
            "generation_seconds": time.perf_counter() - started,
        }
