"""Sampling prompt/template identity; no controller or model execution."""

import hashlib
import json

from baseline_agent.prompt import CONTINUE, SYSTEM_PROMPT
from batched_inference.conversation import ASSISTANT_PREFILL, TOKEN_HISTORY_VERSION


def template_digest(tokenizer, *, system_prompt=SYSTEM_PROMPT, continuation=CONTINUE):
    """Hash the tokenizer chat template, system prompt and token-history strategy.

    This identifies rendering behavior, not actual model/tokenizer weight bytes.
    Raises ValueError if the tokenizer has no explicit supported chat template.
    """
    template = tokenizer.chat_template
    if not isinstance(template, (str, dict)) or not template:
        raise ValueError("An explicit chat template is required")
    settings = {
        "chat_template": template,
        "token_history": TOKEN_HISTORY_VERSION,
        "system_prompt": system_prompt,
        "continue": continuation,
        "prefill": ASSISTANT_PREFILL,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "add_special_tokens_on_encode": False,
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
