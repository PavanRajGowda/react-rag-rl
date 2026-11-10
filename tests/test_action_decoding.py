"""Exact action-boundary equivalence, including byte splits and replay alignment."""

import json
import random
from types import SimpleNamespace as NS

import pytest

from tinker_training.backend import TinkerBackend
from tinker_training.decoding import BYTE_VALUES, ActionDecoder


class ByteTokenizer:
    """Faithful byte-level decoder with independently configurable token boundaries."""

    is_fast = True
    clean_up_tokenization_spaces = False
    all_special_ids = (10000,)
    eos_token_id = 10000

    def __init__(self, chunks=None):
        alphabet = {v: c for c, v in BYTE_VALUES.items()}
        self.pieces = {i: bytes([i]) for i in range(256)}
        self.pieces.update(chunks or {})
        self.vocab = {
            i: "".join(alphabet[b] for b in raw) for i, raw in self.pieces.items()
        }
        self.backend_tokenizer = NS(
            decoder=NS(__getstate__=lambda: json.dumps({"type": "ByteLevel"}))
        )
        self.calls = 0

    def get_added_vocab(self):
        return {"<eos>": 10000}

    def convert_ids_to_tokens(self, token):
        return self.vocab[token]

    def decode(self, ids, skip_special_tokens=True):
        self.calls += 1
        return b"".join(self.pieces[i] for i in ids if i != 10000).decode(
            "utf-8", errors="replace"
        )


CASES = [
    "reasoning only",
    "x</reasoning><search>facts</search>",
    "x</reasoning><search>facts</search>UNWANTED SUFFIX",
    "x</reasoning><answer>é 中文 🐍</answer>more",
    'Quoted "</search>" in reasoning.</reasoning><search>real</search>',
    "quoted </reasoning><search>example</search> more reasoning</reasoning><answer>yes</answer>",
    "x</reasoning><search></search>",
    "x</reasoning><search>unterminated",
    "x</reasoning><search>nested <answer>x</answer></search>",
    "x</reasoning><answer>yes</answer>\u2003\n",
    "</search>",
    "x</reasoning><search>a</search><search>b</search>",
]


@pytest.mark.parametrize("text", CASES)
def test_byte_and_arbitrary_chunk_boundaries_match_original_scan(text):
    """Do not assume a closing tag, Unicode character or suffix is one token."""
    raw = text.encode()
    for seed in range(8):
        rng = random.Random(seed)
        chunks = {}
        ids = []
        offset = 0
        while offset < len(raw):
            count = rng.randint(1, 12)
            token = 256 + len(chunks)
            chunks[token] = raw[offset : offset + count]
            ids.append(token)
            if rng.random() < 0.2:
                ids.append(10000)
            offset += count
        tokenizer = ByteTokenizer(chunks)
        decoder = ActionDecoder(tokenizer)
        assert decoder.first_action_end(ids, "<reasoning>") == decoder.reference_end(
            ids, "<reasoning>"
        )


def test_split_unicode_whitespace_after_close_and_skipped_specials():
    """An incomplete UTF-8 suffix may become valid whitespace on the next token."""
    raw = b"x</reasoning><answer>yes</answer>"
    tokenizer = ByteTokenizer({256: raw + b"\xe2", 257: b"\x80", 258: b"\x83"})
    decoder = ActionDecoder(tokenizer)
    ids = [256, 10000, 257, 258, 10000]
    assert (
        decoder.first_action_end(ids, "<reasoning>")
        == decoder.reference_end(ids, "<reasoning>")
        == 4
    )


def test_fast_path_checks_candidates_without_decoding_every_prefix():
    """Normal output uses one full decode and one exact boundary verification."""
    tokenizer = ByteTokenizer()
    ids = list(b"Some reasoning.</reasoning><search>facts</search>suffix")
    end = ActionDecoder(tokenizer).first_action_end(ids, "<reasoning>")
    assert end == len(b"Some reasoning.</reasoning><search>facts</search>")
    assert tokenizer.calls == 2


def test_fallback_preserves_nonstandard_tokenizer_and_prefill_semantics():
    """Unsupported decoder behavior retains the original exhaustive algorithm."""
    for prefill in ["<reasoning>", "</reasoning><answer>yes</answer>"]:
        tokenizer = ByteTokenizer()
        tokenizer.clean_up_tokenization_spaces = True
        decoder = ActionDecoder(tokenizer)
        ids = list(b" </reasoning><search>x</search>")
        assert decoder.first_action_end(ids, prefill) == decoder.reference_end(
            ids, prefill
        )


def test_replay_keeps_original_ids_and_corresponding_logprobs_after_trimming():
    """Different per-token logprobs detect alignment errors and accidental re-encoding."""
    tokenizer = ByteTokenizer()
    backend = object.__new__(TinkerBackend)
    backend.tokenizer = tokenizer
    backend.config = NS(max_new_tokens=512)
    backend.action_decoder = ActionDecoder(tokenizer)
    ids = tuple(b"x</reasoning><search>facts</search>EXTRA")
    logps = tuple(-0.01 * (i + 1) for i in range(len(ids)))
    request = NS(assistant_prefill="<reasoning>")
    result, raw_count = backend._decode_response(
        request, NS(sequences=[NS(tokens=ids, logprobs=logps)])
    )
    expected = backend.action_decoder.reference_end(ids, request.assistant_prefill)
    assert result.completion_ids == ids[:expected]
    assert result.sampling_logprobs == logps[:expected]
    assert result.finish_reason == "action"
    assert raw_count == len(ids)


def test_random_bytes_and_truncated_sequences_match_reference():
    """Malformed UTF-8 and partial tags cannot fabricate a completed action."""
    rng = random.Random(42)
    tokenizer = ByteTokenizer()
    decoder = ActionDecoder(tokenizer)
    for _ in range(250):
        raw = bytes(rng.randrange(256) for _ in range(rng.randrange(30)))
        raw += rng.choice(
            [
                b"</reasoning><search>x</search>",
                b"</reasoning><answer>x</answer>",
                b"</search>",
            ]
        )
        raw += bytes(rng.randrange(256) for _ in range(rng.randrange(10)))
        ids = list(raw[: rng.randrange(len(raw) + 1)])
        assert decoder.first_action_end(ids, "<reasoning>") == decoder.reference_end(
            ids, "<reasoning>"
        )


@pytest.mark.parametrize("text", CASES)
def test_first_closing_tag_stops_even_malformed_reasoning(text):
    """Stopping is unconditional; validity must never determine when generation ends."""
    tokenizer = ByteTokenizer()
    ids = list(text.encode())
    decoder = ActionDecoder(tokenizer)
    assert decoder.first_stop_end(ids, "<reasoning>") == decoder.reference_stop_end(
        ids, "<reasoning>"
    )


def test_tag_quoted_in_reasoning_stops_without_accepting_later_action():
    """Malformed early stops cannot be bypassed by looking for a later valid answer."""
    tokenizer = ByteTokenizer()
    backend = object.__new__(TinkerBackend)
    backend.tokenizer = tokenizer
    backend.config = NS(max_new_tokens=512)
    backend.action_decoder = ActionDecoder(tokenizer)
    text = "Quoted </search> here.</reasoning><answer>yes</answer>"
    ids = tuple(text.encode())
    logps = tuple(-0.01 * (i + 1) for i in range(len(ids)))
    result, count = backend._decode_response(
        NS(assistant_prefill="<reasoning>"),
        NS(sequences=[NS(tokens=ids, logprobs=logps)]),
    )
    end = len(b"Quoted </search>")
    assert result.completion_ids == ids[:end]
    assert result.sampling_logprobs == logps[:end]
    assert result.finish_reason == "protocol_stop"
    assert count == len(ids)
