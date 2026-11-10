"""Find exact sampled action boundaries without repeatedly decoding growing prefixes."""

import json
import re
from bisect import bisect_left

from baseline_agent.protocol import ACTION_STOP_SEQUENCES, action_complete


def byte_alphabet():
    """Invert the standard GPT-2/Hugging Face ByteLevel byte-to-character alphabet."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = dict(zip(visible, visible, strict=True))
    extra = 0
    for value in range(256):
        if value not in chars:
            chars[value] = 256 + extra
            extra += 1
    return {chr(char): value for value, char in chars.items()}


BYTE_VALUES = byte_alphabet()
CLOSE_ACTION = re.compile(rb"</(?:search|answer)>")


class ActionDecoder:
    """Fast ByteLevel candidate finder with exact tokenizer-prefix verification.

    Only the unmodified fast ByteLevel decoder without whitespace cleanup uses the
    optimized path. Unsupported tokenizers, non-special added tokens, or a full-text
    mismatch use the original reference scan. Original IDs and log probabilities
    are always sliced; decoded text is never re-tokenized for training.
    """

    def __init__(self, tokenizer):
        """Inspect decoder capabilities once and cache vocabulary bytes lazily."""
        self.tokenizer = tokenizer
        self.cache = {}
        self.special_ids = set(getattr(tokenizer, "all_special_ids", ()))
        self.added_ids = set()
        self.byte_level = False
        try:
            state = json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
            self.byte_level = (
                tokenizer.is_fast
                and state.get("type") == "ByteLevel"
                and not tokenizer.clean_up_tokenization_spaces
            )
            self.added_ids = (
                set(tokenizer.get_added_vocab().values()) - self.special_ids
            )
        except (AttributeError, TypeError, ValueError):
            pass

    def reference_end(self, ids, prefill):
        """Original exhaustive behavior, retained as a correctness fallback/oracle."""
        for end in range(1, len(ids) + 1):
            if action_complete(
                prefill
                + self.tokenizer.decode(list(ids[:end]), skip_special_tokens=True)
            ):
                return end
        return None

    def first_action_end(self, ids, prefill):
        """Return the earliest original token boundary with a complete action, or None.

        Byte offsets retain skipped-special-token positions and split UTF-8 bytes.
        Only boundaries whose final closing tag is followed by Unicode whitespace
        can satisfy action_complete. Verify those candidates using the real decoder,
        preserving quoted tags, malformed actions, trailing suffixes and exact IDs.
        """
        if not self.byte_level or prefill != "<reasoning>":
            return self.reference_end(ids, prefill)
        mapped = self._byte_offsets(ids)
        if mapped is None:
            return self.reference_end(ids, prefill)
        raw, offsets = mapped
        closes = [m.end() for m in CLOSE_ACTION.finditer(raw)]
        if not closes:
            return None
        first = bisect_left(offsets, closes[0])
        closing = 0
        previous_offset = None
        for index in range(first, len(ids)):
            end = offsets[index]
            while closing + 1 < len(closes) and closes[closing + 1] <= end:
                closing += 1
            if end == previous_offset:
                continue
            previous_offset = end
            if raw[closes[closing] : end].decode("utf-8", errors="replace").strip():
                continue
            # This is the only text accepted by the policy controller, so verify
            # exact original prefix semantics rather than trusting a byte offset.
            if action_complete(
                prefill
                + self.tokenizer.decode(
                    list(ids[: index + 1]), skip_special_tokens=True
                )
            ):
                return index + 1
        return None

    def _byte_offsets(self, ids):
        """Return verified ByteLevel bytes and cumulative original-token offsets."""
        pieces, offsets, size = [], [], 0
        for token in ids:
            if token in self.added_ids:
                return None
            if token not in self.cache:
                if token in self.special_ids:
                    self.cache[token] = b""
                else:
                    value = self.tokenizer.convert_ids_to_tokens(token)
                    try:
                        self.cache[token] = bytes(BYTE_VALUES[c] for c in value)
                    except (KeyError, TypeError):
                        return None
            piece = self.cache[token]
            pieces.append(piece)
            size += len(piece)
            offsets.append(size)
        raw = b"".join(pieces)
        decoded = self.tokenizer.decode(list(ids), skip_special_tokens=True)
        if raw.decode("utf-8", errors="replace") != decoded:
            return None
        return raw, offsets

    def reference_stop_end(self, ids, prefill):
        """Fallback: first sampled token boundary containing either closing tag."""
        for end in range(1, len(ids) + 1):
            text = prefill + self.tokenizer.decode(
                list(ids[:end]), skip_special_tokens=True
            )
            if any(stop in text for stop in ACTION_STOP_SEQUENCES):
                return end
        return None

    def first_stop_end(self, ids, prefill):
        """Stop at the first closing action tag, even if the turn is malformed.

        Protocol validation is separate: a tag inside reasoning must terminate
        generation but can never be repaired or accepted as a valid action.
        """
        if not self.byte_level or prefill != "<reasoning>":
            return self.reference_stop_end(ids, prefill)
        mapped = self._byte_offsets(ids)
        if mapped is None:
            return self.reference_stop_end(ids, prefill)
        raw, offsets = mapped
        match = CLOSE_ACTION.search(raw)
        if match is None:
            return None
        end = bisect_left(offsets, match.end()) + 1
        text = prefill + self.tokenizer.decode(
            list(ids[:end]), skip_special_tokens=True
        )
        if not any(stop in text for stop in ACTION_STOP_SEQUENCES):
            return self.reference_stop_end(ids, prefill)
        return end
