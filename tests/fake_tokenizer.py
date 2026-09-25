"""A character-level stand-in for a real tokenizer.

Nimble's ``prepare_prompts`` only needs four things from a tokenizer: a chat
template, ``encode``, ``all_special_ids`` and a pad id. A character-level
tokenizer satisfies all of them - and, because every character is exactly one
token, it also satisfies the "each choice code must be a single token at the
answer boundary" rule by construction.

That makes it possible to test the whole data path - prompt rendering, choice
permutation, canonical mapping, batching - offline, with no model download.
"""


class FakeTokenizer:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        self.all_special_ids = [0]
        self.pad_token_id = 0
        self.pad_token = "\x00"
        self.eos_token = "\x00"
        self.padding_side = "left"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False, **kwargs):
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        return [(ord(character) % (self.vocab_size - 1)) + 1 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value - 1) for value in ids if value > 0)

    def convert_ids_to_tokens(self, ids):
        return [chr(value - 1) for value in ids]
