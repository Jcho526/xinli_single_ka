import os
from typing import Dict, List, Optional, Union

import sentencepiece as spm
from transformers import PreTrainedTokenizer


class SPTokenizer:
    def __init__(self, model_path: str):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")

        self.processor = spm.SentencePieceProcessor()
        self.processor.Load(model_path)

        self.n_words = self.processor.vocab_size()
        self.bos_id = self.processor.bos_id()
        self.eos_id = self.processor.eos_id()
        self.pad_id = self.processor.pad_id()
        self.unk_id = self.processor.unk_id()

    def tokenize(self, s: str) -> List[str]:
        return self.processor.EncodeAsPieces(s)

    def encode(self, s: str) -> List[int]:
        return self.processor.EncodeAsIds(s)

    def decode(self, t: List[int]) -> str:
        return self.processor.DecodeIds(t)

    def convert_token_to_id(self, token: str) -> int:
        return self.processor.PieceToId(token)

    def convert_id_to_token(self, index: int) -> str:
        return self.processor.IdToPiece(index)

    def __len__(self) -> int:
        return self.n_words


class ChatGLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask", "position_ids"]

    def __init__(self, vocab_file, padding_side="left", clean_up_tokenization_spaces=False, **kwargs):
        # 关键修复：不要把这些 special token 从 kwargs 直接传给父类，
        # 否则 transformers 初始化时会尝试 setattr(self, "eos_token", ...)
        # 而老代码里如果把它们写成只读 property 就会报错。
        kwargs.pop("eos_token", None)
        kwargs.pop("pad_token", None)
        kwargs.pop("unk_token", None)
        kwargs.pop("bos_token", None)

        self.name = "GLMTokenizer"
        self.vocab_file = vocab_file
        self.tokenizer = SPTokenizer(vocab_file)

        self.special_tokens = {
            "<bos>": self.tokenizer.bos_id,
            "<eos>": self.tokenizer.eos_id,
            "<pad>": self.tokenizer.pad_id,
        }

        super().__init__(
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

        # 关键修复：在父类初始化完成后，用内部字段设置 special tokens，
        # 避免和 property/setter 机制冲突。
        self._bos_token = "<bos>"
        self._eos_token = "<eos>"
        self._pad_token = "<pad>"
        self._unk_token = "<unk>"

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    def get_vocab(self):
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.convert_tokens_to_ids(token)

    @property
    def bos_token_id(self):
        return self.get_command("<bos>")

    @property
    def eos_token_id(self):
        return self.get_command("<eos>")

    @property
    def pad_token_id(self):
        return self.get_command("<pad>")

    @property
    def unk_token_id(self):
        return self.tokenizer.unk_id

    def _tokenize(self, text, **kwargs):
        return self.tokenizer.tokenize(text)

    def _convert_token_to_id(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.tokenizer.convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        if index == self.bos_token_id:
            return "<bos>"
        if index == self.eos_token_id:
            return "<eos>"
        if index == self.pad_token_id:
            return "<pad>"
        return self.tokenizer.convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens):
        ids = [self._convert_token_to_id(token) for token in tokens]
        return self.tokenizer.decode(ids)

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None):
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path ({save_directory}) should be a directory")

        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            with open(self.vocab_file, "rb") as fin:
                with open(out_vocab_file, "wb") as fout:
                    fout.write(fin.read())

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return len([self.bos_token_id] + token_ids_0 + [self.eos_token_id]) * [0]
        return len(
            [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]
        ) * [0]

    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        **kwargs,
    ):
        ids_0 = self.tokenizer.encode(text)
        ids_1 = self.tokenizer.encode(text_pair) if text_pair is not None else None

        if add_special_tokens:
            return self.build_inputs_with_special_tokens(ids_0, ids_1)
        return ids_0 if ids_1 is None else ids_0 + ids_1

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ):
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        if skip_special_tokens:
            token_ids = [
                i for i in token_ids
                if i not in {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            ]

        return self.tokenizer.decode(token_ids)