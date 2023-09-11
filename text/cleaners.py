"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter.
"""

import re
from typing import List
from torchtext.vocab import Vocab
from phonemizer import phonemize
from phonemizer.separator import Separator


from text.normalize_numbers import normalize_numbers

from text.symbols import _punctuation, PAD_ID, UNK_ID, BOS_ID, EOS_ID


_whitespace_re = re.compile(r"\s+")
_preserved_symbols_re = re.compile(rf"[{_punctuation}]|<.*?>")
separator = Separator(word="<space>", phone=" ")


# ---------------------------------------------------------------------------- #
# |                                Text cleaners                             | #
# ---------------------------------------------------------------------------- #
def lowercase(text: str, *args, **kwargs):
    return text.lower()


def collapse_whitespace(text: str, *args, **kwargs):
    return re.sub(_whitespace_re, " ", text)


def expand_numbers(text: str, *args, **kwargs):
    return normalize_numbers(text)


def phonemize_text(text: List[str] | str, *args, language="en-us", **kwargs):
    return phonemize(text, language=language, backend="espeak", separator=separator, strip=True, preserve_punctuation=True, punctuation_marks=_preserved_symbols_re, with_stress=True, njobs=8)


def add_spaces(text: str, *args, **kwargs):
    spaced_text = re.sub(_preserved_symbols_re, r" \g<0> ", text)
    cleaned_text = re.sub(_whitespace_re, " ", spaced_text)
    return cleaned_text.strip()


# ---------------------------------------------------------------------------- #
# |                               Token cleaners                             | #
# ---------------------------------------------------------------------------- #


def tokenize_text(text: str, vocab: Vocab, *args, **kwargs):
    tokens = text.split()
    return vocab(tokens)


def add_bos_eos(tokens: List[int], *args, **kwargs):
    return [BOS_ID] + tokens + [EOS_ID]


def add_blank(tokens: List[int], *args, **kwargs):
    result = [PAD_ID] * (len(tokens) * 2 + 1)
    result[1::2] = tokens
    return result


def delete_unks(tokens: List[int], *args, **kwargs):
    return [token for token in tokens if token != UNK_ID]


def detokenize_sequence(sequence: List[int], vocab: Vocab, *args, **kwargs):
    return "".join(vocab.lookup_tokens(sequence))
