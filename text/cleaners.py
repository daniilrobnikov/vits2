"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter.
"""

import re
from typing import List
from phonemizer import phonemize

from text.normalize_numbers import normalize_numbers
from text.symbols import _punctuation, symbols, PAD_ID, BOS_ID, EOS_ID


_whitespace_re = re.compile(r"\s+")
_special_symbols_re = re.compile(r"<.*?>")
_preserved_symbols_re = re.compile(rf"[{_punctuation}]|<.*?>")


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
    return phonemize(text, language=language, backend="espeak", strip=True, preserve_punctuation=True, punctuation_marks=_preserved_symbols_re, with_stress=True, tie=True, njobs=8)


# ---------------------------------------------------------------------------- #
# |                               Token cleaners                             | #
# ---------------------------------------------------------------------------- #
def tokenize_text(text: str, *args, **kwargs):
    tokens = list(text)
    specials_tokens = [(m.start(), m.end()) for m in re.finditer(_special_symbols_re, text)]
    for start, end in reversed(specials_tokens):
        tokens[start:end] = [text[start:end]]
    return symbols(tokens)


def add_bos_eos(tokens: List[int], *args, **kwargs):
    return [BOS_ID] + tokens + [EOS_ID]


def add_blank(tokens: List[int], *args, **kwargs):
    result = [PAD_ID] * (len(tokens) * 2 + 1)
    result[1::2] = tokens
    return result


def detokenize_sequence(sequence: List[int], *args, **kwargs):
    return "".join(symbols.lookup_tokens(sequence))
