""" from https://github.com/keithito/tacotron """

"""
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter.
"""

import re
from typing import List
from phonemizer import phonemize

from .numbers import normalize_numbers


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def expand_numbers(text: List[str] | str, *args, **kwargs):
    if isinstance(text, str):
        return normalize_numbers(text)
    return [normalize_numbers(x) for x in text]


def lowercase(text: List[str] | str, *args, **kwargs):
    if isinstance(text, str):
        return text.lower()
    return [x.lower() for x in text]


def collapse_whitespace(text: List[str] | str, *args, **kwargs):
    if isinstance(text, str):
        return re.sub(_whitespace_re, " ", text)
    return [re.sub(_whitespace_re, " ", x) for x in text]


def phonemize_text(text: List[str] | str, *args, language="en-us", **kwargs):
    return phonemize(text, language=language, backend="espeak", strip=True, preserve_punctuation=True, with_stress=True, tie=True, njobs=8)
