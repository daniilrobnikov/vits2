from text import cleaners
from text.symbols import symbols
from typing import List


def tokenizer(text: str, cleaner_names: List[str], language="en-us", cleaned_text=False) -> List[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text: string to convert to a sequence of IDs
        cleaner_names: names of the cleaner functions from text/cleaners.py
        language: language ID from https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
        cleaned_text: whether the text has already been cleaned
    Returns:
        List of integers corresponding to the symbols in the text
    """
    if not cleaned_text:
        return _clean_text(text, cleaner_names, language=language)
    else:
        return list(map(int, text.split("\t")))


def detokenizer(sequence: List[int]) -> str:
    """Converts a sequence of tokens back to a string"""
    return "".join(symbols.lookup_tokens(sequence))


def _clean_text(text: str, cleaner_names: List[str], language="en-us") -> str:
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        assert callable(cleaner), f"Unknown cleaner: {name}"
        text = cleaner(text, language=language)
    return text


if __name__ == "__main__":
    cleaner_names = ["phonemize_text", "tokenize_text", "add_bos_eos", "detokenize_sequence"]
    text = "Well, I like pizza. <laugh> You know … Who doesn't like pizza? <laugh>"
    print(tokenizer(text, cleaner_names, language="en-us", cleaned_text=False))
