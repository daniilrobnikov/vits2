""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols

# Special symbol ids
PAD_ID = symbols.index("_")
SPACE_ID = symbols.index(" ")

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names, language="en-us"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    cleaned_text = _clean_text(text, cleaner_names, language=language)
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    return [_symbol_to_id[symbol] for symbol in cleaned_text]


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    return "".join([_id_to_symbol[id] for id in sequence])


def _clean_text(text, cleaner_names, language="en-us"):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        assert callable(cleaner), f"Unknown cleaner: {name}"
        text = cleaner(text, language=language)
    return text
