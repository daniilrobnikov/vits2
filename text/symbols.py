from torchtext.vocab import vocab as transform_vocab
from collections import OrderedDict


"""
Set of symbols
"""
_space = " "
_punctuation = ';:,.!?¡¿—…"«»“”'
_tones = "12345"
_symbols = "abcdefhijklmnopqrstuvwxyzæçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʰʲʷʼˈˌːˑ˔˞ˠˡˤ˥˦˧˨˩̴̘̙̜̝̞̟̠̤̥̩̪̬̮̯̰̹̺̻̼͈͉̃̆̈̊̽͆̚͡βθχᵝᶣ‖‿ⁿ↑↓↗↘◌ⱱꜛꜜ︎ᵻ"

# Combine all symbols into one set
_symbols = _space + _punctuation + _tones + _symbols


"""
Special symbols
"""
# Define special symbols and indices
special_symbols = ["<pad>", "<unk>", "<bos>", "<eos>", "<laugh>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

symbols = transform_vocab(OrderedDict([(symbol, 1) for symbol in _symbols]), specials=special_symbols)
symbols.set_default_index(UNK_ID)
