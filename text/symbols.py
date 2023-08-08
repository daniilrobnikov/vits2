""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# _data_letters_ipa = "̃"
# _data_letters = "ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়ঢ়য়ৠ"
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

# Export all symbols:
_symbols = ' !",.:;?abcdefhijklmnopqrstuvwxyz¡«»¿æçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʰʲʷʼˈˌːˑ˔˞ˠˡˤ˥˦˧˨˩̴̘̙̜̝̞̟̠̤̥̩̪̬̮̯̰̹̺̻̼͈͉̃̆̈̊̽͆̚͡βθχᵝᶣ—‖“”…‿ⁿ↑↓↗↘◌ⱱꜛꜜ︎ᵻ'
symbols = list(_symbols)

# Special symbol ids
SPACE_ID = symbols.index(" ")
