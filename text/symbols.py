""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_symbols = " !\"',-.:;?abcdefhijklmnopqrstuvwxyz¡«»¿æçðøħŋœǀǁǂǃɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɪɫɬɭɮɯɰɱɲɳɴɵɶɸɹɺɻɽɾʀʁʂʃʄʈʉʊʋʌʍʎʏʐʑʒʔʕʘʙʛʜʝʟʡʢʰʲʷʼˈˌːˑ˔˞ˠˡˤ˥˦˧˨˩̴̘̙̜̝̞̟̠̤̥̩̪̬̮̯̰̹̺̻̼͈͉̃̆̈̊̽͆̚͡βθχᵝᶣ—‖“”…‿ⁿ↑↓↗↘◌ⱱꜛꜜ︎ᵻ"
symbols = list(_pad) + list(_symbols)
