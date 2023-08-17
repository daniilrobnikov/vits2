# VITS2 | Preprocessing

## Todo

- [ ] text preprocessing
  - [x] update vocabulary to support all symbols and features from IPA. See [phonemes.md](https://github.com/espeak-ng/espeak-ng/blob/ed9a7bcf5778a188cdec202ac4316461badb28e1/docs/phonemes.md#L5)
  - [x] per dataset filelists preprocessing. Please refer [prepare/filelists.ipynb](datasets/ljs_base/prepare/filelists.ipynb)
  - [ ] handling unknown (out of vocabulary) symbols. Please refer [vocab - TorchText](https://pytorch.org/text/stable/vocab.html)
  - [ ] remove necessity for speakers indexation. See [vits/issues/58](https://github.com/jaywalnut310/vits/issues/58)
- [ ] audio preprocessing
  - [x] replaced scipy and librosa dependencies with torchaudio. See docs [torchaudio.load](https://pytorch.org/audio/stable/backend.html#id2) and [torchaudio.transforms](https://pytorch.org/audio/stable/transforms.html)
  - [ ] update batch audio resampling. Please refer [audio_resample.ipynb](preprocess/audio_resample.ipynb)
  - [ ] test stereo audio (multi-channel) training

# VITS2 | Preprocessing
