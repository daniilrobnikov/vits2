# VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design

### Jungil Kong, Jihoon Park, Beomjeong Kim, Jeongmin Kim, Dohee Kong, Sangjin Kim

### SK Telecom, South Korea

Single-stage text-to-speech models have been actively studied recently, and their results have outperformed two-stage pipeline systems. Although the previous single-stage model has made great progress, there is room for improvement in terms of its intermittent unnaturalness, computational efficiency, and strong dependence on phoneme conversion. In this work, we introduce VITS2, a single-stage text-to-speech model that efficiently synthesizes a more natural speech by improving several aspects of the previous work. We propose improved structures and training mechanisms and present that the proposed methods are effective in improving naturalness, similarity of speech characteristics in a multi-speaker model, and efficiency of training and inference. Furthermore, we demonstrate that the strong dependence on phoneme conversion in previous works can be significantly reduced with our method, which allows a fully end-to-end single-stage approach.

Demo: https://vits-2.github.io/demo/

Paper: https://arxiv.org/abs/2307.16430

Unofficial implementation of VITS2. This is a work in progress. Please refer to [TODO](#todo) for more details.

<table style="width:100%">
  <tr>
    <th>Duration Predictor</th>
    <th>Normalizing Flows</th>
    <th>Text Encoder</th>
  </tr>
  <tr>
    <td><img src="figures/figure01.png" alt="Duration Predictor" width="100%" style="width:100%"></td>
    <td><img src="figures/figure02.png" alt="Normalizing Flows" width="100%" style="width:100%"></td>
    <td><img src="figures/figure03.png" alt="Text Encoder" width="100%" style="width:100%"></td>
  </tr>
</table>

## Installation:

<a name="installation"></a>

**Clone the repo**

```shell
git clone git@github.com:daniilrobnikov/vits2.git
cd vits2
```

## Setting up the conda env

This is assuming you have navigated to the `vits2` root after cloning it.

**NOTE:** This is tested under `python3.11` with conda env. For other python versions, you might encounter version conflicts.

**PyTorch 2.0**
Please refer [requirements.txt](requirements.txt)

```shell
# install required packages (for pytorch 2.0)
conda create -n vits2 python=3.11
conda activate vits2
pip install -r requirements.txt

conda env config vars set PYTHONPATH="/path/to/vits2"
```

## Download datasets

There are three options you can choose from: LJ Speech, VCTK, and custom dataset.

1. LJ Speech: [LJ Speech dataset](#lj-speech-dataset). Used for single speaker TTS.
2. VCTK: [VCTK dataset](#vctk-dataset). Used for multi-speaker TTS.
3. Custom dataset: You can use your own dataset. Please refer [here](#custom-dataset).

### LJ Speech dataset

1. download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)

```shell
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
cd LJSpeech-1.1/wavs
rm -rf wavs
```

2. save high-resolution mel-spectrograms. See [audio_mel.py](preprocess/audio_mel.py)

```shell
python preprocess/audio_mel.py --data_dir /path/to/LJSpeech-1.1
```

3. rename or create a link to the dataset folder

```shell
ln -s /path/to/LJSpeech-1.1/wavs DUMMY1
```

### VCTK dataset

1. download and extract the [VCTK dataset](https://www.kaggle.com/datasets/showmik50/vctk-dataset)

```shell
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip
```

2. (optional): downsample the audio files to 22050 Hz. See [audio_resample.ipynb](preprocess/audio_resample.ipynb)

3. save high-resolution mel-spectrograms. See [audio_mel.py](preprocess/audio_mel.py)

```shell
python preprocess/audio_mel.py --data_dir /path/to/VCTK-Corpus-0.92
```

4. rename or create a link to the dataset folder

```shell
ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2
```

### Custom dataset

1. create a folder with wav files
2. duplicate the `ljs_base` in `datasets` directory and rename it to `custom_base`
3. open [custom_base](datasets/custom_base) and change the following fields in `config.yaml`:

```yaml
data:
  training_files: datasets/custom_base/filelists/custom_audio_text_train_filelist.txt.cleaned
  validation_files: datasets/custom_base/filelists/custom_audio_text_val_filelist.txt.cleaned
  text_cleaners:
    - english_cleaners2  # text cleaner
  bits_per_sample: 16 # bit depth of wav files
  sample_rate: 22050 # sampling rate if you resampled your wav files
  ...
  n_speakers: 0 # number of speakers in your dataset if you use multi-speaker setting
  cleaned_text: true # See text_phonemizer.ipynb
```

4. save high-resolution mel-spectrograms. See [audio_mel.py](preprocess/audio_mel.py)

```shell
python preprocess/audio_mel.py --data_dir /path/to/custom_dataset
```

5. rename or create a link to the dataset folder. Please refer [text_split.ipynb](preprocess/text_split.ipynb)

```shell
ln -s /path/to/custom_dataset DUMMY3
```

6. install espeak-ng (optional)

**NOTE:** This is required for the [preprocess.py](preprocess.py) and [inference.ipynb](inference.ipynb) notebook to work. If you don't need it, you can skip this step. Please refer [espeak-ng](https://github.com/espeak-ng/espeak-ng)

7. preprocess text

You can do this step by step way:

- create a dataset of text files. See [text_dataset.ipynb](preprocess/text_dataset.ipynb)
- phonemize or just clean up the text. Please refer [text_phonemizer.ipynb](preprocess/text_phonemizer.ipynb)
- create filelists and cleaned version with train test split. See [text_split.ipynb](preprocess/text_split.ipynb)

## Training Examples

```shell
# LJ Speech
python train.py -c datasets/ljs_base/config.yaml -m ljs_base

# VCTK
python train_ms.py -c datasets/vctk_base/config.yaml -m vctk_base

# Custom dataset (multi-speaker)
python train_ms.py -c datasets/custom_base/config.yaml  -m custom_base
```

## Inference Example

See [inference.ipynb](inference.ipynb)
See [inference_batch.ipynb](inference_batch.ipynb) for multiple sentences inference

## Pretrained Models

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing)

## Audio Samples

## Todo

- [ ] text preprocessing
  - [x] update cleaners for multi-language support with 100+ languages
  - [x] update vocabulary to support all symbols and features from IPA. See [phonemes.md](https://github.com/espeak-ng/espeak-ng/blob/ed9a7bcf5778a188cdec202ac4316461badb28e1/docs/phonemes.md#L5)
  - [x] handling unknown, out of vocabulary symbols. Please refer [vocab.py](text/vocab.py) and [vocab - TorchText](https://pytorch.org/text/stable/vocab.html)
  - [x] remove cleaners from text preprocessing. Most cleaners are already implemented in [phonemizer](https://github.com/bootphon/phonemizer). See [cleaners.py](text/cleaners.py)
  - [ ] remove necessity for speakers indexation. See [vits/issues/58](https://github.com/jaywalnut310/vits/issues/58)
- [ ] audio preprocessing
  - [x] batch audio resampling. Please refer [audio_resample.ipynb](preprocess/audio_resample.ipynb)
  - [x] code snippets to find corrupted files in dataset. Please refer [audio_find_corrupted.ipynb](preprocess/audio_find_corrupted.ipynb)
  - [x] code snippets to delete by extension files in dataset. Please refer [delete_by_ext.ipynb](preprocess/delete_by_ext.ipynb)
  - [x] replace scipy and librosa dependencies with torchaudio. See [load](https://pytorch.org/audio/stable/backend.html#id2) and [MelScale](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelScale.html) docs
  - [x] automatic audio range normalization. Please refer [Loading audio data - Torchaudio docs](https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data)
  - [x] add support for stereo audio (multi-channel). See [Loading audio data - Torchaudio docs](https://pytorch.org/audio/stable/tutorials/audio_io_tutorial.html#loading-audio-data)
  - [x] add support for various audio bit depths (bits per sample). See [load - Torchaudio docs](https://pytorch.org/audio/stable/backend.html#id2)
  - [x] add support for various sample rates. Please refer [load - Torchaudio docs](https://pytorch.org/audio/stable/backend.html#id2)
  - [ ] test stereo audio (multi-channel) training
- [x] filelists preprocessing
  - [x] add filelists preprocessing for multi-speaker. Please refer [text_split.ipynb](preprocess/text_split.ipynb)
  - [x] code snippets for train test split. Please refer [text_split.ipynb](preprocess/text_split.ipynb)
  - [x] notebook to link filelists with actual wavs. Please refer [text_split.ipynb](preprocess/text_split.ipynb)
- [ ] other
  - [x] rewrite code for python 3.11
  - [x] replace Cython Monotonic Alignment Search with numba.jit. See [vits-finetuning](https://github.com/SayaSS/vits-finetuning)
  - [x] updated inference to support batch processing
- [ ] pretrained models
  - [ ] training the model for Bengali language. (For now: 55_000 iterations, ~26 epochs)
  - [ ] add pretrained models for multiple languages
- [ ] future work

  - [ ] update model to vits2. Please refer [VITS2](https://arxiv.org/abs/2307.16430)
  - [ ] update model to naturalspeech. Please refer [naturalspeech](https://arxiv.org/abs/2205.04421)
  - [ ] add support for streaming. Please refer [vits_chinese](https://github.com/PlayVoice/vits_chinese/blob/master/text/symbols.py)
  - [ ] update naturalspeech to multi-speaker
  - [ ] replace speakers with multi-speaker embeddings
  - [ ] replace speakers with multilingual training. Each speaker is a language with thhe same IPA symbols
  - [ ] add support for in-context learning

  - [ ] use high-resolution mel-spectrograms in training
  - [ ] compare IPA with ARPABET phonemizer
  - [ ] test different loss functions
  - [ ] test different optimizers, hyperparameters, and schedulers

## Acknowledgements

- This is unofficial repo based on [VITS2](https://arxiv.org/abs/2307.16430)
- Text to phones converter for multiple languages is based on [phonemizer](https://github.com/bootphon/phonemizer)
- We also thank GhatGPT for providing writing assistance.

## References

- [VITS2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design](https://arxiv.org/abs/2307.16430)
- [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://arxiv.org/abs/2106.06103)
- [NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality](https://arxiv.org/abs/2205.04421)
- [A TensorFlow implementation of Google's Tacotron speech synthesis with pre-trained model (unofficial)](https://github.com/keithito/tacotron)

# VITS2
