import os
import sys
import glob
import logging
import argparse
import traceback
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torchaudio

from utils.hparams import get_hparams_from_file, HParams
from utils.mel_processing import wav_to_mel

os.environ["OMP_NUM_THREADS"] = "1"
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("-c", "--config", type=str, required=True, help="YAML file for configuration")
    args = parser.parse_args()

    hparams = get_hparams_from_file(args.config)
    hparams.data_dir = args.data_dir
    return hparams


def process_batch(batch, sr_hps, n_fft, hop_size, win_size, n_mels, fmin, fmax):
    wavs = []
    for ifile in batch:
        try:
            wav, sr = torchaudio.load(ifile)
            assert sr == sr_hps, f"sample rate: {sr}, expected: {sr_hps}"
            wavs.append(wav)
        except:
            traceback.print_exc()
            print("Failed to process {}".format(ifile))
            return None

    wav_lengths = torch.tensor([x.size(1) for x in wavs])
    max_wav_len = wav_lengths.max()

    wav_padded = torch.zeros(len(batch), 1, max_wav_len)
    for i, wav in enumerate(wavs):
        wav_padded[i, :, : wav.size(1)] = wav

    spec = wav_to_mel(wav_padded, n_fft, n_mels, sr_hps, hop_size, win_size, fmin, fmax, center=False, norm=False)
    spec = torch.squeeze(spec, 1)

    for i, ifile in enumerate(batch):
        ofile = ifile.replace(".wav", ".spec.pt")
        spec_i = spec[i, :, : wav_lengths[i] // hop_size].clone()
        torch.save(spec_i, ofile)

    return batch


def process_data(hps: HParams):
    wav_fns = sorted(glob.glob(f"{hps.data_dir}/**/*.wav", recursive=True))
    # wav_fns = wav_fns[:100]  # * Enable for testing
    logging.info(f"Max: {mp.cpu_count()}; using 32 CPU cores")
    logging.info(f"Preprocessing {len(wav_fns)} files...")

    sr = hps.data.sample_rate
    n_fft = hps.data.n_fft
    hop_size = hps.data.hop_length
    win_size = hps.data.win_length
    n_mels = hps.data.n_mels
    fmin = hps.data.f_min
    fmax = hps.data.f_max

    # Batch files to optimize disk I/O and computation
    batch_size = 128  # Change as needed
    audio_file_batches = [wav_fns[i : i + batch_size] for i in range(0, len(wav_fns), batch_size)]

    # Use multiprocessing to speed up the conversion
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(process_batch, batch, sr, n_fft, hop_size, win_size, n_mels, fmin, fmax) for batch in audio_file_batches]
        for future in tqdm(futures):
            if future.result() is None:
                logging.warning(f"Failed to process a batch.")
                return


def get_size_by_ext(directory, extension):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(extension):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    return total_size


def human_readable_size(size):
    """Converts size in bytes to a human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}PB"  # PB is for petabyte, which will be used if the size is too large.


if __name__ == "__main__":
    from time import time

    hps = parse_args()

    start = time()
    process_data(hps)
    logging.info(f"Processed data in {time() - start} seconds")

    extension = ".spec.pt"
    size_spec = get_size_by_ext(hps.data_dir, extension)
    logging.info(f"{extension}: \t{human_readable_size(size_spec)}")
    extension = ".wav"
    size_wav = get_size_by_ext(hps.data_dir, extension)
    logging.info(f"{extension}: \t{human_readable_size(size_wav)}")
    logging.info(f"Total: \t\t{human_readable_size(size_spec + size_wav)}")
