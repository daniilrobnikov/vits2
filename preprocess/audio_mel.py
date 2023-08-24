import os
import sys
import glob
import logging
import argparse
import traceback
from tqdm import tqdm
from multiprocessing.pool import Pool
import torch
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


def process_item(ifile: str, sr_config, n_fft, hop_size, win_size, n_mels, fmin, fmax, value_ratio):
    ofile = ifile.replace(".wav", ".spec.pt")

    try:
        wav, sr = torchaudio.load(ifile)

        assert sr == sr_config, f"sample rate: {sr}, expected: {sr_config}"

        spec = wav_to_mel(wav, n_fft, n_mels, sr, hop_size, win_size, fmin, fmax, center=False)
        spec = torch.squeeze(spec, 0) / value_ratio

        torch.save(spec, ofile)
    except:
        traceback.print_exc()
        print("Failed to process {}".format(ifile))
        return None

    return ifile


def process_data(hps: HParams):
    wav_fns = sorted(glob.glob(f"{hps.data_dir}/**/*.wav", recursive=True))
    # wav_fns = wav_fns[:100] # * Enable for testing
    logging.info(f"Preprocessing {len(wav_fns)} files...")

    sr = hps.data.sample_rate
    n_fft = hps.data.n_fft
    hop_size = hps.data.hop_length
    win_size = hps.data.win_length
    n_mels = hps.data.n_mels
    fmin = hps.data.f_min
    fmax = hps.data.f_max

    # Value ratio is used to normalize the spectrogram values to be
    # the same magnitude as the ratio between the n_fft and n_mels (32)
    value_ratio = n_fft / (n_mels * 32)

    p = Pool(int(os.getenv("N_PROC", os.cpu_count())))
    futures = []

    for ifile in wav_fns:
        futures.append(p.apply_async(process_item, args=(ifile, sr, n_fft, hop_size, win_size, n_mels, fmin, fmax, value_ratio)))
    p.close()

    for future in tqdm(futures):
        future.get()
    p.join()


def get_size_by_ext(directory, extension):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(extension):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    return total_size


def get_size(path="."):
    total = 0
    # Use os.scandir() as it's faster than os.listdir()
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_size(entry.path)
    return total


def human_readable_size(size):
    """Converts size in bytes to a human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size}{unit}"
        size //= 1024
    return f"{size}PB"  # PB is for petabyte, which will be used if the size is too large.


if __name__ == "__main__":
    from time import time

    hps = parse_args()

    start = time()
    process_data(hps)
    logging.info(f"Processed data in {time() - start} seconds")

    extension = ".spec.pt"
    logging.info(f"{extension}: \t{human_readable_size(get_size_by_ext(hps.data_dir, extension))}")
    extension = ".wav"
    logging.info(f"{extension}: \t{human_readable_size(get_size_by_ext(hps.data_dir, extension))}")
    logging.info(f"Total: \t\t{human_readable_size(get_size(path=hps.data_dir))}")
