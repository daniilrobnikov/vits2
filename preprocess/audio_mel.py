# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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

from tts_utils.mel_processing import wav_to_mel

os.environ["OMP_NUM_THREADS"] = "1"
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt="%m/%d %I:%M:%S %p")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./datasets/LJSpeech-1.1", type=str, help="Directory containing LJSpeech-1.1")
    return parser.parse_args()


def process_item(ifile):
    ofile = ifile.replace(".wav", ".spec.pt")

    n_fft = 16384  # 1024 # TODO 32768 for num_mels = 513; 65536 for num_mels = 1025
    hop_size = 256  # 256
    win_size = 1024  # 1024
    num_mels = 384  # 80
    fmin = 0
    fmax = None

    value_ratio = n_fft // ((num_mels - 1) * 32)

    assert value_ratio % 2 == 0 or value_ratio == 1

    try:
        wav, sr = torchaudio.load(ifile)

        assert sr == 16000, f"sample rate: {sr}"

        spec = wav_to_mel(wav, n_fft, num_mels, sr, hop_size, win_size, fmin, fmax, center=False)
        spec = torch.squeeze(spec, 0) / value_ratio

        torch.save(spec, ofile)
    except:
        traceback.print_exc()
        print("Failed to process {}".format(ifile))
        return None

    return ifile


def process_data(i_dir):
    wav_fns = sorted(glob.glob(f"{i_dir}/**/*.wav", recursive=True))
    # wav_fns = wav_fns[:100]
    logging.info(f"train {len(wav_fns)}")

    p = Pool(int(os.getenv("N_PROC", os.cpu_count())))
    futures = []

    for ifile in wav_fns:
        futures.append(p.apply_async(process_item, args=(ifile,)))
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

    args = parse_args()
    data_dir = args.data_dir

    start = time()
    process_data(data_dir)
    logging.info(f"Processed data in {time() - start} seconds")

    extension = ".spec.pt"
    logging.info(f"{extension}: \t{human_readable_size(get_size_by_ext(data_dir, extension))}")
    extension = ".wav"
    logging.info(f"{extension}: \t{human_readable_size(get_size_by_ext(data_dir, extension))}")
    logging.info(f"Total: \t\t{human_readable_size(get_size(path=data_dir))}")
