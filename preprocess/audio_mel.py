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
    ofile = ifile.replace(".wav", ".mel.pt")

    n_fft = 32768  # 1024 # TODO 32768 for num_mels = 513; 65536 for num_mels = 1025
    hop_size = 256  # 256
    win_size = 1024  # 1024
    num_mels = 513  # 80
    fmin = 0
    fmax = None

    value_ratio = n_fft // ((num_mels - 1) * 32)

    assert value_ratio % 2 == 0 or value_ratio == 1

    try:
        wav, sr = torchaudio.load(ifile)

        assert sr == 22050, f"sample rate: {sr}"

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


def get_total_size(directory, extension):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(extension):
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

    if total_size < 1024:
        total_size = str(total_size) + "B"
    elif total_size < (1024 * 1024):
        total_size = str(total_size // 1024) + "KB"
    elif total_size < (1024 * 1024 * 1024):
        total_size = str(total_size // (1024 * 1024)) + "MB"
    else:
        total_size = str(total_size // (1024 * 1024 * 1024)) + "GB"
    return total_size


if __name__ == "__main__":
    from time import time

    args = parse_args()
    data_dir = args.data_dir

    start = time()
    process_data(data_dir)
    logging.info(f"Processed data in {time() - start} seconds")

    extension = ".mel.pt"
    logging.info(f"{extension}: \t{get_total_size(data_dir, extension)}")
    extension = ".wav"
    logging.info(f"{extension}: \t{get_total_size(data_dir, extension)}")
