import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# Load test data as

dataset_path = "filelists/madasr23_test.csv"
output_path = "/gpfs/mariana/home/darobn/datasets/madasr23/bn.tts"
data = pd.read_csv(dataset_path, sep="|")
print(data.head())


hps = utils.get_hparams_from_file("./configs/madasr23_base.json")

net_g = SynthesizerTrn(len(symbols), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("./logs/madasr23_base/G_15000.pth", net_g, None)


class MyDataset(Dataset):
    def __init__(self, dataframe, hps):
        self.data = dataframe
        self.hps = hps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid_idx = self.data["sid_idx"][idx]
        sid = self.data["sid"][idx]
        phonemes = self.data["phonemes"][idx]
        stn_tst = get_text(phonemes, self.hps)
        return sid_idx, sid, stn_tst, idx


# Initialize the dataset and data loader
dataset = MyDataset(data, hps)
data_loader = DataLoader(dataset, batch_size=1, num_workers=8)

for sid_idx, spk_id, stn_tst, i in tqdm(data_loader):
    sid_idx = int(sid_idx)
    spk_id = int(spk_id)
    i = int(i)
    stn_tst = stn_tst[0]
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([sid_idx]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][0].data.cpu()
        torchaudio.save(f"{output_path}/{spk_id}_{i}.wav", audio, hps.data.sampling_rate, bits_per_sample=hps.data.bits_per_sample)

print("Done!")
