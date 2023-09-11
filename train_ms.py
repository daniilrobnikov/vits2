import os
import tqdm
import logging
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from typing import List

import utils.task as task
from utils.hparams import get_hparams
from model.models import SynthesizerTrn
from model.discriminator import MultiPeriodDiscriminator
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate, DistributedBucketSampler
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss, kl_loss_normal
from utils.mel_processing import wav_to_mel, spec_to_mel, spectral_norm
from utils.model import slice_segments, clip_grad_value_


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8000"

    hps = get_hparams()
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = task.get_logger(hps.model_dir)
        logger.info(hps)
        task.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(train_dataset, hps.train.batch_size, [32, 300, 400, 500, 600, 700, 800, 900, 1000], num_replicas=n_gpus, rank=rank, shuffle=True)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False, batch_size=hps.train.batch_size, pin_memory=True, drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        len(train_dataset.vocab), hps.data.n_mels if hps.data.use_mel else hps.data.n_fft // 2 + 1, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = task.load_checkpoint(task.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = task.load_checkpoint(task.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
        net_g.module.mas_noise_scale = max(hps.model.mas_noise_scale - global_step * hps.model.mas_noise_scale_decay, 0.0)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets: List[torch.nn.parallel.DistributedDataParallel], optims: List[torch.optim.Optimizer], schedulers, scaler: GradScaler, loaders, logger: logging.Logger, writers):
    net_g, net_d = nets

    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        loader = train_loader
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        speakers = speakers.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (m_p_text, logs_p_text),
                (m_p_dur, logs_p_dur, z_q_dur, logs_q_dur),
                (m_p_audio, logs_p_audio, m_q_audio, logs_q_audio),
            ) = net_g(x, x_lengths, spec, spec_lengths, speakers)

            mel = spectral_norm(spec) if hps.data.use_mel else spec_to_mel(spec, hps.data.n_fft, hps.data.n_mels, hps.data.sample_rate, hps.data.f_min, hps.data.f_max)
            y_hat_mel = wav_to_mel(y_hat.squeeze(1), hps.data.n_fft, hps.data.n_mels, hps.data.sample_rate, hps.data.hop_length, hps.data.win_length, hps.data.f_min, hps.data.f_max)

            y_mel = slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y = slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                # TODO Test gain constant
                if False:
                    loss_kl_text = kl_loss_normal(m_q_text, logs_q_text, m_p_text, logs_p_text, x_mask) * hps.train.c_kl_text
                loss_kl_dur = kl_loss(z_q_dur, logs_q_dur, m_p_dur, logs_p_dur, z_mask) * hps.train.c_kl_dur
                loss_kl_audio = kl_loss_normal(m_p_audio, logs_p_audio, m_q_audio, logs_q_audio, z_mask) * hps.train.c_kl_audio

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl_dur + loss_kl_audio  # TODO + loss_kl_text
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl_dur, loss_kl_audio]  # TODO loss_kl_text
                losses_str = " ".join(f"{loss.item():.3f}" for loss in losses)
                loader.set_postfix_str(f"{losses_str}, {global_step}, {lr:.9f}")

                # scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                # scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl_dur})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                # image_dict = {
                #     "slice/mel_org": task.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                #     "slice/mel_gen": task.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                #     "all/mel": task.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                #     "all/attn": task.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
                # }
                # task.summarize(writer=writer, global_step=global_step, images=image_dict, scalars=scalar_dict, sample_rate=hps.data.sample_rate)

            # Save checkpoint on CPU to prevent GPU OOM
            if global_step % hps.train.eval_interval == 0:
                # evaluate(hps, net_g, eval_loader, writer_eval)
                task.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                task.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speakers = speakers.cuda(0)

            # remove else
            x = x[:1]
            x_lengths = x_lengths[:1]
            spec = spec[:1]
            spec_lengths = spec_lengths[:1]
            y = y[:1]
            y_lengths = y_lengths[:1]
            speakers = speakers[:1]
            break
        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)
        y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        mel = spectral_norm(spec) if hps.data.use_mel else spec_to_mel(spec, hps.data.n_fft, hps.data.n_mels, hps.data.sample_rate, hps.data.f_min, hps.data.f_max)
        y_hat_mel = wav_to_mel(y_hat.squeeze(1).float(), hps.data.n_fft, hps.data.n_mels, hps.data.sample_rate, hps.data.hop_length, hps.data.win_length, hps.data.f_min, hps.data.f_max)
    image_dict = {"gen/mel": task.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())}
    audio_dict = {"gen/audio": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update({"gt/mel": task.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    task.summarize(writer=writer_eval, global_step=global_step, images=image_dict, audios=audio_dict, sample_rate=hps.data.sample_rate)
    generator.train()


if __name__ == "__main__":
    main()
