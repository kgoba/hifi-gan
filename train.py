import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os, sys
import time
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint, count_parameters

torch.backends.cudnn.benchmark = True


def train_step(models, optimizers, batch, h):
    generator, msd, mpd = models
    optim_g, optim_d = optimizers
    x, y, y_mel = batch

    y = y.unsqueeze(1)

    # Generator
    mpd.requires_grad_(False)
    msd.requires_grad_(False)

    y_g_hat = generator(x)
    y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                    h.fmin, h.fmax_for_loss)

    _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
    _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
    loss_gen = generator_loss(y_df_hat_g) + generator_loss(y_ds_hat_g)
    loss_fm = feature_loss(fmap_f_r, fmap_f_g) + feature_loss(fmap_s_r, fmap_s_g)
    loss_mel = F.l1_loss(y_mel, y_g_hat_mel)
    loss_gen_all = 1.0 * loss_gen + 2.0 * loss_fm + 5 * loss_mel

    optim_g.zero_grad()
    loss_gen_all.backward()
    optim_g.step()

    # Discriminator
    mpd.requires_grad_(True)
    msd.requires_grad_(True)

    y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
    loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g)
    loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
    loss_disc_all = loss_disc_s + loss_disc_f

    optim_d.zero_grad()
    loss_disc_all.backward()
    optim_d.step()

    return {
        "gen_all": loss_gen_all.item(),
        "gen": loss_gen.item(),
        "fm": loss_fm.item(),
        "mel": loss_mel.item(),
        "disc_all": loss_disc_all.item()
    }


def train(rank, a, h, device):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    if device == "cuda":
        device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        print(f"Generator parameters: {count_parameters(generator)}")
        print(f"Discriminator parameters: {count_parameters(mpd) + count_parameters(msd)}")
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("Checkpoints directory: ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device="cpu",
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, None, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, shuffle=False,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    loss_mel_hist = []
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        pbar_epoch = tqdm(train_loader, total=len(trainset) // h.batch_size)
        for batch in pbar_epoch:
            pbar_epoch.set_description(f"Step {steps}")
            x, y, _, y_mel = batch
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))

            loss = train_step([generator, mpd, msd], [optim_g, optim_d], [x, y, y_mel], h)

            if rank == 0:
                # STDOUT logging
                # if steps % a.stdout_interval == 0:
                    # with torch.no_grad():
                        # mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                loss_mel_hist = [loss["mel"]] + loss_mel_hist[:49]
                loss_mel_avg = np.mean(loss_mel_hist)

                pbar_epoch.set_postfix_str(f'L_d: {loss["disc_all"]:.2f}, L_g: {loss["gen"]:.2f}, L_f: {loss["fm"]:.2f}, L_mel: {loss_mel_avg:.3f}, L: {loss["gen_all"]:.2f}')

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss["gen_all"], steps)
                    sw.add_scalars("training/mel_spec_error",
                                    {
                                        "mel_avg": loss_mel_avg,
                                        "gen": loss["gen"],
                                        "dis": loss["disc_all"]
                                    }, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                # y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                #                              h.sampling_rate, h.hop_size, h.win_size,
                                #                              h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_g_hat_mel.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = "cuda"

        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device {device}")

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h, device)


if __name__ == '__main__':
    main()
