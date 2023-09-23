import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os, sys, glob
import argparse
import json
import random
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from env import AttrDict, build_env
from hifigan.meldataset import MelDataset
from hifigan.models import Generator, feature_loss, generator_loss, discriminator_loss
from hifigan.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from hifigan.utils import (
    plot_spectrogram,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    count_parameters,
)
from hifigan.audio import AudioFrontendConfig, AudioFrontend

# import torchmetrics
import pytorch_lightning as pl


class TrainerTask(pl.LightningModule):
    def __init__(
        self,
        generator,
        discriminators,
        audio_frontend,
        lr=0.0002,
        beta1=0.8,
        beta2=0.99,
        sample_rate=22050,
    ):
        super().__init__()
        self.G = generator
        self.D_list = discriminators
        self.audio_frontend = audio_frontend
        # self.save_hyperparameters(ignore=["generator", "discriminators", "audio_frontend"])
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.sample_rate = sample_rate

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        x, y, _, y_mel = batch
        y = y.unsqueeze(1)

        # Generate
        y_g_hat = self.G(x)

        # Optimize Discriminator
        self.G.requires_grad_(False)
        self.D_list.requires_grad_(True)
        loss_disc = 0
        for D in self.D_list:
            y_d_hat_r, _ = D(y)
            y_d_hat_g, _ = D(y_g_hat.detach())
            loss_disc += discriminator_loss(y_d_hat_r, y_d_hat_g)

        d_opt.zero_grad()
        self.manual_backward(loss_disc)
        d_opt.step()

        # Optimize Generator
        self.G.requires_grad_(True)
        self.D_list.requires_grad_(False)
        _, y_g_hat_mel = self.audio_frontend.encode(y_g_hat.squeeze(1).cpu())
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel.to(y_mel.device))
        loss_gen_all = 15 * loss_mel

        loss_gen, loss_fm = 0, 0
        for D in self.D_list:
            _, fmap_r = D(y)
            y_d_hat_g, fmap_g = D(y_g_hat)
            loss_gen += generator_loss(y_d_hat_g)
            loss_fm += feature_loss(fmap_r, fmap_g)

        loss_gen_all += loss_gen + 2 * loss_fm

        g_opt.zero_grad()
        self.manual_backward(loss_gen_all)
        g_opt.step()

        self.log_dict(
            {
                "L_d": loss_disc,
                "L_g": loss_gen,
                "L_fm": loss_fm,
                "L_g_all": loss_gen_all,
                "L_mel": loss_mel,
            },
            prog_bar=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y, _, y_mel = batch
        y = y.unsqueeze(1)

        # Generate
        y_g_hat = self.G(x)

        _, y_g_hat_mel = self.audio_frontend.encode(y_g_hat.squeeze(1).cpu())
        val_loss = F.l1_loss(y_mel.cpu(), y_g_hat_mel)

        x, y, y_mel, y_g_hat_mel = x.cpu(), y.cpu(), y_mel.cpu(), y_g_hat_mel.cpu()

        if batch_idx < 5:
            # Get tensorboard logger
            tb_logger = None
            for logger in self.loggers:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    tb_logger = logger.experiment
                    break

            if tb_logger != None:
                # tb_logger.add_audio(f"gt/y_{batch_idx}", y[0], self.global_step, self.sample_rate)
                # tb_logger.add_figure(
                #     f"gt/y_spec_{batch_idx}", plot_spectrogram(x[0]), self.global_step
                # )
                tb_logger.add_audio(
                    f"generated/y_hat_{batch_idx}",
                    y_g_hat[0],
                    self.global_step,
                    self.sample_rate,
                )
                tb_logger.add_figure(
                    f"generated/y_hat_spec_{batch_idx}",
                    plot_spectrogram(y_g_hat_mel[0]),
                    self.global_step,
                )

        self.log("L_val", val_loss, batch_size=1)

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(self.G.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        d_opt = torch.optim.AdamW(
            self.D_list.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2),
        )
        # Return:
        # * List or Tuple of optimizers.
        # * Two lists - The first list has multiple optimizers, and the second has multiple LR schedulers (or multiple lr_scheduler_config).
        return g_opt, d_opt


def load_legacy_checkpoint(a, generator, mpd, msd, device):
    # os.makedirs(a.checkpoint_path, exist_ok=True)
    # print("Checkpoints directory: ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, "g_")
        cp_do = scan_checkpoint(a.checkpoint_path, "do_")
        # cp_do = os.path.join(a.checkpoint_path, "do_current")
        print(f"Found checkpoints {cp_g} and {cp_do}")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]


def save_scripted_generator(generator, path="generator_traced.pt"):
    x_sample = torch.zeros((1, 80, 32))
    trace = torch.jit.trace(generator, x_sample)
    # generator.remove_weight_norm()
    # script = torch.jit.script(generator)
    x_test = torch.zeros((1, 80, 64))
    trace(x_test)
    print(f"Saving scripted generator to {path}")
    trace.save(path)


def train(a, h, device):
    generator = Generator(h)
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    discriminators = nn.ModuleList([mpd, msd])

    load_legacy_checkpoint(a, generator, mpd, msd, device)

    audio_config = AudioFrontendConfig()
    audio_config.sample_rate = h.sampling_rate
    audio_config.num_mels = h.num_mels
    audio_config.hop_length = h.hop_size
    audio_config.win_length = h.win_size
    audio_config.fmin = h.fmin
    audio_config.fmax = h.fmax
    audio_frontend = AudioFrontend(audio_config)

    # training_filelist, validation_filelist = get_dataset_filelist(a)
    audio_filelist = sorted(glob.glob("*.wav", root_dir=a.input_wavs_dir))
    random.seed(42)
    random.shuffle(audio_filelist)
    train_filelist = audio_filelist[25:]
    test_filelist = audio_filelist[:25]

    train_loader = DataLoader(
        MelDataset(train_filelist, a.input_wavs_dir, h.segment_size, audio_frontend),
        num_workers=4,  # h.num_workers,
        shuffle=True,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    validation_loader = DataLoader(
        MelDataset(test_filelist, a.input_wavs_dir, None, audio_frontend),
        num_workers=4,
        shuffle=False,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    logger = pl.loggers.tensorboard.TensorBoardLogger(save_dir="lightning_logs", version=0)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=a.checkpoint_path,
        # filename="cp-last",
        save_last=True,
        # every_n_train_steps=10,
        every_n_epochs=1,
    )
    trainer = pl.Trainer(
        logger=logger,
        # callbacks=[pl.callbacks.OnExceptionCheckpoint("."), checkpoint_callback],
        callbacks=[
            checkpoint_callback,
            pl.callbacks.LambdaCallback(
                on_load_checkpoint=lambda trainer, pl_module, checkpoint: save_scripted_generator(
                    pl_module.G
                )
            ),
        ],
        # callbacks=[
        #     pl.callbacks.LambdaCallback(
        #         on_train_epoch_end=lambda *args: trainer.save_checkpoint("last.ckpt")
        #     )
        # ],
        accelerator="auto",
        # enable_checkpointing=True,
        limit_train_batches=100,
        # limit_val_batches=10,
    )

    trainer_task = TrainerTask(generator, discriminators, audio_frontend, lr=h.learning_rate)

    trainer.fit(
        model=trainer_task,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        # ckpt_path="checkpoints/cp-last-v5.ckpt",
        ckpt_path="last",
    )


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="")
    parser.add_argument("--input_wavs_dir", default="LJSpeech-1.1/wavs")
    parser.add_argument("--input_mels_dir", default="ft_dataset")
    parser.add_argument("--input_training_file", default="LJSpeech-1.1/training.txt")
    parser.add_argument("--input_validation_file", default="LJSpeech-1.1/validation.txt")
    parser.add_argument("--checkpoint_path", default="checkpoints")
    parser.add_argument("--training_epochs", default=3100, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=5000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=1000, type=int)
    parser.add_argument("--fine_tuning", default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = "cuda"

        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print("Batch size per GPU :", h.batch_size)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device {device}")

    train(a, h, device)


if __name__ == "__main__":
    main()
