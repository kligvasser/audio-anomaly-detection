import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import defaultdict

import models.misc
import utils.recorder
import utils.misc


class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = args.device

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=self.args.betas,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma
        )

        self.criterion = nn.L1Loss()
        self.losses = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }

        self.writer = utils.recorder.RecoderX(args.save_path)

        self.train_steps = len(train_loader.dataset) // args.batch_size
        self.eval_steps = len(val_loader.dataset) // args.batch_size

        logging.info("Training steps in epoch: {}.".format(self.train_steps))
        logging.info("Evaluating steps in epoch: {}.".format(self.eval_steps))

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch(epoch=epoch)
            self.eval_epoch(epoch=epoch)

            if epoch % self.args.save_every == 0:
                models.misc.save_model(
                    self.model,
                    os.path.join(
                        self.args.save_path,
                        "checkpoints",
                        "classifier_check_{}.pt".format(epoch),
                    ),
                )
            logging.info(
                "Epoch: {}, Train loss: {:.4f}, Val loss {:.4f}".format(
                    epoch + 1,
                    np.mean(self.losses["train"]["loss"][-self.train_steps :]),
                    np.mean(self.losses["eval"]["loss"][-self.eval_steps :]),
                )
            )

            self.writer.add_scalar(
                "epoch/loss/train",
                np.mean(self.losses["train"]["loss"][-self.train_steps :]),
                epoch,
            )
            self.writer.add_scalar(
                "epoch/loss/eval",
                np.mean(self.losses["eval"]["loss"][-self.eval_steps :]),
                epoch,
            )

        models.misc.save_model(
            self.model,
            os.path.join(
                self.args.save_path, "checkpoints", "autoencoder_check_last.pt"
            ),
        )
        models.misc.save_model_entire(
            self.model,
            os.path.join(
                self.args.save_path, "checkpoints", "autoencoder_check_last_entire.pt"
            ),
        )
        self.writer.close()

    def eval(self):
        self.eval_epoch(epoch=0)

        logging.info(
            "Evaluation: Val loss {:.4f}, Val accuracy {:.2f}, Val average-precision {:.2f}".format(
                np.mean(self.losses["eval"]["loss"][:]),
                np.mean(self.losses["eval"]["accuracy"][:]),
                self.losses["eval"]["auprc"][-1],
            )
        )

    def train_epoch(self, epoch):
        self.model.train()
        self.scheduler.step(epoch=epoch)
        for step, data in enumerate(self.train_loader):
            self.train_step(data)
            if step % self.args.print_every == 0:
                logging.info(
                    "Step: {}, Loss: {:.4f}".format(
                        step,
                        self.losses["train"]["loss"][-1],
                    )
                )

    def eval_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            for _, data in enumerate(self.val_loader):
                inputs, predictions = self.eval_step(data)

        self.writer.plot_spectograms(
            "spectograms/inputs", inputs.cpu().numpy(), self.args.sampling_rate, epoch
        )
        self.writer.plot_spectograms(
            "spectograms/predictions",
            predictions.cpu().numpy(),
            self.args.sampling_rate,
            epoch,
        )

    def train_step(self, data):
        inputs = data["input"].to(self.device)

        self.optimizer.zero_grad()

        predictions = self.model(inputs)
        loss = self.criterion(predictions, inputs)

        loss.backward()
        self.optimizer.step()

        self.losses["train"]["loss"].append(loss.item())
        self.writer.add_scalar(
            "loss/train", loss.item(), len(self.losses["train"]["loss"])
        )

    def eval_step(self, data):
        inputs = data["input"].to(self.device)

        predictions = self.model(inputs)
        loss = self.criterion(predictions, inputs)

        self.losses["eval"]["loss"].append(loss.item())
        self.writer.add_scalar(
            "loss/eval", loss.item(), len(self.losses["eval"]["loss"])
        )

        return inputs, predictions
