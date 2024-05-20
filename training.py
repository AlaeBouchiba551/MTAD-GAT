import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="C:\\Users\\alaeb\\mtad-gat-pytorch\\dload saver",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_recon": [],
            "val_total": [],
            "val_recon": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[1]:.5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[1]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            recon_b_losses = []

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                recons = self.model(x)

                # Adjust the target to match the output shape
                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = recon_loss

                loss.backward()
                self.optimizer.step()

                recon_b_losses.append(recon_loss.item())

            recon_b_losses = np.array(recon_b_losses)
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())
            total_epoch_loss = recon_epoch_loss

            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            if val_loader is not None:
                recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_recon"].append(recon_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if total_val_loss <= self.losses["val_total"][-1]:
                    self.save(f"model.pt")

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_recon_loss = {recon_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )

                s += f" [{epoch_time:.1f}s]"
                print(s)

        if val_loader is None:
            self.save(f"model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        self.model.eval()
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                recons = self.model(x)

                # Adjust the target to match the output shape
                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                recon_losses.append(recon_loss.item())

        recon_losses = np.array(recon_losses)
        recon_loss = np.sqrt((recon_losses ** 2).mean())
        total_loss = recon_loss

        return (recon_loss, total_loss)

    def save(self, file_name):
        os.makedirs(self.dload, exist_ok=True)  # Create directory if it doesn't exist
        PATH = os.path.join(self.dload, file_name)  # Use os.path.join to create the full file path
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)
