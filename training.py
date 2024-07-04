import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd  # Add this import statement

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,  # Add target_dims parameter
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims  # Initialize target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            print(f"args_summary (in Trainer): {args_summary} (type: {type(args_summary)})")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[2]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[2]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds, recons = self.model(x)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = forecast_loss + recon_loss

                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
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
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
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

        forecast_losses = []
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds, recons = self.model(x)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_loss = np.sqrt((recon_losses ** 2).mean())

        total_loss = forecast_loss + recon_loss

        return forecast_loss, recon_loss, total_loss

    def save(self, file_name):
        PATH = os.path.join(self.dload, file_name)
        if os.path.exists(self.dload):
            pass
        else:
            os.makedirs(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)

    def load_results(self, result_path, model_id):
        train_output_path = os.path.join(result_path, 'train_output.pkl')
        test_output_path = os.path.join(result_path, 'test_output.pkl')
        if os.path.exists(train_output_path):
            self.train_output = pd.read_pickle(train_output_path)
        if os.path.exists(test_output_path):
            self.test_output = pd.read_pickle(test_output_path)

    def get_train_output(self):
        return self.train_output

    def get_test_output(self):
        return self.test_outputimport os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.metrics import f1_score  # Add this import
import matplotlib.pyplot as plt  # Add this import

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,  # Add target_dims parameter
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=True,
        dload="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):
        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims  # Initialize target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.forecast_criterion = forecast_criterion
        self.recon_criterion = recon_criterion
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": [],
            "train_forecast": [],
            "train_recon": [],
            "val_total": [],
            "val_forecast": [],
            "val_recon": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            print(f"args_summary (in Trainer): {args_summary} (type: {type(args_summary)})")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[2]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[2]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            forecast_b_losses = []
            recon_b_losses = []

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                preds, recons = self.model(x)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))
                loss = forecast_loss + recon_loss

                loss.backward()
                self.optimizer.step()

                forecast_b_losses.append(forecast_loss.item())
                recon_b_losses.append(recon_loss.item())

            forecast_b_losses = np.array(forecast_b_losses)
            recon_b_losses = np.array(recon_b_losses)

            forecast_epoch_loss = np.sqrt((forecast_b_losses ** 2).mean())
            recon_epoch_loss = np.sqrt((recon_b_losses ** 2).mean())

            total_epoch_loss = forecast_epoch_loss + recon_epoch_loss

            self.losses["train_forecast"].append(forecast_epoch_loss)
            self.losses["train_recon"].append(recon_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            if val_loader is not None:
                forecast_val_loss, recon_val_loss, total_val_loss = self.evaluate(val_loader)
                self.losses["val_forecast"].append(forecast_val_loss)
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
                    f"forecast_loss = {forecast_epoch_loss:.5f}, "
                    f"recon_loss = {recon_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )

                if val_loader is not None:
                    s += (
                        f" ---- val_forecast_loss = {forecast_val_loss:.5f}, "
                        f"val_recon_loss = {recon_val_loss:.5f}, "
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

        forecast_losses = []
        recon_losses = []

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds, recons = self.model(x)

                forecast_loss = torch.sqrt(self.forecast_criterion(y, preds))
                recon_loss = torch.sqrt(self.recon_criterion(x, recons))

                forecast_losses.append(forecast_loss.item())
                recon_losses.append(recon_loss.item())

        forecast_losses = np.array(forecast_losses)
        recon_losses = np.array(recon_losses)

        forecast_loss = np.sqrt((forecast_losses ** 2).mean())
        recon_loss = np.sqrt((recon_losses ** 2).mean())

        total_loss = forecast_loss + recon_loss

        return forecast_loss, recon_loss, total_loss

    def save(self, file_name):
        PATH = os.path.join(self.dload, file_name)
        if os.path.exists(self.dload):
            pass
        else:
            os.makedirs(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)

    def load_results(self, result_path, model_id):
        train_output_path = os.path.join(result_path, 'train_output.pkl')
        test_output_path = os.path.join(result_path, 'test_output.pkl')
        if os.path.exists(train_output_path):
            self.train_output = pd.read_pickle(train_output_path)
        if os.path.exists(test_output_path):
            self.test_output = pd.read_pickle(test_output_path)

    def get_train_output(self):
        return self.train_output

    def get_test_output(self):
        return self.test_output

    def sliding_window_evaluation(self, x_test, y_test):
        self.model.eval()
        f1_scores = []

        with torch.no_grad():
            for i in range(len(x_test) - self.window_size + 1):
                window_data = x_test[i:i + self.window_size]
                window_label = y_test[i + self.window_size - 1] if y_test is not None else None

                # Assuming your model's predict method returns binary anomaly scores
                preds, _ = self.model(window_data.unsqueeze(0).to(self.device))
                preds = preds.squeeze(0).cpu().numpy()

                if window_label is not None:
                    # Here we assume that preds contains anomaly scores and window_label contains the true label
                    f1 = f1_score([window_label], preds > 0.5)  # Adjust threshold as needed
                    f1_scores.append(f1)

        mean_f1_score = np.mean(f1_scores)
        print(f"Mean F1 Score: {mean_f1_score:.5f}")

        # Save F1 scores
        np.save(os.path.join(self.dload, "f1_scores.npy"), f1_scores)

        # Visualize F1 scores
        plt.plot(f1_scores)
        plt.xlabel('Timestamp')
        plt.ylabel('F1 Score')
        plt.title('F1 Score per Timestamp')
        plt.savefig(os.path.join(self.dload, "f1_scores.png"))
        plt.show()

        return mean_f1_score

