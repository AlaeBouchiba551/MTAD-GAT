class Trainer:
    """Trainer class for MTAD-GAT model.

    :param model: MTAD-GAT model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

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
        self.target_dims = target_dims
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

        self.out_dim = n_features if target_dims is None else len(target_dims) if isinstance(target_dims, list) else 1

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
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """
        for epoch in range(self.n_epochs):
            start_time = time.time()
            self.model.train()
            total_loss = 0
            total_forecast_loss = 0
            total_recon_loss = 0
            for x in train_loader:
                x = x.to(self.device)
                self.optimizer.zero_grad()
                forecast, recon = self.model(x)
                forecast_loss = self.forecast_criterion(forecast, x[:, -1, :self.out_dim])
                recon_loss = self.recon_criterion(recon, x)
                loss = forecast_loss + recon_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_forecast_loss += forecast_loss.item()
                total_recon_loss += recon_loss.item()

            n_batches = len(train_loader)
            avg_loss = total_loss / n_batches
            avg_forecast_loss = total_forecast_loss / n_batches
            avg_recon_loss = total_recon_loss / n_batches

            self.losses["train_total"].append(avg_loss)
            self.losses["train_forecast"].append(avg_forecast_loss)
            self.losses["train_recon"].append(avg_recon_loss)
            self.epoch_times.append(time.time() - start_time)

            if val_loader is not None:
                val_loss, val_forecast_loss, val_recon_loss = self.evaluate(val_loader)
                self.losses["val_total"].append(val_loss)
                self.losses["val_forecast"].append(val_forecast_loss)
                self.losses["val_recon"].append(val_recon_loss)

            if self.log_tensorboard:
                self.writer.add_scalar("Loss/train_total", avg_loss, epoch)
                self.writer.add_scalar("Loss/train_forecast", avg_forecast_loss, epoch)
                self.writer.add_scalar("Loss/train_recon", avg_recon_loss, epoch)
                if val_loader is not None:
                    self.writer.add_scalar("Loss/val_total", val_loss, epoch)
                    self.writer.add_scalar("Loss/val_forecast", val_forecast_loss, epoch)
                    self.writer.add_scalar("Loss/val_recon", val_recon_loss, epoch)

            if (epoch + 1) % self.print_every == 0:
                print(
                    f"Epoch {epoch+1}/{self.n_epochs}, "
                    f"Train Loss: {avg_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}" if val_loader is not None else ""
                )

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        total_forecast_loss = 0
        total_recon_loss = 0
        with torch.no_grad():
            for x in data_loader:
                x = x.to(self.device)
                forecast, recon = self.model(x)
                forecast_loss = self.forecast_criterion(forecast, x[:, -1, :self.out_dim])
                recon_loss = self.recon_criterion(recon, x)
                loss = forecast_loss + recon_loss
                total_loss += loss.item()
                total_forecast_loss += forecast_loss.item()
                total_recon_loss += recon_loss.item()
        n_batches = len(data_loader)
        return total_loss / n_batches, total_forecast_loss / n_batches, total_recon_loss / n_batches
