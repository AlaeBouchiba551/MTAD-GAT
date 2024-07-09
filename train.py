import os
import json
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from utils import get_data, normalize_data
from mtad_gat import MTAD_GAT
from training import Trainer
from prediction import Predictor
from args import get_parser
from torch.utils.data import DataLoader, TensorDataset
from eval_methods import sliding_window_evaluation, calculate_f1

def model_prediction(model, window_data):
    model.eval()
    with torch.no_grad():
        window_data_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0)
        forecast, _ = model(window_data_tensor)
        return forecast.squeeze(0).argmax(dim=1).numpy()

def run_experiment(train_data, test_data, window_size_min, window_size_max, window_step_size, args):
    results = []
    for window_size in range(window_size_min, window_size_max + 1, window_step_size):
        print(f'Running experiment with window size: {window_size}')

        # Set device
        device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

        # Convert data to torch.float32
        train_data_tensor = torch.tensor(train_data[0], dtype=torch.float32)
        test_data_tensor = torch.tensor(test_data[0], dtype=torch.float32)
        test_labels_tensor = torch.tensor(test_data[1], dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(train_data_tensor, train_labels), batch_size=args.bs, shuffle=args.shuffle_dataset)
        val_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), batch_size=args.bs, shuffle=args.shuffle_dataset)
        test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), batch_size=args.bs, shuffle=False)

        # Get the feature dimension from the first batch
        first_batch = next(iter(train_loader))
        n_features = first_batch[0].shape[1]

        # Define the output dimension
        out_dim = 1

        # Load model
        model = MTAD_GAT(window_size=window_size, n_features=n_features, out_dim=out_dim).to(device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

        # Trainer
        trainer = Trainer(model, optimizer, window_size, n_features=n_features,
                          target_dims=None, n_epochs=args.epochs, batch_size=args.bs, init_lr=args.init_lr,
                          forecast_criterion=nn.MSELoss(), recon_criterion=nn.MSELoss(),
                          use_cuda=args.use_cuda, dload="", log_dir="output/", print_every=args.print_every,
                          log_tensorboard=args.log_tensorboard, args_summary="")

        # Training
        trainer.fit(train_loader, val_loader)

        # Perform sliding window evaluation
        mean_f1 = sliding_window_evaluation(test_data[0], window_size, args.step_size, model, model_prediction, true_labels=test_data[1])

        results.append((window_size, mean_f1))
        print(f'Window size: {window_size}, Mean F1 Score: {mean_f1}')

    return results

if __name__ == "__main__":
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    parser = get_parser()
    parser.add_argument('--window_size_min', type=int, default=10, help='Minimum window size')
    parser.add_argument('--window_size_max', type=int, default=50, help='Maximum window size')
    parser.add_argument('--window_step_size', type=int, default=5, help='Step size for window')
    args = parser.parse_args()

    dataset = args.dataset
    window_size_min = args.window_size_min
    window_size_max = args.window_size_max
    window_step_size = args.window_step_size
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard

    # Load data
    train_data, test_data = get_data(dataset)

    # Handle the case when train_data labels are None
    if train_data[1] is None:
        train_labels = torch.zeros(len(train_data[0]), dtype=torch.long)
    else:
        train_labels = torch.tensor(train_data[1], dtype=torch.long)

    # Run experiments with different window sizes
    results = run_experiment(train_data, test_data, window_size_min, window_size_max, window_step_size, args)
    print(results)
