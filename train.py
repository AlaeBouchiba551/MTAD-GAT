import os
import json
from datetime import datetime
import torch
import torch.nn as nn
from args import get_parser
from utils import get_data, SlidingWindowDataset, create_data_loaders
from training import Trainer
from mtad_gat import MTAD_GAT

if __name__ == "__main__":
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

    # Initialize
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
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

    # Set device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    # Load data
    train_data, test_data = get_data(dataset)

    # Handle the case when train_data labels are None
    if train_data[1] is None:
        train_labels = torch.zeros(len(train_data[0]), dtype=torch.long)
    else:
        train_labels = torch.tensor(train_data[1], dtype=torch.long)

    # Convert data to torch.float32
    train_data_tensor = torch.tensor(train_data[0], dtype=torch.float32)
    test_data_tensor = torch.tensor(test_data[0], dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_data[1], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_data_tensor, train_labels), batch_size=batch_size,
                              shuffle=shuffle_dataset)
    val_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), batch_size=batch_size,
                            shuffle=shuffle_dataset)
    test_loader = DataLoader(TensorDataset(test_data_tensor, test_labels_tensor), batch_size=batch_size, shuffle=False)

    # Get the feature dimension from the first batch
    first_batch = next(iter(train_loader))
    n_features = first_batch[0].shape[1]

    # Define the output dimension
    out_dim = 1

    # Load model
    model = MTAD_GAT(window_size=window_size, n_features=n_features, out_dim=out_dim).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # Trainer
    trainer = Trainer(model, optimizer, window_size, n_features=n_features,
                      target_dims=None, n_epochs=n_epochs, batch_size=batch_size, init_lr=init_lr,
                      forecast_criterion=nn.MSELoss(), recon_criterion=nn.MSELoss(),
                      use_cuda=use_cuda, dload="", log_dir="output/", print_every=print_every,
                      log_tensorboard=log_tensorboard, args_summary="")

    # Training
    trainer.fit(train_loader, val_loader)

    # Save model
    model_id = id
    model_path = f"./output/{dataset}/{model_id}"
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), f"{model_path}/model.pt")

    # Perform sliding window evaluation
    from eval_methods import sliding_window_evaluation

    mean_f1 = sliding_window_evaluation(test_data[0], args.window_size, args.step_size, model, model_prediction,
                                        true_labels=test_data[1])

    print(f"Mean F1 Score: {mean_f1}")
