import json
from datetime import datetime
import torch
import torch.nn as nn
from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer

if __name__ == "__main__":
    import os
    import json
    from datetime import datetime
    import torch
    import torch.nn as nn
    import numpy as np
    from sklearn.metrics import f1_score
    import matplotlib.pyplot as plt
    from data_utils import get_data, SlidingWindowDataset, create_data_loaders
    from model import MTAD_GAT
    from trainer import Trainer
    from predictor import Predictor
    from parser import get_parser

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

    # Load data
    train_loader, val_loader, test_loader = create_data_loaders(dataset, window_size, batch_size, val_split,
                                                                shuffle_dataset)

    # Load model
    model = MTAD_GAT(window_size=window_size, n_features=train_loader.dataset[0][0].shape[1], use_cuda=use_cuda)
    if use_cuda:
        model.cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    # Trainer
    trainer = Trainer(model, optimizer, window_size, n_features=train_loader.dataset[0][0].shape[1],
                      target_dims=None, n_epochs=n_epochs, batch_size=batch_size, init_lr=init_lr,
                      forecast_criterion=nn.MSELoss(), recon_criterion=nn.MSELoss(),
                      use_cuda=use_cuda, dload="", log_dir="output/", print_every=print_every,
                      log_tensorboard=log_tensorboard, args_summary="")

    # Training
    trainer.train(train_loader, val_loader)

    # Save model
    model_id = id
    model_path = f"./output/{dataset}/{model_id}"
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.state_dict(), f"{model_path}/model.pt")

    # Load the dataset for evaluation
    data = np.load(f"./data/{dataset}/test.npy")
    true_labels = np.load(f"./data/{dataset}/test_labels.npy")

    # Perform sliding window evaluation
    from eval_methods import sliding_window_evaluation

    mean_f1 = sliding_window_evaluation(data, args.window_size, args.step_size, model, model_prediction,
                                        true_labels=true_labels)

    print(f"Mean F1 Score: {mean_f1}")
