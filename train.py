import json
from datetime import datetime
import torch.nn as nn
import os
import torch

from args import get_parser
from utils import *
from mtad_gat import ReconstructionModel
from training import Trainer


if __name__ == "__main__":
    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    window_size = args.lookback
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

    # Load data
    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    # Initialize the reconstruction model
    model = ReconstructionModel(
        window_size=window_size,
        in_dim=n_features,
        hid_dim=args.recon_hid_dim,
        out_dim=out_dim,
        n_layers=args.recon_n_layers,
        dropout=args.dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=target_dims,
        n_epochs=n_epochs,
        batch_size=batch_size,
        init_lr=init_lr,
        recon_criterion=recon_criterion,
        use_cuda=use_cuda,
        dload="",  # Assuming this argument is optional
        log_dir=log_dir,
        print_every=print_every,
        log_tensorboard=log_tensorboard,
        args_summary=args_summary
    )

    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test reconstruction loss: {test_loss:.5f}")

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
