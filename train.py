import json
from datetime import datetime
import torch.nn as nn
import os
import torch

from args import get_parser
from utils import get_data, get_loaders, get_target_dims
from mtad_gat import MTAD_GAT
from training import Trainer

if __name__ == "__main__":

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
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

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

    # Setting target_dims to 1 to focus on the second feature
    target_dims = 1

    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")

    model = MTAD_GAT(
        n_features=n_features,
        window_size=window_size,
        out_dim=out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        window_size=window_size,
        n_features=n_features,
        target_dims=target_dims,
        n_epochs=n_epochs,
        batch_size=batch_size,
        init_lr=init_lr,
        forecast_criterion=nn.MSELoss(),
        recon_criterion=nn.MSELoss(),
        use_cuda=use_cuda,
        dload=save_path,
        log_dir=log_dir,
        print_every=print_every,
        log_tensorboard=log_tensorboard,
        args_summary=args_summary,
    )

    # Prepare data loaders
    train_loader, val_loader = get_loaders(x_train, val_split, batch_size, shuffle_dataset)

    trainer.fit(train_loader, val_loader)
