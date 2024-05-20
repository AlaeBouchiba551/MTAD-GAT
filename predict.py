import argparse
import json
import datetime
import torch
import os

from args import get_parser, str2bool
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor

if __name__ == "__main__":

    parser = get_parser()
    parser.add_argument("--model_id", type=str, default=None,
                        help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")
    parser.add_argument("--load_scores", type=str2bool, default=False, help="To use already computed anomaly scores")
    parser.add_argument("--save_output", type=str2bool, default=False)
    args = parser.parse_args()
    print(args)

    dataset = args.dataset
    if args.model_id is None:
        if dataset == 'SMD':
            dir_path = f"./output/{dataset}/{args.group}"
        else:
            dir_path = f"./output/{dataset}"
        dir_content = os.listdir(dir_path)
        subfolders = [subf for subf in dir_content if os.path.isdir(f"{dir_path}/{subf}") and subf != "logs"]
        date_times = [datetime.datetime.strptime(subf, '%d%m%Y_%H%M%S') for subf in subfolders]
        date_times.sort()
        model_datetime = date_times[-1]
        model_id = model_datetime.strftime('%d%m%Y_%H%M%S')

    else:
        model_id = args.model_id

    if dataset == "SMD":
        model_path = f"./output/{dataset}/{args.group}/{model_id}"
    elif dataset in ['MSL', 'SMAP']:
        model_path = f"./output/{dataset}/{model_id}"
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    # Check that the model exists
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get the model configuration
    print(f'Using model from {model_path}')
    model_args_path = f"{model_path}/config.txt"
    with open(model_args_path, "r") as f:
        model_args = json.load(f)
    window_size = model_args['lookback']

    # Load data
    if dataset == "SMD":
        (x_train, _), (x_test, y_test) = get_data(f"machine-{model_args['group'][0]}-{model_args['group'][2:]}", normalize=model_args['normalize'])
    else:
        (x_train, _), (x_test, y_test) = get_data(args.dataset, normalize=model_args['normalize'])

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]

    target_dims = get_target_dims(args.dataset)
    if target_dims is None:
        out_dim = n_features
    elif isinstance(target_dims, int):
        out_dim = 1
    else:
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, model_args['bs'], model_args['val_split'], model_args['shuffle_dataset'], test_dataset=test_dataset
    )

    # Initialize the MTAD-GAT model
    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=model_args['kernel_size'],
        use_gatv2=model_args['use_gatv2'],
        feat_gat_embed_dim=model_args['feat_gat_embed_dim'],
        time_gat_embed_dim=model_args['time_gat_embed_dim'],
        gru_n_layers=model_args['gru_n_layers'],
        gru_hid_dim=model_args['gru_hid_dim'],
        recon_n_layers=model_args['recon_n_layers'],
        recon_hid_dim=model_args['recon_hid_dim'],
        dropout=model_args['dropout'],
        alpha=model_args['alpha']
    )

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    load(model, f"{model_path}/model.pt", device=device)
    model.to(device)

    # Specify prediction arguments
    prediction_args = {
        'dataset': args.dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": model_args['level'],
        "q": model_args['q'],
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": model_args['reg_level'],
        "save_path": f"{model_path}",
    }

    # Predict anomalies
    label = y_test[window_size:] if y_test is not None else None
    predictor = Predictor(model, window_size, n_features, prediction_args)
    predictor.predict_anomalies(x_train, x_test, label, load_scores=args.load_scores, save_output=args.save_output)
