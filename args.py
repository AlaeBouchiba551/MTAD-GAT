import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_parser():
    parser = argparse.ArgumentParser(description="MTAD-GAT")

    # Existing arguments
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--group', type=str, required=True, help='Group name')
    parser.add_argument('--lookback', type=int, required=True, help='Lookback window size')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train')
    parser.add_argument('--bs', type=int, default=256, help='Batch size')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--shuffle_dataset', type=bool, default=True, help='Whether to shuffle the dataset')
    parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use GPU for training')
    parser.add_argument('--print_every', type=int, default=1, help='How often to print training progress')
    parser.add_argument('--log_tensorboard', type=bool, default=False, help='Log to TensorBoard')
    parser.add_argument('--normalize', type=bool, default=True, help='Normalize the data')
    parser.add_argument('--spec_res', type=bool, default=False, help='Specific results')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size')
    parser.add_argument('--use_gatv2', type=bool, default=True, help='Use GATv2')
    parser.add_argument('--feat_gat_embed_dim', type=int, default=32, help='Feature GAT embedding dimension')
    parser.add_argument('--time_gat_embed_dim', type=int, default=32, help='Time GAT embedding dimension')
    parser.add_argument('--gru_n_layers', type=int, default=1, help='Number of GRU layers')
    parser.add_argument('--gru_hid_dim', type=int, default=150, help='GRU hidden dimension')
    parser.add_argument('--fc_n_layers', type=int, default=3, help='Number of fully connected layers')
    parser.add_argument('--fc_hid_dim', type=int, default=150, help='Fully connected hidden dimension')
    parser.add_argument('--recon_n_layers', type=int, default=3, help='Number of reconstruction layers')
    parser.add_argument('--recon_hid_dim', type=int, default=150, help='Reconstruction hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value')

    # Add the new arguments
    parser.add_argument('--start_index', type=int, default=0, help='Start index for the training window')
    parser.add_argument('--end_index', type=int, default=-1, help='End index for the training window')

    return parser
