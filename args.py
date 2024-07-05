import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    parser = argparse.ArgumentParser(description="MTAD-GAT Argument Parser")

    # Add your arguments here
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use")
    parser.add_argument("--group", type=str, help="Group for the SMD dataset")
    parser.add_argument("--lookback", type=int, default=100, help="Window size for the model")
    parser.add_argument("--spec_res", type=bool, default=False, help="Specific resolution")
    parser.add_argument("--normalize", type=str2bool, default=True, help="Normalize the data")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=256, help="Batch size")
    parser.add_argument("--init_lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True, help="Shuffle the dataset")
    parser.add_argument("--use_cuda", type=str2bool, default=True, help="Use CUDA for training")
    parser.add_argument("--print_every", type=int, default=1, help="Print progress every n epochs")
    parser.add_argument("--log_tensorboard", type=str2bool, default=True, help="Log to TensorBoard")
    parser.add_argument("--kernel_size", type=int, default=7, help="Kernel size for convolutional layers")
    parser.add_argument("--use_gatv2", type=str2bool, default=True, help="Use GATv2")
    parser.add_argument("--feat_gat_embed_dim", type=int, default=150, help="Embedding dimension for feature GAT")
    parser.add_argument("--time_gat_embed_dim", type=int, default=150, help="Embedding dimension for temporal GAT")
    parser.add_argument("--gru_n_layers", type=int, default=1, help="Number of GRU layers")
    parser.add_argument("--gru_hid_dim", type=int, default=150, help="Hidden dimension for GRU layers")
    parser.add_argument("--fc_n_layers", type=int, default=1, help="Number of layers in the forecasting model")
    parser.add_argument("--fc_hid_dim", type=int, default=150, help="Hidden dimension for the forecasting model")
    parser.add_argument("--recon_n_layers", type=int, default=1, help="Number of layers in the reconstruction model")
    parser.add_argument("--recon_hid_dim", type=int, default=150, help="Hidden dimension for the reconstruction model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--alpha", type=float, default=0.2, help="Negative slope for leaky ReLU")
    parser.add_argument("--level", type=float, default=None, help="Level for POT")
    parser.add_argument("--q", type=float, default=None, help="q for POT")
    parser.add_argument("--scale_scores", type=str2bool, default=False, help="Scale anomaly scores")
    parser.add_argument("--dynamic_pot", type=str2bool, default=False, help="Use dynamic POT")
    parser.add_argument("--use_mov_av", type=str2bool, default=False, help="Use moving average for anomaly scores")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for score calculation")

    # Add window_size and step_size arguments
    parser.add_argument("--window_size", type=int, default=1000, help="Size of the sliding window in timestamps")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for sliding window")

    return parser
