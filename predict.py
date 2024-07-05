import argparse
import json
import datetime
import os
import torch
from args import get_parser, str2bool
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor

def model_prediction(model, data):
    # Assuming the model's forward method is used for predictions
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.float32)
        output = model(data_tensor)
    # Apply any necessary post-processing to the output to get binary predictions
    # This example assumes a threshold-based method for anomaly detection
    threshold = 0.5
    predictions = (output > threshold).int().numpy()
    return predictions

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--model_id", type=str, default=None,
                        help="ID (datetime) of pretrained model to use, '-1' for latest, '-2' for second latest, etc")
    parser.add_argument("--load_scores", type=str2bool, default=False, help="To use already computed anomaly scores")
    parser.add_argument("--save_output", type=str2bool, default=False)
    parser.add_argument("--window_size", type=int, default=1000, help="Size of the sliding window in timestamps")
    parser.add_argument("--step_size", type=int, default=1, help="Step size for sliding window")
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

    # Check that model exist
    if not os.path.isfile(f"{model_path}/model.pt"):
        raise Exception(f"<{model_path}/model.pt> does not exist.")

    # Get configs of model
    print(f'Using model from {model_path}')
    model_parser = argparse.ArgumentParser()
    model_args, unknown = model_parser.parse_known_args()
    model_args_path = f"{model_path}/config.txt"

    with open(model_args_path, "r") as f:
        model_args.__dict__ = json.load(f)

    # Load the model
    model = MTAD_GAT(**model_args.__dict__)
    model.load_state_dict(torch.load(f"{model_path}/model.pt"))

    # Load the dataset
    data = np.load(f"./data/{dataset}/test.npy")
    true_labels = np.load(f"./data/{dataset}/test_labels.npy")

    # Perform sliding window evaluation
    from eval_methods import sliding_window_evaluation
    mean_f1 = sliding_window_evaluation(data, args.window_size, args.step_size, model, model_prediction, true_labels=true_labels)

    print(f"Mean F1 Score: {mean_f1}")
