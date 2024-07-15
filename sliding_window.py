import numpy as np
import pandas as pd
import time
from sklearn.metrics import f1_score
from predict import predict_anomalies  # Assuming predict_anomalies is a function that predicts anomalies for a given window of data

def extract_labels_from_sequences(anomaly_sequences, length):
    """
    Create a label array from anomaly sequences.

    Args:
    - anomaly_sequences (list of lists): List of anomaly sequences.
    - length (int): Length of the data sequence.

    Returns:
    - labels (np.array): Array of labels where 1 indicates an anomaly and 0 indicates normal.
    """
    labels = np.zeros(length)
    for seq in anomaly_sequences:
        for start, end in seq:
            labels[start:end] = 1
    return labels

def sliding_window_f1(data, anomaly_sequences, window_size=100, step_size=1):
    """
    Apply a sliding window to detect anomalies and calculate F1 scores.

    Args:
    - data (pd.DataFrame): The input data.
    - anomaly_sequences (list of lists): List of anomaly sequences.
    - window_size (int): The size of the sliding window.
    - step_size (int): The step size to move the window.

    Returns:
    - mean_f1 (float): The mean F1 score over all windows.
    - f1_scores (list): The list of F1 scores for each window.
    - total_time (float): The total time taken for processing.
    """
    length = len(data)
    labels = extract_labels_from_sequences(anomaly_sequences, length)

    start_time = time.time()
    f1_scores = []
    num_windows = (length - window_size) // step_size + 1

    for start in range(0, length - window_size + 1, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]
        window_labels = labels[start:end]

        # Predict anomalies for the current window
        predictions = predict_anomalies(window_data)

        # Calculate F1 score for the current window
        f1 = f1_score(window_labels, predictions)
        f1_scores.append(f1)

    total_time = time.time() - start_time
    mean_f1 = np.mean(f1_scores)

    return mean_f1, f1_scores, total_time

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv('msl_train.csv')
    labeled_anomalies = pd.read_csv('labeled_anomalies.csv')
    anomaly_sequences = labeled_anomalies['anomaly_sequences'].apply(eval).tolist()

    mean_f1, f1_scores, total_time = sliding_window_f1(data, anomaly_sequences)
    print(f"Mean F1 Score: {mean_f1}")
    print(f"Total Time: {total_time} seconds")
