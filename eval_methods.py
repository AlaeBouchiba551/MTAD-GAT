import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
from spot import SPOT, dSPOT
from sklearn.metrics import f1_score


def sliding_window_anomaly_detection(time_series, window_size, step_size, detection_function, *args, **kwargs):
    """
    Apply sliding window to time series data and perform anomaly detection.

    Parameters:
    - time_series: The time series data (2D array where each row is a timestamp)
    - window_size: The size of the sliding window
    - step_size: The step size for the sliding window
    - detection_function: The anomaly detection function to apply
    - *args, **kwargs: Additional arguments for the detection function

    Returns:
    - results: A list of results from the detection function for each window
    """
    num_windows = (len(time_series) - window_size) // step_size + 1
    results = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window_data = time_series[start:end]
        result = detection_function(window_data, *args, **kwargs)
        results.append(result)

    return results


def sliding_window_evaluation(time_series, window_size, step_size, model, detection_function, *args, **kwargs):
    num_windows = (len(time_series) - window_size) // step_size + 1
    f1_scores = []

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window_data = time_series[start:end]

        # Get predictions
        predictions = detection_function(model, window_data, *args, **kwargs)

        # Assuming that the true labels are provided in the kwargs
        true_labels = kwargs.get('true_labels')[start:end]

        # Calculate F1 score for the current window
        f1 = f1_score(true_labels, predictions, average='macro')
        f1_scores.append(f1)

    mean_f1 = np.mean(f1_scores)
    return mean_f1

# Example usage:
# mean_f1 = sliding_window_evaluation(time_series, window_size, step_size, model, detection_function, true_labels=true_labels)
