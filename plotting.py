from utils import get_data_dim, get_series_color, get_y_height
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import plotly as py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import cufflinks as cf
cf.go_offline()

class Plotter:
    def __init__(self, result_path, model_id='-1'):
        self.result_path = result_path
        self.model_id = model_id
        self.train_output = None
        self.test_output = None
        self.labels_available = True
        self.pred_cols = None
        self._load_results()
        self.train_output["timestamp"] = self.train_output.index
        self.test_output["timestamp"] = self.test_output.index
        self.trainer = Trainer()  # Initialize your Trainer class instance

        config_path = f"{self.result_path}/config.txt"
        with open(config_path) as f:
            self.lookback = json.load(f)["lookback"]

        if "SMD" in self.result_path:
            self.pred_cols = [f"feat_{i}" for i in range(self.trainer.get_data_dim("machine"))]
        elif "SMAP" in self.result_path or "MSL" in self.result_path:
            self.pred_cols = ["feat_1"]

    def _load_results(self):
        # Load results using your Trainer class instance
        if self.model_id.startswith('-'):
            self.trainer.load_results(self.result_path, self.model_id)
            self.train_output = self.trainer.train_output
            self.test_output = self.trainer.test_output
        else:
            # Handle loading when model_id is not negative
            pass

    def result_summary(self):
        # Extract results summary using your Trainer class instance
        self.trainer.result_summary()

    def get_anomaly_sequences(self, values):
        # Move this method to be a static method or method of Trainer class
        return self.trainer.get_anomaly_sequences(values)

    def plot_all_features(self, plot_train=False, plot_errors=True, plot_feature_anom=True, start=None, end=None):
        # Use trainer instance methods to get data and sequences
        test_copy = self.trainer.get_test_output()

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            test_copy = test_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            test_copy = test_copy.iloc[: end - start, :]

        plot_data = [test_copy]

        if plot_train:
            train_copy = self.trainer.get_train_output()
            plot_data.append(train_copy)

        for nr, data_copy in enumerate(plot_data):
            is_test = nr == 0

            for i in range(len(self.pred_cols)):
                if f"Forecast_{i}" not in data_copy.columns:
                    continue

                plot_values = {
                    "timestamp": data_copy["timestamp"].values,
                    "y_forecast": data_copy[f"Forecast_{i}"].values,
                    "y_recon": data_copy[f"Recon_{i}"].values,
                    "y_true": data_copy[f"True_{i}"].values,
                    "errors": data_copy[f"A_Score_{i}"].values,
                    "threshold": data_copy[f"Thresh_{i}"].values
                }

                anomaly_sequences = {
                    "pred": self.trainer.get_anomaly_sequences(data_copy[f"A_Pred_{i}"].values),
                    "true": self.trainer.get_anomaly_sequences(data_copy["A_True_Global"].values),
                }

                if is_test and start is not None:
                    anomaly_sequences['pred'] = [[s + start, e + start] for [s, e] in anomaly_sequences['pred']]
                    anomaly_sequences['true'] = [[s + start, e + start] for [s, e] in anomaly_sequences['true']]

                y_min = 1.1 * plot_values["y_true"].min()
                y_max = 1.1 * plot_values["y_true"].max()
                e_max = 1.5 * plot_values["errors"].max()

                y_shapes = self.create_shapes(anomaly_sequences["pred"], "predicted", y_min, y_max, plot_values,
                                              is_test=is_test)
                e_shapes = self.create_shapes(anomaly_sequences["pred"], "predicted", 0, e_max, plot_values,
                                              is_test=is_test)
                if self.labels_available and ('SMAP' in self.result_path or 'MSL' in self.result_path):
                    y_shapes += self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values,
                                                   is_test=is_test)
                    e_shapes += self.create_shapes(anomaly_sequences["true"], "true", 0, e_max, plot_values,
                                                   is_test=is_test)

                y_df = pd.DataFrame(
                    {
                        "timestamp": plot_values["timestamp"].reshape(-1, ),
                        "y_forecast": plot_values["y_forecast"].reshape(-1, ),
                        "y_recon": plot_values["y_recon"].reshape(-1, ),
                        "y_true": plot_values["y_true"].reshape(-1, )
                    }
                )

                e_df = pd.DataFrame(
                    {
                        "timestamp": plot_values["timestamp"],
                        "e_s": plot_values["errors"].reshape(-1, ),
                        "threshold": plot_values["threshold"].reshape(-1, ),
                    }
                )

                data_type = "Test data" if is_test else "Train data"
                y_layout = {
                    "title": f"{data_type} | Forecast & reconstruction vs true value for {self.pred_cols[i] if self.pred_cols is not None else ''} ",
                    "showlegend": True,
                    "height": 400,
                    "width": 1100,
                    "shapes": y_shapes if plot_feature_anom else []
                }

                e_layout = {
                    "title": f"{data_type} | Error for {self.pred_cols[i] if self.pred_cols is not None else ''}",
                    "height": 400,
                    "width": 1100,
                    "shapes": e_shapes if plot_feature_anom else []
                }

                lines = [
                    go.Scatter(
                        x=y_df["timestamp"],
                        y=y_df["y_true"],
                        line_color="rgb(0, 204, 150, 0.5)",
                        name="y_true",
                        line=dict(width=2)),
                    go.Scatter(
                        x=y_df["timestamp"],
                        y=y_df["y_forecast"],
                        line_color="rgb(255, 127, 14, 1)",
                        name="y_forecast",
                        line=dict(width=2)),
                    go.Scatter(
                        x=y_df["timestamp"],
                        y=y_df["y_recon"],
                        line_color="rgb(31, 119, 180, 1)",
                        name="y_recon",
                        line=dict(width=2)),
                ]

                fig = go.Figure(data=lines, layout=y_layout)
                py.offline.iplot(fig)

                e_lines = [
                    go.Scatter(
                        x=e_df["timestamp"],
                        y=e_df["e_s"],
                        name="Error",
                        line=dict(color="red", width=1))]
                if plot_feature_anom:
                    e_lines.append(
                        go.Scatter(
                            x=e_df["timestamp"],
                            y=e_df["threshold"],
                            name="Threshold",
                            line=dict(color="black", width=1, dash="dash")))

                if plot_errors:
                    e_fig = go.Figure(data=e_lines, layout=e_layout)
                    py.offline.iplot(e_fig)

    def plot_all_features_together(self, plot_train=False, plot_errors=True, plot_feature_anom=True, start=None, end=None):
        # Use trainer instance methods to get data and sequences
        test_copy = self.trainer.get_test_output()

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            test_copy = test_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            test_copy = test_copy.iloc[: end - start, :]

        plot_data = [test_copy]

        if plot_train:
            train_copy = self.trainer.get_train_output()
            plot_data.append(train_copy)

        for nr, data_copy in enumerate(plot_data):
            is_test = nr == 0

            fig = make_subplots(rows=len(self.pred_cols), cols=1, shared_xaxes=True)

            for row, i in enumerate(self.pred_cols, 1):
                if f"Forecast_{i}" not in data_copy.columns:
                    continue

                plot_values = {
                    "timestamp": data_copy["timestamp"].values,
                    "y_forecast": data_copy[f"Forecast_{i}"].values,
                    "y_recon": data_copy[f"Recon_{i}"].values,
                    "y_true": data_copy[f"True_{i}"].values,
                    "errors": data_copy[f"A_Score_{i}"].values,
                    "threshold": data_copy[f"Thresh_{i}"].values
                }

                anomaly_sequences = {
                    "pred": self.trainer.get_anomaly_sequences(data_copy[f"A_Pred_{i}"].values),
                    "true": self.trainer.get_anomaly_sequences(data_copy["A_True_Global"].values),
                }

                if is_test and start is not None:
                    anomaly_sequences['pred'] = [[s + start, e + start] for [s, e] in anomaly_sequences['pred']]
                    anomaly_sequences['true'] = [[s + start, e + start] for [s, e] in anomaly_sequences['true']]

                y_min = 1.1 * plot_values["y_true"].min()
                y_max = 1.1 * plot_values["y_true"].max()

                y_shapes = self.create_shapes(anomaly_sequences["pred"], "predicted", y_min, y_max, plot_values, is_test=is_test)
                if self.labels_available and ('SMAP' in self.result_path or 'MSL' in self.result_path):
                    y_shapes += self.create_shapes(anomaly_sequences["true"], "true", y_min, y_max, plot_values, is_test=is_test)

                fig.add_trace(go.Scatter(x=plot_values["timestamp"], y=plot_values["y_true"], mode='lines', name=f'y_true_{i}'), row=row, col=1)
                fig.add_trace(go.Scatter(x=plot_values["timestamp"], y=plot_values["y_forecast"], mode='lines', name=f'y_forecast_{i}'), row=row, col=1)
                fig.add_trace(go.Scatter(x=plot_values["timestamp"], y=plot_values["y_recon"], mode='lines', name=f'y_recon_{i}'), row=row, col=1)

                fig.update_yaxes(range=[y_min, y_max], row=row, col=1)

            fig.update_layout(height=400 * len(self.pred_cols), width=1100, title="All Features | Forecast & reconstruction vs true value")
            py.offline.iplot(fig)

    # The rest of your Plotter class methods...

# Usage example
plotter = Plotter(result_path='path/to/results', model_id='-1')
plotter.plot_all_features_together(plot_train=False, plot_errors=True, plot_feature_anom=True, start=None, end=None)
