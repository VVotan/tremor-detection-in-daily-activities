"""Command-line entrypoint for the tremor analysis pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime

from pathlib import Path
from pprint import pformat
from typing import Any, Mapping

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from src.fft_analysis import start_fft_analysis
from src.filter_utils import lowpass, highpass
from src.hdf5_utils import load_signal
from src.visualization import TimeSeries, VisualizationConfig, Visualizer
from src.wavelet_analysis import start_wavelet_analysis
from scipy.signal import welch


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tremor Frequency Analysis")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to YAML configuration file. Example: configs/example_config.yaml",
    )
    return parser.parse_args()


def load_config(config_file: Path) -> dict[str, Any]:
    """Load YAML configuration file."""

    with open(config_file, "r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle) or {}


def _analysis_method(config: Mapping[str, Any]) -> str:
    method = str(config.get("analysis", {}).get("method", "both")).strip().lower()
    if method not in {"fft", "cwt", "both"}:
        raise ValueError("analysis.method must be one of: fft, cwt, both")
    return method


class TremorAnalysisPipeline:
    """Orchestrates loading, preprocessing, analysis, and plotting."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.input_config = dict(config.get("input", {}))
        self.signal_config = dict(config.get("signal", {}))
        self.filter_config = dict(config.get("filter", {}))
        self.analysis_config = dict(config.get("analysis", {}))
        self.cwt_config = dict(config.get("cwt", {}))
        self.fft_config = dict(config.get("fft", {}))
        self.output_config = dict(config.get("output", {}))

        self.input_path = Path(self.input_config.get("file", ""))
        self.imu = str(self.input_config.get("imu", "left_hand"))
        self.axis = str(self.signal_config.get("axis", "mag"))
        self.sampling_rate = float(self.signal_config.get("sampling_rate", 120.0))
        self.lowcut = float(self.filter_config.get("lowcut", 2.0))
        self.highcut = float(self.filter_config.get("highcut", 12.0))
        self.filter_order = int(self.filter_config.get("order", 2))
        self.nfft = int(self.fft_config.get("nfft", 2**10))
        self.min_frequency = float(self.cwt_config.get("min_frequency", 2.0))
        self.max_frequency = float(self.cwt_config.get("max_frequency", 6.0))
        self.wavelet = str(self.cwt_config.get("wavelet", "morl"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{self.output_config.get('directory', 'results')}/{self.input_path.stem}_{timestamp}")
        self.save_figures = bool(self.output_config.get("save_figures", False))

    def _to_report_lines(self) -> list[str]:
        lines = []

        for key, value in vars(self).items():
            lines.append(f"{key}: {value}")

        return lines

    def export_metadata(self, results: [str]) -> Path:
        """
        Export the metadata next to the generated plots.
        """
        output_file = self.output_dir / "metadata.txt"

        with output_file.open("w", encoding="utf-8") as f:
            f.write("=== Used Config Parameters ===\n\n")
            for line in self._to_report_lines():
                f.write(f"{line}\n")
            f.write("=== Results ===\n\n")
            for line in results:
                f.write(f"{line}\n")

        return output_file

    def run(self):
        Visualizer.configure(
            VisualizationConfig(
                save_figures=self.save_figures,
                output_dir=self.output_dir,
                show_by_default=True,
                tremor_band=(self.lowcut, self.highcut),
            )
        )

        Visualizer.begin_dashboard(
            "Tremor Analysis with dataset:" + str(self.input_path.stem)
        )


        raw_signal = load_signal(self.input_path, self.imu, self.axis, missing_policy="trim_edges")
        print(f"Picking IMU: {self.imu}")
        print(f"Loaded signal from {raw_signal.source_path}")

        preprocessing = highpass(raw_signal.values, self.sampling_rate, self.lowcut, self.filter_order)
        preprocessing = lowpass(preprocessing, self.sampling_rate, self.highcut, self.filter_order)
        preprocessed_data = preprocessing - np.mean(preprocessing)

        print(f"Applied band-pass filter: highpass={self.lowcut} Hz, lowpass={self.highcut} Hz")

        raw_signal_x = load_signal(self.input_path,self.imu, "x", missing_policy="trim_edges")
        raw_signal_y = load_signal(self.input_path,self.imu, "y", missing_policy="trim_edges")
        raw_signal_z = load_signal(self.input_path,self.imu, "z", missing_policy="trim_edges")

        sample_count = min(
            raw_signal_x.values.size,
            raw_signal_y.values.size,
            raw_signal_z.values.size,
        )
        multichannel_signal = np.vstack(
            [
                raw_signal_x.values[:sample_count],
                raw_signal_y.values[:sample_count],
                raw_signal_z.values[:sample_count],
            ]
        )
        time = np.arange(sample_count) / self.sampling_rate

        Visualizer.plot_multichannel_signal(
            multichannel_signal,
            sampling_rate=self.sampling_rate,
            channel_names=["x", "y", "z"],
            colors=["#1f77b4", "#1f7700", "#1f7777"],
            title="Acceleration (Axes: x, y, z)",
            x_label="time [s]",
            y_label="acceleration [m/s^2]",
        )

        _, _, animation = Visualizer.animate_timeseries(
            time=time,
            series=[
                TimeSeries(
                    label="x",
                    values=raw_signal_x.values[:sample_count],
                    color="#1f77b4",
                ),
                TimeSeries(
                    label="y",
                    values=raw_signal_y.values[:sample_count],
                    color="#1f7700",
                ),
                TimeSeries(
                    label="z",
                    values=raw_signal_z.values[:sample_count],
                    color="#1f7777",
                ),
            ],
            start_time=20,
            end_time=35,
            save_path="raw_data_timeseries_animation.mp4",
            ax=None,
            title="Acceleration (Axes: x, y, z)",
            x_label="time [s]",
            y_label="acceleration [m/s^2]",
            figsize=(6.5, 3.5),
            relative_time=False,
            show=False
        )

        results=[]
        fft_freqs = None
        power = None
        coefs = None
        freqs = None
        f_mean_t = None

        method = _analysis_method(self.config)
        if method in {"fft", "both"}:
            fft_freqs, power = start_fft_analysis(preprocessed_data, self.sampling_rate, self.nfft, output_dir=self.output_dir)
            results.append(str(fft_freqs))
            results.append(str(power))
        if method in {"cwt", "both"}:
            coefs, freqs, f_mean_t, _ = start_wavelet_analysis(preprocessed_data, self.wavelet, self.min_frequency, self.max_frequency,
                                             self.sampling_rate, output_dir=self.output_dir)
            results.append(str(coefs))
            results.append(str(freqs))
            results.append(str(f_mean_t))

        Visualizer.export_dashboard_plots()
        Visualizer.show_dashboard()
        if self.save_figures:
            self.export_metadata(results)

        return fft_freqs, power, coefs, freqs


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    TremorAnalysisPipeline(config).run()


if __name__ == "__main__":
    main()
