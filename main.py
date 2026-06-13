"""Command-line entrypoint for the tremor analysis pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime

from pathlib import Path
from typing import Any, Mapping
import yaml
from src.fft_analysis import start_fft_analysis
from src.filter_utils import lowpass, highpass
from src.hdf5_utils import load_signal
from src.visualization import VisualizationConfig, Visualizer
from src.wavelet_analysis import start_wavelet_analysis



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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"{self.output_config.get('directory', 'results')}/{self.input_path.stem}_{timestamp}")
        self.save_figures = bool(self.output_config.get("save_figures", False))

    def run(self):
        Visualizer.configure(
            VisualizationConfig(
                save_figures=self.save_figures,
                output_dir=self.output_dir,
                show_by_default=True,
                tremor_band=(self.lowcut, self.highcut),
            )
        )
        raw_signal = load_signal(self.input_path, self.imu, self.axis)
        print(f"Picking IMU: {self.imu}")
        print(f"Loaded signal from {raw_signal.source_path}")

        preprocessing = lowpass(raw_signal.values, self.sampling_rate, self.lowcut, self.filter_order)
        preprocessing = highpass(preprocessing.data, self.sampling_rate,self.highcut, self.filter_order)
        print(f"Applied band-pass filter: highpass={self.lowcut} Hz, lowpass={self.highcut} Hz")

        Visualizer.compare_signals(
            raw_signal.values,
            preprocessing.data,
            self.sampling_rate,
            reference_label="Raw signal",
            comparison_label="Filtered signal",
            difference_label="Residual",
            title="Raw vs filtered signal",
        )

        fft_result = None
        wavelet = None

        method = _analysis_method(self.config)
        if method in {"fft", "both"}:
            fft_result = start_fft_analysis(preprocessing.data, self.sampling_rate, self.nfft)

        if method in {"cwt", "both"}:
            wavelet = start_wavelet_analysis(preprocessing.data, self.min_frequency, self.max_frequency,
                                             self.sampling_rate)



        return raw_signal, preprocessing, fft_result, wavelet


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    TremorAnalysisPipeline(config).run()


if __name__ == "__main__":
    main()
