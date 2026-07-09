# tremor-detection-in-daily-activities
This project applies signal processing techniques to data of tremorous motion in order to extract information about the tremor, in particular its frequency. 

The movement data was recorded with Inertial Measurement Units (IMUs) and includes simple movements and more complex real-world acitvites (e.g. drinking and pouring water into a glass) with emulated tremors of different frequency. For data analyis first a bandpass filter is applied to isolate tremor-relevant frequencies. Then a Fast Fourier Transform (FFT) for global frequency analysis and a Continuous Wavelet Transform (CWT) for time-frequency visualization is performed. Further a mean frequency estimation at each time point is calculated for robust tracking of the tremor frequency.

This project was conducted by Odin Göggerle, Linus Zeidler and Natalie Kraus as part of the seminar "Motion in Human and Machine" at the University of Tübingen.

## How to run the Analysis
Before execution, you have to write a config file. 
See `configs/example_config.yaml` to create your own.

If you have your configuration file, you can run the program with 
`python main.py --config configs/example_config.yaml`

## How to use the Visualizer

Import it to your python file
`from src.visualization import VisualizationConfig, Visualizer`

### Build a Visualizer Object once for the hole Program.
**That is already happening in main.py!**
```
        Visualizer.configure(
            VisualizationConfig(
                save_figures=self.save_figures,
                output_dir=self.output_dir,
                show_by_default=True,
                tremor_band=(self.lowcut, self.highcut),
            )
        )
```
### Examples how to plot a Signal
```
        Visualizer.plot_time_series(
            signal=raw_signal.values,
            sampling_rate=self.sampling_rate,
            title="Accelerometer X Axis",
            label="Acc X",
            y_label="Acceleration [m/s²]",
            x_label="Time [s]",
            show=True,
        )

        Visualizer.compare_signals(
            raw_signal.values,
            preprocessing.data,
            self.sampling_rate,
            reference_label="Raw signal",
            comparison_label="Filtered signal",
            difference_label="Residual",
            title="Raw vs filtered signal",
        )
```
