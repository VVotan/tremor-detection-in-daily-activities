# tremor-detection-in-daily-activities

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
