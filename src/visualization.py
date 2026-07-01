"""Shared visualization layer for tremor analysis.

"""

from __future__ import annotations

from contextlib import contextmanager
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure


ArrayLike1D = Sequence[float] | Any
ArrayLike2D = Sequence[Sequence[float]] | Any




def _as_1d_array(values: ArrayLike1D, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _as_2d_array(values: ArrayLike2D, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {array.shape}")
    if 0 in array.shape:
        raise ValueError(f"{name} must not be empty")
    return array


def _coerce_tuple(value: Any, *, name: str) -> tuple[float, float]:
    try:
        first, second = value
    except Exception as exc:  # pragma: no cover - defensive normalization
        raise TypeError(f"{name} must be an iterable with two values") from exc
    return float(first), float(second)


def _normalize_config_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] = dict(mapping)

    visualization_section = raw.get("visualization")
    if isinstance(visualization_section, Mapping):
        raw.update(visualization_section)

    output_section = raw.get("output")
    if isinstance(output_section, Mapping):
        raw.setdefault("output_dir", output_section.get("directory"))
        raw.setdefault("save_figures", output_section.get("save_figures"))

    aliases = {
        "figsize": "figure_size",
        "save_dir": "output_dir",
        "cmap": "cwt_cmap",
        "band": "tremor_band",
    }
    valid_fields = {field.name for field in fields(VisualizationConfig)}
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        field_name = aliases.get(key, key)
        if field_name in valid_fields:
            normalized[field_name] = value

    if "output_dir" in normalized and normalized["output_dir"] is not None:
        normalized["output_dir"] = Path(normalized["output_dir"])
    if "figure_size" in normalized and normalized["figure_size"] is not None:
        normalized["figure_size"] = _coerce_tuple(normalized["figure_size"], name="figure_size")
    if "tremor_band" in normalized and normalized["tremor_band"] is not None:
        normalized["tremor_band"] = _coerce_tuple(normalized["tremor_band"], name="tremor_band")

    return normalized


@dataclass(slots=True)
class DashboardEntry:
    renderer: Any
    kwargs: dict[str, Any]

    def render(
        self,
        *,
        ax: Axes | None = None,
        show: bool = False,
    ):
        payload = dict(self.kwargs)
        payload["ax"] = ax
        payload["show"] = show
        return self.renderer(**payload)


@dataclass(frozen=True, slots=True)
class TimeSeries:
    """Label and samples for a single plotted time series."""

    label: str
    values: ArrayLike1D
    color: str | None = None
    line_style: str = "-"


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Immutable settings shared by all plots."""

    style: str = "seaborn-v0_8-whitegrid"
    figure_size: tuple[float, float] = (11.0, 4.0)
    dpi: int = 160
    font_size: int = 11
    title_size: int = 13
    label_size: int = 11
    tick_size: int = 10
    line_width: float = 1.8
    grid_alpha: float = 0.25
    signal_color: str = "#1f77b4"
    filtered_color: str = "#d62728"
    spectrum_color: str = "#1f77b4"
    peak_color: str = "#ff7f0e"
    band_color: str = "#2ca02c"
    cwt_cmap: str = "magma"
    band_alpha: float = 0.12
    save_figures: bool = True
    output_dir: Path | None = None
    default_format: str = "png"
    transparent: bool = False
    tremor_band: tuple[float, float] | None = (4.0, 12.0)
    show_legend: bool = True
    show_grid: bool = True
    show_by_default: bool = False
    tight_layout: bool = True

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "VisualizationConfig":
        if not mapping:
            return cls()
        return replace(cls(), **_normalize_config_mapping(mapping))

    def merged(self, **changes: Any) -> "VisualizationConfig":
        normalized = _normalize_config_mapping(changes)
        return replace(self, **normalized)

    def rc_params(self) -> dict[str, Any]:
        return {
            "figure.figsize": self.figure_size,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
            "font.size": self.font_size,
            "axes.titlesize": self.title_size,
            "axes.labelsize": self.label_size,
            "xtick.labelsize": self.tick_size,
            "ytick.labelsize": self.tick_size,
            "lines.linewidth": self.line_width,
            "axes.grid": self.show_grid,
            "grid.alpha": self.grid_alpha,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "savefig.transparent": self.transparent,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }


class Visualizer:
    """Class-level plotting facade with shared configuration."""

    _config: ClassVar[VisualizationConfig] = VisualizationConfig()
    _dashboard_active: ClassVar[bool] = False
    _dashboard_title: ClassVar[str | None] = None
    _dashboard_entries: ClassVar[list[DashboardEntry]] = []

    @classmethod
    def configure(
        cls,
        config: VisualizationConfig | Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> VisualizationConfig:
        """Update the shared plotting configuration."""

        if config is None:
            new_config = cls._config
        elif isinstance(config, VisualizationConfig):
            new_config = config
        else:
            new_config = cls._config.merged(**_normalize_config_mapping(config))

        if overrides:
            new_config = new_config.merged(**overrides)

        cls._config = new_config
        return cls._config


    @classmethod
    def begin_dashboard(cls, title: str | None = None) -> None:
        cls._dashboard_active = True
        cls._dashboard_title = title
        cls._dashboard_entries.clear()

    @classmethod
    def end_dashboard(cls) -> None:
        cls._dashboard_active = False
        cls._dashboard_title = None
        cls._dashboard_entries.clear()

    @classmethod
    def clear_dashboard(cls) -> None:
        cls._dashboard_entries.clear()

    @classmethod
    def _register_dashboard_plot( cls, renderer: Any, kwargs: dict[str, Any]) -> None:
        cls._dashboard_entries.append(
            DashboardEntry(renderer=renderer, kwargs=kwargs)
        )


    @classmethod
    def _dashboard_or_continue(cls, renderer: Any, kwargs: dict[str, Any]) -> bool:
        if cls._dashboard_active and kwargs.get("ax") is None:
            payload = dict(kwargs)
            payload.pop("cls", None)
            cls._register_dashboard_plot(renderer, payload)
            return True
        return False

    @classmethod
    def show_dashboard(cls, *, columns: int = 2, show: bool = True):
        if not cls._dashboard_entries:
            raise ValueError("Dashboard is empty")

        plot_count = len(cls._dashboard_entries)
        rows = math.ceil(plot_count / columns)

        with cls.style_context():
            fig, axes = plt.subplots(
                rows,
                columns,
                figsize=(
                    columns * cls._config.figure_size[0],
                    rows * cls._config.figure_size[1],
                ),
                constrained_layout=True,
            )

            axes = np.atleast_1d(axes).flatten()

            for ax, entry in zip(axes, cls._dashboard_entries):
                entry.render(ax=ax, show=False)

            for ax in axes[plot_count:]:
                ax.set_visible(False)

            if cls._dashboard_title:
                fig.suptitle(
                    cls._dashboard_title,
                    fontsize=cls._config.title_size + 2,
                )

        if show:
            plt.show()

        return fig, axes


    @classmethod
    def plot_signal(
        cls,
        signal: ArrayLike1D,
        sampling_rate: float,
        *,
        constant: float=None,
        constant_label: str = None,
        ax: Axes | None = None,
        title: str | None = None,
        label: str | None = None,
        y_label: str = "Amplitude",
        x_label: str = "Time [s]",
        color: str | None = None,
        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot a single time-domain signal."""

        if cls._dashboard_or_continue(
            cls.plot_signal,
            locals(),
        ):
            return None, None


        samples = _as_1d_array(signal, name="signal")
        time = np.arange(samples.size) / float(sampling_rate)

        with cls.style_context():
            fig, axis, owned_figure = cls._prepare_axis(ax=ax)
            axis.plot(
                time,
                samples,
                color=color or cls._config.signal_color,
                linewidth=cls._config.line_width,
                label=label,
            )
            # Add constant in graph if handed as parameter
            if constant is not None:
                axis.axhline(
                    y=constant,
                    color=cls._config.peak_color,
                    linestyle="--",
                    linewidth=cls._config.line_width,
                    label=constant_label,
                )

            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            if title:
                axis.set_title(title)
            if label:
                axis.legend(loc="upper right")
            if cls._config.show_grid:
                axis.grid(True, alpha=cls._config.grid_alpha)

        cls._finalize_figure(fig, save_path=title, show=show, owned_figure=owned_figure)
        return fig, axis

    @classmethod
    def plot_multichannel_signal(
        cls,
        signals: ArrayLike2D,
        sampling_rate: float,
        *,
        channel_names: Sequence[str] | None = None,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str = "Time [s]",
        y_label: str = "Amplitude",
        colors: Sequence[str] | None = None,
        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot multiple channels on a shared time axis."""

        if cls._dashboard_or_continue(
            cls.plot_multichannel_signal,
            locals(),
        ):
            return None, None

        multichannel = _as_2d_array(signals, name="signals")
        channel_count, sample_count = multichannel.shape
        if channel_names is not None and len(channel_names) != channel_count:
            raise ValueError("channel_names must match the number of channels")
        if colors is not None and len(colors) < channel_count:
            raise ValueError("colors must provide at least one color per channel")

        time = np.arange(sample_count) / float(sampling_rate)
        labels = (
            list(channel_names)
            if channel_names is not None
            else [f"Channel {index + 1}" for index in range(channel_count)]
        )

        with cls.style_context():
            fig, axis, owned_figure = cls._prepare_axis(ax=ax)
            for index, channel in enumerate(multichannel):
                line_kwargs: dict[str, Any] = {"linewidth": cls._config.line_width}
                if colors is not None:
                    line_kwargs["color"] = colors[index]
                axis.plot(time, channel, label=labels[index], **line_kwargs)

            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            if title:
                axis.set_title(title)
            if cls._config.show_grid:
                axis.grid(True, alpha=cls._config.grid_alpha)
            if cls._config.show_legend:
                axis.legend(loc="upper right")

        cls._finalize_figure(fig, save_path=title, show=show, owned_figure=owned_figure)
        return fig, axis

    @classmethod
    def plot_spectrum(
        cls,
        frequencies: ArrayLike1D,
        power: ArrayLike1D,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str = "Frequency [Hz]",
        y_label: str = "Power",
        x_limits: tuple[float, float] | None = None,
        y_limits: tuple[float, float] | None = None,
        tremor_band: tuple[float, float] | None = None,
        color: str | None = None,
        line_style: str = "-",
        log_scale: bool = False,
        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot a frequency spectrum."""

        if cls._dashboard_or_continue(
            cls.plot_spectrum,
            locals(),
        ):
            return None, None

        frequency_values = _as_1d_array(frequencies, name="frequencies")
        power_values = _as_1d_array(power, name="power")
        if frequency_values.size != power_values.size:
            raise ValueError("frequencies and power must have the same length")

        with cls.style_context():
            fig, axis, owned_figure = cls._prepare_axis(ax=ax)
            axis.plot(
                frequency_values,
                power_values,
                color=color or cls._config.spectrum_color,
                linestyle=line_style,
                linewidth=cls._config.line_width,
            )

            band = tremor_band if tremor_band is not None else cls._config.tremor_band
            if band is not None:
                lower, upper = band
                axis.axvspan(
                    lower,
                    upper,
                    color=cls._config.band_color,
                    alpha=cls._config.band_alpha,
                )

            if x_limits is not None:
                axis.set_xlim(*x_limits)
            if y_limits is not None:
                axis.set_ylim(*y_limits)
            if log_scale:
                axis.set_yscale("log")

            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            if title:
                axis.set_title(title)
            if cls._config.show_grid:
                axis.grid(True, which="both", alpha=cls._config.grid_alpha)

        cls._finalize_figure(fig, save_path=title, show=show, owned_figure=owned_figure)
        return fig, axis

    @classmethod
    def plot_spectrum_peaks(
        cls,
        frequencies: ArrayLike1D,
        power: ArrayLike1D,
        *,
        peak_indices: Sequence[int] | None = None,
        peak_count: int = 3,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str = "Frequency [Hz]",
        y_label: str = "Power",
        x_limits: tuple[float, float] | None = None,
        y_limits: tuple[float, float] | None = None,
        tremor_band: tuple[float, float] | None = None,
        annotate: bool = True,
        peak_label: str = "Peak",
        color: str | None = None,
        peak_color: str | None = None,
        line_style: str = "-",
        log_scale: bool = False,
        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot a spectrum and mark its strongest peaks."""

        if cls._dashboard_or_continue(
            cls.plot_spectrum_peaks,
            locals(),
        ):
            return None, None

        frequency_values = _as_1d_array(frequencies, name="frequencies")
        power_values = _as_1d_array(power, name="power")
        if frequency_values.size != power_values.size:
            raise ValueError("frequencies and power must have the same length")

        with cls.style_context():
            fig, axis, owned_figure = cls._prepare_axis(ax=ax)
            axis.plot(
                frequency_values,
                power_values,
                color=color or cls._config.spectrum_color,
                linestyle=line_style,
                linewidth=cls._config.line_width,
            )

            band = tremor_band if tremor_band is not None else cls._config.tremor_band
            if band is not None:
                lower, upper = band
                axis.axvspan(
                    lower,
                    upper,
                    color=cls._config.band_color,
                    alpha=cls._config.band_alpha,
                )

            if peak_indices is None:
                selected_count = max(0, min(int(peak_count), power_values.size))
                if selected_count == 0:
                    selected_indices = np.array([], dtype=int)
                else:
                    selected_indices = np.argsort(power_values)[::-1][:selected_count]
            else:
                selected_indices = np.asarray(peak_indices, dtype=int)

            point_color = peak_color or cls._config.peak_color
            for rank, index in enumerate(selected_indices, start=1):
                frequency = float(frequency_values[index])
                value = float(power_values[index])
                axis.scatter([frequency], [value], color=point_color, zorder=3)
                if annotate:
                    axis.annotate(
                        f"{peak_label} {rank}: {frequency:.2f} Hz",
                        xy=(frequency, value),
                        xytext=(10, 10 + 12 * (rank - 1)),
                        textcoords="offset points",
                        arrowprops={"arrowstyle": "->", "color": point_color},
                    )

            if x_limits is not None:
                axis.set_xlim(*x_limits)
            if y_limits is not None:
                axis.set_ylim(*y_limits)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            if title:
                axis.set_title(title)
            if log_scale:
                axis.set_yscale("log")
            if cls._config.show_grid:
                axis.grid(True, which="both", alpha=cls._config.grid_alpha)

        cls._finalize_figure(fig, save_path=title, show=show, owned_figure=owned_figure)
        return fig, axis

    @classmethod
    def plot_heatmap(
        cls,
        values: ArrayLike2D,
        *,
        x_values: ArrayLike1D | None = None,
        y_values: ArrayLike1D | None = None,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str = "X",
        y_label: str = "Y",
        colorbar_label: str = "Magnitude",
        cmap: str | None = None,
        x_limits: tuple[float, float] | None = None,
        y_limits: tuple[float, float] | None = None,
        band: tuple[float, float] | None = None,

        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot a generic 2D heatmap."""

        if cls._dashboard_or_continue(
            cls.plot_heatmap,
            locals(),
        ):
            return None, None

        heatmap = _as_2d_array(values, name="values")
        if x_values is None:
            x_coords = np.arange(heatmap.shape[1], dtype=float)
        else:
            x_coords = _as_1d_array(x_values, name="x_values")
        if y_values is None:
            y_coords = np.arange(heatmap.shape[0], dtype=float)
        else:
            y_coords = _as_1d_array(y_values, name="y_values")
        if heatmap.shape != (y_coords.size, x_coords.size):
            raise ValueError("values must match the shape implied by x_values and y_values")

        with cls.style_context():
            fig, axis, owned_figure = cls._prepare_axis(ax=ax)
            mesh = axis.pcolormesh(
                x_coords,
                y_coords,
                heatmap,
                shading="auto",
                cmap=cmap or cls._config.cwt_cmap,
            )
            colorbar = fig.colorbar(mesh, ax=axis)
            colorbar.set_label(colorbar_label)
            if band is not None:
                lower, upper = band
                axis.axhspan(lower, upper, color=cls._config.band_color, alpha=cls._config.band_alpha)
            if x_limits is not None:
                axis.set_xlim(*x_limits)
            if y_limits is not None:
                axis.set_ylim(*y_limits)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            if title:
                axis.set_title(title)
            if cls._config.show_grid:
                axis.grid(True, alpha=cls._config.grid_alpha)

        cls._finalize_figure(fig, save_path=title, show=show, owned_figure=owned_figure)
        return fig, axis

    @classmethod
    def compare_signals(
        cls,
        reference_signal: ArrayLike1D,
        comparison_signal: ArrayLike1D,
        sampling_rate: float,
        *,
        reference_label: str = "Reference signal",
        comparison_label: str = "Comparison signal",
        difference_label: str = "Difference",
        title: str | None = None,
        show: bool | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Compare two time-domain signals in a stacked layout."""

        reference = _as_1d_array(reference_signal, name="reference_signal")
        comparison = _as_1d_array(comparison_signal, name="comparison_signal")
        if reference.size != comparison.size:
            raise ValueError("reference_signal and comparison_signal must have the same length")

        time = np.arange(reference.size) / float(sampling_rate)
        difference = reference - comparison

        with cls.style_context():
            fig, axes = plt.subplots(
                3,
                1,
                sharex=True,
                figsize=(cls._config.figure_size[0], cls._config.figure_size[1] * 1.8),
                constrained_layout=True,
            )
            axes = np.asarray(axes)

            axes[0].plot(time, reference, color=cls._config.signal_color, linewidth=cls._config.line_width, label=reference_label)
            axes[1].plot(time, comparison, color=cls._config.filtered_color, linewidth=cls._config.line_width, label=comparison_label)
            axes[2].plot(time, difference, color=cls._config.band_color, linewidth=cls._config.line_width, label=difference_label)

            for axis in axes:
                axis.set_ylabel("Amplitude")
                if cls._config.show_grid:
                    axis.grid(True, alpha=cls._config.grid_alpha)
                axis.legend(loc="upper right")

            axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
            axes[2].set_xlabel("Time [s]")
            if title:
                fig.suptitle(title)
        cls._finalize_figure(fig, save_path=title, show=show, owned_figure=True)
        return fig, axes

    @classmethod
    def compare_spectra(
        cls,
        frequencies: ArrayLike1D,
        reference_power: ArrayLike1D,
        comparison_power: ArrayLike1D,
        *,
        reference_label: str = "Reference spectrum",
        comparison_label: str = "Comparison spectrum",
        difference_label: str = "Difference spectrum",
        title: str | None = None,
        tremor_band: tuple[float, float] | None = None,
        log_scale: bool = False,

        show: bool | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Compare two spectra with an overlay and a difference panel."""

        frequency_values = _as_1d_array(frequencies, name="frequencies")
        reference_values = _as_1d_array(reference_power, name="reference_power")
        comparison_values = _as_1d_array(comparison_power, name="comparison_power")
        if not (frequency_values.size == reference_values.size == comparison_values.size):
            raise ValueError("frequencies, reference_power, and comparison_power must have the same length")

        difference = reference_values - comparison_values

        with cls.style_context():
            fig, axes = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(cls._config.figure_size[0], cls._config.figure_size[1] * 1.4),
                constrained_layout=True,
            )
            axes = np.asarray(axes)

            axes[0].plot(
                frequency_values,
                reference_values,
                color=cls._config.signal_color,
                linewidth=cls._config.line_width,
                label=reference_label,
            )
            axes[0].plot(
                frequency_values,
                comparison_values,
                color=cls._config.filtered_color,
                linewidth=cls._config.line_width,
                label=comparison_label,
            )
            band = tremor_band if tremor_band is not None else cls._config.tremor_band
            if band is not None:
                lower, upper = band
                axes[0].axvspan(lower, upper, color=cls._config.band_color, alpha=cls._config.band_alpha)
            if log_scale:
                axes[0].set_yscale("log")
            axes[0].set_ylabel("Power")
            axes[0].legend(loc="upper right")
            if cls._config.show_grid:
                axes[0].grid(True, which="both", alpha=cls._config.grid_alpha)

            axes[1].plot(
                frequency_values,
                difference,
                color=cls._config.peak_color,
                linewidth=cls._config.line_width,
                label=difference_label,
            )
            axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
            axes[1].set_xlabel("Frequency [Hz]")
            axes[1].set_ylabel("Difference")
            axes[1].legend(loc="upper right")
            if cls._config.show_grid:
                axes[1].grid(True, alpha=cls._config.grid_alpha)

            if title:
                fig.suptitle(title)

        cls._finalize_figure(fig, save_path=title, show=show, owned_figure=True)
        return fig, axes

    @classmethod
    @contextmanager
    def style_context(cls):
        """Apply the shared style only for the enclosed plotting code."""

        config = cls._config
        with plt.style.context(config.style), matplotlib.rc_context(config.rc_params()):
            yield

    @classmethod
    def _prepare_axis(
        cls,
        ax: Axes | None = None,
        *,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, Axes, bool]:
        if ax is not None:
            return ax.figure, ax, False

        fig, created_ax = plt.subplots(
            figsize=figsize or cls._config.figure_size,
            constrained_layout=True,
        )
        return fig, created_ax, True

    @classmethod
    def _finalize_figure(
        cls,
        fig: Figure,
        *,
        save_path: Path | str | None = None,
        show: bool | None = None,
        close: bool | None = None,
        owned_figure: bool = False,
    ) -> Figure:
        config = cls._config
        if config.tight_layout and not fig.get_constrained_layout():
            fig.tight_layout()

        if save_path is not None and config.save_figures and owned_figure:
            cls.save_figure(fig, save_path)

        should_show = config.show_by_default if show is None else show
        if should_show:
            plt.show()

        if close is None:
            close = owned_figure and not should_show
        if close:
            plt.close(fig)

        return fig

    @classmethod
    def resolve_save_path(
        cls,
        save_path: Path | str,
        *,
        default_suffix: str | None = None,
    ) -> Path:
        """Resolve a save path against the configured output directory."""

        path = Path(save_path)
        if not path.is_absolute() and cls._config.output_dir is not None:
            path = cls._config.output_dir / path
        if path.suffix == "":
            suffix = default_suffix or f".{cls._config.default_format}"
            path = path.with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def save_figure(cls, fig: Figure, save_path: Path | str) -> Path:
        """Persist a figure using the shared save settings."""

        path = cls.resolve_save_path(save_path)
        fig.savefig(
            path,
            dpi=cls._config.dpi,
            bbox_inches="tight",
            transparent=cls._config.transparent,
        )
        return path

    @classmethod
    def save_animation(
        cls,
        animation: FuncAnimation,
        save_path: Path | str,
        *,
        fps: float | None = None,
        dpi: int | None = None,
        codec: str = "libx264",
        extra_args: Sequence[str] | None = None,
    ) -> Path:
        """Persist an animation as a video file."""

        path = cls.resolve_save_path(save_path, default_suffix=".mp4")
        save_fps = float(fps) if fps is not None else 30.0

        if path.suffix.lower() == ".mp4":
            if not FFMpegWriter.isAvailable():
                raise RuntimeError(
                    "Saving MP4 animations requires ffmpeg. Install ffmpeg or choose another format."
                )
            writer = FFMpegWriter(
                fps=save_fps,
                codec=codec,
                extra_args=list(extra_args) if extra_args is not None else ["-pix_fmt", "yuv420p"],
            )
        else:
            writer = None

        animation.save(
            path,
            writer=writer,
            dpi=cls._config.dpi if dpi is None else dpi,
        )
        return path

    @classmethod
    def plot_time_series(
        cls,
        signal: ArrayLike1D,
        sampling_rate: float,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        label: str | None = None,
        constant: float = None,
        constant_label: str = None,
        y_label: str = "Amplitude",
        x_label: str = "Time [s]",
        color: str | None = None,
        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        return cls.plot_signal(
            signal,
            sampling_rate,
            ax=ax,
            title=title,
            label=label,
            constant=constant,
            constant_label=constant_label,
            y_label=y_label,
            x_label=x_label,
            color=color,
            show=show,
        )

    @classmethod
    def animate_timeseries(
        cls,
        time: ArrayLike1D,
        series: Sequence[TimeSeries],
        *,
        start_time: float,
        end_time: float,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str = "Time [s]",
        y_label: str = "Acceleration [m/s^2]",
        relative_time: bool = True,
        time_window_seconds: float | None = None,
        figsize: Sequence[float] | None = None,
        interval_ms: float | None = None,
        save_path: Path | str | None = None,
        fps: float | None = None,
        show: bool | None = None,
    ) -> tuple[Figure, Axes, FuncAnimation]:
        """Animate one or more time series over a selected window.

        Use ``time_window_seconds`` to keep the visible x-axis short and
        slide-friendly. When it is shorter than the selected data window,
        the plot uses a rolling window so the animation stays compact.
        """

        time_values = _as_1d_array(time, name="time")
        if not np.all(np.isfinite(time_values)):
            raise ValueError("time must contain only finite values")
        if np.any(np.diff(time_values) < 0):
            raise ValueError("time must be sorted in non-decreasing order")
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            raise ValueError("start_time and end_time must be finite numbers")
        if start_time >= end_time:
            raise ValueError("start_time must be smaller than end_time")
        if time_window_seconds is not None:
            time_window_seconds = float(time_window_seconds)
            if not np.isfinite(time_window_seconds) or time_window_seconds <= 0:
                raise ValueError("time_window_seconds must be a positive finite number")
        figsize_tuple = _coerce_tuple(figsize, name="figsize") if figsize is not None else None

        series_items = list(series)
        if not series_items:
            raise ValueError("series must not be empty")

        sampled_series: list[np.ndarray] = []
        labels: list[str] = []
        colors: list[str | None] = []
        line_styles: list[str] = []

        for index, item in enumerate(series_items):
            try:
                label = item.label
                values = item.values
            except AttributeError as exc:  # pragma: no cover - defensive normalization
                raise TypeError(
                    "series items must provide 'label' and 'values' attributes"
                ) from exc

            samples = _as_1d_array(values, name=f"series[{index}].values")
            if samples.size != time_values.size:
                raise ValueError(
                    "Each series must have the same number of samples as time"
                )

            sampled_series.append(samples)
            labels.append(label)
            colors.append(getattr(item, "color", None))
            line_styles.append(getattr(item, "line_style", "-"))

        window_start = max(float(start_time), float(time_values[0]))
        window_end = min(float(end_time), float(time_values[-1]))
        if window_start >= window_end:
            raise ValueError("The requested time window does not overlap the data")

        start_index = int(np.searchsorted(time_values, window_start, side="left"))
        end_index = int(np.searchsorted(time_values, window_end, side="right"))
        if start_index >= end_index:
            raise ValueError("No samples fall within the requested time window")

        window_time = time_values[start_index:end_index]
        if relative_time:
            display_time = window_time - window_time[0]
            x_start = 0.0
            x_end = float(display_time[-1])
        else:
            display_time = window_time
            x_start = float(display_time[0])
            x_end = float(display_time[-1])

        selected_span = x_end - x_start
        if selected_span <= 0:
            def frame_xlim(_: float) -> tuple[float, float]:
                return x_start - 0.5, x_end + 0.5
        else:
            if time_window_seconds is None:
                x_window_span = selected_span
                dynamic_x_limits = False
            else:
                x_window_span = min(float(time_window_seconds), selected_span)
                dynamic_x_limits = x_window_span < selected_span

            def frame_xlim(current_time: float) -> tuple[float, float]:
                if not dynamic_x_limits:
                    return x_start, x_end

                x_right = min(max(current_time, x_start + x_window_span), x_end)
                x_left = max(x_start, x_right - x_window_span)
                return x_left, x_right

        window_series = [samples[start_index:end_index] for samples in sampled_series]
        flat_values = np.concatenate([samples.ravel() for samples in window_series])
        finite_values = flat_values[np.isfinite(flat_values)]
        if finite_values.size == 0:
            raise ValueError(
                "series must contain at least one finite value in the selected window"
            )

        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
        y_span = y_max - y_min
        y_padding = 0.05 * y_span if y_span > 0 else 1.0

        with cls.style_context():
            fig, axis, _ = cls._prepare_axis(ax=ax, figsize=figsize_tuple)

            palette = [
                cls._config.signal_color,
                cls._config.filtered_color,
                cls._config.band_color,
                cls._config.peak_color,
            ]
            cycle = plt.rcParams.get("axes.prop_cycle")
            if cycle is not None:
                for color in cycle.by_key().get("color", []):
                    if color not in palette:
                        palette.append(color)

            plotted_lines: list[Any] = []
            for index, samples in enumerate(window_series):
                color = colors[index] or palette[index % len(palette)]
                line, = axis.plot(
                    [],
                    [],
                    label=labels[index],
                    color=color,
                    linestyle=line_styles[index],
                    linewidth=cls._config.line_width,
                )
                plotted_lines.append(line)

            cursor_line = axis.axvline(
                x=display_time[0],
                color=cls._config.peak_color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.9,
            )

            axis.set_xlim(*frame_xlim(display_time[0]))
            axis.set_ylim(y_min - y_padding, y_max + y_padding)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            if title:
                axis.set_title(title)
            if cls._config.show_grid:
                axis.grid(True, alpha=cls._config.grid_alpha)
            if cls._config.show_legend:
                axis.legend(loc="upper right")

            def init_animation():
                for line in plotted_lines:
                    line.set_data([], [])
                cursor_line.set_xdata([display_time[0], display_time[0]])
                axis.set_xlim(*frame_xlim(display_time[0]))
                return [*plotted_lines, cursor_line]

            def update(frame_index: int):
                current_time = display_time[frame_index]
                for line, samples in zip(plotted_lines, window_series):
                    line.set_data(
                        display_time[: frame_index + 1],
                        samples[: frame_index + 1],
                    )
                cursor_line.set_xdata([current_time, current_time])
                axis.set_xlim(*frame_xlim(current_time))
                return [*plotted_lines, cursor_line]

            if interval_ms is None:
                deltas = np.diff(window_time)
                positive_deltas = deltas[deltas > 0]
                sample_interval = (
                    float(np.median(positive_deltas))
                    if positive_deltas.size
                    else 1.0
                )
                animation_interval = max(sample_interval * 1000.0, 1.0)
            else:
                animation_interval = float(interval_ms)

            animation = FuncAnimation(
                fig,
                update,
                frames=window_time.size,
                init_func=init_animation,
                interval=animation_interval,
                blit=False,
                repeat=False,
            )

            if save_path is not None:
                save_fps = fps if fps is not None else max(1.0, 1000.0 / animation_interval)
                cls.save_animation(
                    animation,
                    save_path,
                    fps=save_fps,
                )

        cls._finalize_figure(fig, show=show, owned_figure=False)
        return fig, axis, animation


    @classmethod
    def plot_frequency_spectrum(
        cls,
        frequencies: ArrayLike1D,
        power: ArrayLike1D,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str = "Frequency [Hz]",
        y_label: str = "Power",
        x_limits: tuple[float, float] | None = None,
        y_limits: tuple[float, float] | None = None,
        tremor_band: tuple[float, float] | None = None,
        annotate_peak: bool = True,
        peak_label: str = "Peak",
        color: str | None = None,
        line_style: str = "-",
        log_scale: bool = False,
        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        if annotate_peak:
            return cls.plot_spectrum_peaks(
                frequencies,
                power,
                ax=ax,
                title=title,
                x_label=x_label,
                y_label=y_label,
                x_limits=x_limits,
                y_limits=y_limits,
                tremor_band=tremor_band,
                peak_label=peak_label,
                color=color,
                line_style=line_style,
                log_scale=log_scale,
                show=show,
            )
        return cls.plot_spectrum(
            frequencies,
            power,
            ax=ax,
            title=title,
            x_label=x_label,
            y_label=y_label,
            x_limits=x_limits,
            y_limits=y_limits,
            tremor_band=tremor_band,
            color=color,
            line_style=line_style,
            log_scale=log_scale,
            show=show,
        )

    @classmethod
    def plot_scalogram(
        cls,
        time: ArrayLike1D,
        frequencies: ArrayLike1D,
        scalogram: ArrayLike2D,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        x_label: str = "Time [s]",
        y_label: str = "Frequency [Hz]",
        colorbar_label: str = "Magnitude",
        frequency_limits: tuple[float, float] | None = None,
        tremor_band: tuple[float, float] | None = None,
        cmap: str | None = None,
        log_scale: bool = True,
        show: bool | None = None,
    ) -> tuple[Figure, Axes]:
        scalogram_array = _as_2d_array(
            scalogram,
            name="scalogram",
        )

        if log_scale:
            scalogram_array = 10.0 * np.log10(
                scalogram_array + np.finfo(float).eps
            )
            colorbar_label = f"{colorbar_label} [dB]"

        return cls.plot_heatmap(
            scalogram_array,
            x_values=time,
            y_values=frequencies,
            ax=ax,
            title=title,
            x_label=x_label,
            y_label=y_label,
            colorbar_label=colorbar_label,
            cmap=cmap,
            y_limits=frequency_limits,
            band=tremor_band,
            show=show,
        )

    @classmethod
    def export_dashboard_plots(cls):
        previous_state = cls._dashboard_active
        cls._dashboard_active = False

        try:
            if not cls._dashboard_entries:
                raise ValueError("Dashboard is empty")

            for index, entry in enumerate(cls._dashboard_entries, start=1):
                fig, _ = entry.render(ax=None, show=False)

                plt.close(fig)

        finally:
            cls._dashboard_active = previous_state



Visualization = Visualizer

__all__ = ["Visualizer", "Visualization", "TimeSeries"]
