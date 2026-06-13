"""Helpers for reading IMU signals from HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import h5py

IMU_GROUP_IDS: dict[str, str | None] = {
    "left_hand": "movella__DOT_D422CD009F54",
    "left_forearm": "movella__DOT_D422CD008603",
    "left_upperarm": "movella__DOT_D422CD00A220",
    "right_hand": "movella__DOT_D422CD008CC7",
    "right_forearm": "movella__DOT_D422CD009F5B",
    "right_upperarm": "movella__DOT_D422CD009F60",
}

ACCEL_COMPONENTS = ("accel_x", "accel_y", "accel_z")
GYRO_COMPONENTS = ("gyro_x", "gyro_y", "gyro_z")
ORIENTATION_EULER_COMPONENTS = ("roll", "pitch", "yaw")
ORIENTATION_QUATERNION_COMPONENTS = ("quat_w", "quat_x", "quat_y", "quat_z")


@dataclass(frozen=True, slots=True)
class SignalInspection:
    """Raw signal quality metrics before any sanitization."""

    sample_count: int
    finite_count: int
    non_finite_count: int
    finite_ratio: float
    has_non_finite: bool
    is_processable: bool


@dataclass(frozen=True, slots=True)
class LoadedSignal:
    """Resolved one-dimensional signal from an HDF5 dataset."""

    values: Any
    source_group: str
    source_path: str
    axis: str
    inspection: SignalInspection
    missing_policy: str
    is_processable: bool


@dataclass(frozen=True, slots=True)
class LoadedSignalBundle:
    """Resolved multi-axis signal bundle from an HDF5 dataset."""

    components: dict[str, Any]
    source_group: str
    source_paths: dict[str, str]
    component_names: tuple[str, ...]
    signal_kind: str
    inspections: dict[str, SignalInspection]
    missing_policy: str
    is_processable: bool


def _normalize_token(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _top_level_sensor_groups(handle) -> list[str]:
    import h5py

    modalities = handle.get("modalities")
    if modalities is None:
        return [f"/{name}" for name, item in handle.items() if isinstance(item, h5py.Group)]

    groups: list[str] = []
    for name, item in modalities.items():
        if isinstance(item, h5py.Group):
            groups.append(f"/modalities/{name}")
    return sorted(groups)


def _resolve_group_path(handle, imu: str, dataset_path: str | None) -> str:
    normalized_imu = _normalize_token(imu)

    if dataset_path is not None:
        path = dataset_path if dataset_path.startswith("/") else f"/{dataset_path}"
        if path not in handle:
            raise KeyError(f"Dataset path '{path}' does not exist in the HDF5 file")
        item = handle[path]
        if item.__class__.__name__ == "Dataset":
            return str(Path(path).parent)
        return path

    sensor_groups = _top_level_sensor_groups(handle)
    if not sensor_groups:
        raise ValueError("No sensor groups found in the HDF5 file")

    exact_matches = [
        group_path
        for group_path in sensor_groups
        if _normalize_token(Path(group_path).name) == normalized_imu
        or normalized_imu in _normalize_token(group_path)
    ]
    if exact_matches:
        return exact_matches[0]

    movella_groups = [group_path for group_path in sensor_groups if "movella" in _normalize_token(group_path)]
    logical_order = {"hand": 0, "forearm": 1, "upperarm": 2}
    if normalized_imu in logical_order and movella_groups:
        ordered_groups = sorted(movella_groups)
        index = logical_order[normalized_imu]
        if index < len(ordered_groups):
            return ordered_groups[index]
        return ordered_groups[0]

    if len(sensor_groups) == 1:
        return sensor_groups[0]

    raise ValueError(
        "Could not resolve the requested IMU. "
        f"Requested='{imu}', available candidates={sensor_groups[:8]}"
    )


def _candidate_dataset_names(axis: str) -> list[str]:
    normalized_axis = _normalize_token(axis)
    if normalized_axis in {"mag", "magnitude"}:
        return ["accel_x", "accel_y", "accel_z", "x", "y", "z", "gyro_x", "gyro_y", "gyro_z"]
    if normalized_axis in {"x", "y", "z"}:
        return [f"accel_{normalized_axis}", normalized_axis, f"gyro_{normalized_axis}"]
    return [normalized_axis]


def _load_vector_dataset(group, dataset_names: list[str], np_module):
    for dataset_name in dataset_names:
        if dataset_name in group:
            return np_module.asarray(group[dataset_name], dtype=float).reshape(-1), dataset_name
    return None, None


def _load_named_dataset(group, dataset_name: str, np_module):
    values, resolved_name = _load_vector_dataset(group, [dataset_name], np_module)
    if values is not None and resolved_name is not None:
        return values, resolved_name

    normalized_target = _normalize_token(dataset_name)
    for candidate_name in group.keys():
        if _normalize_token(candidate_name) == normalized_target:
            return np_module.asarray(group[candidate_name], dtype=float).reshape(-1), candidate_name
    return None, None


def _finite_indices_are_contiguous(finite_mask) -> bool:
    import numpy as np

    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        return False
    return bool(np.all(np.diff(finite_indices) == 1))


def inspect_signal(values: Any) -> SignalInspection:
    """Inspect a numeric signal without mutating it."""

    import numpy as np

    array = np.asarray(values, dtype=float).reshape(-1)
    sample_count = int(array.size)
    if sample_count == 0:
        return SignalInspection(
            sample_count=0,
            finite_count=0,
            non_finite_count=0,
            finite_ratio=0.0,
            has_non_finite=False,
            is_processable=False,
        )

    finite_mask = np.isfinite(array)
    finite_count = int(finite_mask.sum())
    non_finite_count = sample_count - finite_count
    return SignalInspection(
        sample_count=sample_count,
        finite_count=finite_count,
        non_finite_count=non_finite_count,
        finite_ratio=finite_count / sample_count,
        has_non_finite=non_finite_count > 0,
        is_processable=non_finite_count == 0,
    )


def signal_is_processable(values: Any) -> bool:
    """Return ``True`` if a raw signal contains only finite samples."""

    return inspect_signal(values).is_processable


def _sanitize_signal(values: Any, missing_policy: str, *, label: str) -> tuple[Any, bool]:
    """Apply the configured missing-data policy to a one-dimensional signal."""

    import numpy as np

    array = np.asarray(values, dtype=float).reshape(-1)
    inspection = inspect_signal(array)
    policy = _normalize_token(missing_policy)

    if not inspection.has_non_finite:
        return array, True

    finite_mask = np.isfinite(array)

    if policy == "raise":
        raise ValueError(
            f"{label} contains {inspection.non_finite_count} non-finite samples. "
            "Use missing_policy='interpolate', 'trim_edges', or 'drop' explicitly if you want to repair it."
        )

    if policy == "drop":
        cleaned = array[finite_mask]
        contiguous = _finite_indices_are_contiguous(finite_mask)
        if not contiguous:
            warnings.warn(
                f"{label}: dropping interior non-finite samples changes the sampling grid and is not "
                "recommended for FFT or wavelet analysis.",
                RuntimeWarning,
                stacklevel=3,
            )
        return cleaned, contiguous

    if policy == "trim_edges":
        finite_indices = np.flatnonzero(finite_mask)
        if finite_indices.size == 0:
            raise ValueError(f"{label} does not contain any finite samples")
        start = int(finite_indices[0])
        end = int(finite_indices[-1])
        cleaned = array[start : end + 1]
        if not np.isfinite(cleaned).all():
            raise ValueError(
                f"{label} contains interior non-finite samples. "
                "Trimming edges only works when the invalid samples are confined to the start/end."
            )
        return cleaned, True

    if policy == "interpolate":
        finite_indices = np.flatnonzero(finite_mask)
        if finite_indices.size < 2:
            raise ValueError(
                f"{label} has too few finite samples for interpolation. "
                "At least two finite samples are required."
            )
        cleaned = array.copy()
        missing_indices = np.flatnonzero(~finite_mask)
        cleaned[missing_indices] = np.interp(missing_indices, finite_indices, array[finite_mask])
        return cleaned, True

    raise ValueError(
        f"Unknown missing_policy='{missing_policy}'. Expected one of: raise, drop, trim_edges, interpolate."
    )


def _load_signal_from_group(group, group_path: str, axis: str, missing_policy: str, np_module) -> LoadedSignal:
    normalized_axis = _normalize_token(axis)

    if normalized_axis in {"mag", "magnitude"}:
        for family in ("accel", "", "gyro"):
            if family:
                component_names = [f"{family}_x", f"{family}_y", f"{family}_z"]
            else:
                component_names = ["x", "y", "z"]
            if all(name in group for name in component_names):
                components = [np_module.asarray(group[name], dtype=float).reshape(-1) for name in component_names]
                lengths = {component.size for component in components}
                if len(lengths) != 1:
                    raise ValueError(
                        f"Component lengths differ for {group_path}::{','.join(component_names)}"
                    )
                magnitude = np_module.sqrt(sum(component * component for component in components))
                inspection = inspect_signal(magnitude)
                cleaned, is_processable = _sanitize_signal(magnitude, missing_policy, label=f"{group_path}::magnitude")
                return LoadedSignal(
                    values=cleaned,
                    source_group=group_path,
                    source_path=f"{group_path}::magnitude({','.join(component_names)})",
                    axis=axis,
                    inspection=inspection,
                    missing_policy=_normalize_token(missing_policy),
                    is_processable=is_processable,
                )
        raise KeyError(
            f"Could not compute magnitude for IMU group '{group_path}'. "
            "Expected accel_x/y/z, x/y/z, or gyro_x/y/z datasets."
        )

    candidates = _candidate_dataset_names(axis)
    values, dataset_name = _load_vector_dataset(group, candidates, np_module)
    if values is not None and dataset_name is not None:
        inspection = inspect_signal(values)
        cleaned, is_processable = _sanitize_signal(values, missing_policy, label=f"{group_path}/{dataset_name}")
        return LoadedSignal(
            values=cleaned,
            source_group=group_path,
            source_path=f"{group_path}/{dataset_name}",
            axis=axis,
            inspection=inspection,
            missing_policy=_normalize_token(missing_policy),
            is_processable=is_processable,
        )

    # Fall back to a broader search inside the selected group.
    for dataset_name in sorted(group.keys()):
        if _normalize_token(dataset_name).endswith(f"_{normalized_axis}") or _normalize_token(dataset_name) == normalized_axis:
            values = np_module.asarray(group[dataset_name], dtype=float).reshape(-1)
            inspection = inspect_signal(values)
            cleaned, is_processable = _sanitize_signal(values, missing_policy, label=f"{group_path}/{dataset_name}")
            return LoadedSignal(
                values=cleaned,
                source_group=group_path,
                source_path=f"{group_path}/{dataset_name}",
                axis=axis,
                inspection=inspection,
                missing_policy=_normalize_token(missing_policy),
                is_processable=is_processable,
            )

    raise KeyError(
        f"Could not resolve axis '{axis}' inside IMU group '{group_path}'. "
        f"Available datasets: {list(group.keys())[:12]}"
    )


def _load_bundle_from_group(
    group,
    group_path: str,
    component_names: tuple[str, ...],
    signal_kind: str,
    missing_policy: str,
    np_module,
) -> LoadedSignalBundle:
    import numpy as np

    raw_components: dict[str, Any] = {}
    source_paths: dict[str, str] = {}
    inspections: dict[str, SignalInspection] = {}

    for component_name in component_names:
        values, dataset_name = _load_named_dataset(group, component_name, np_module)
        if values is None or dataset_name is None:
            raise KeyError(
                f"Could not resolve component '{component_name}' inside IMU group '{group_path}'. "
                f"Available datasets: {list(group.keys())[:12]}"
            )
        raw_components[component_name] = values
        source_paths[component_name] = f"{group_path}/{dataset_name}"
        inspections[component_name] = inspect_signal(values)

    lengths = {component.size for component in raw_components.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"Component lengths differ inside '{group_path}' for {signal_kind}: "
            f"{ {name: values.size for name, values in raw_components.items()} }"
        )

    policy = _normalize_token(missing_policy)
    components: dict[str, Any] = {}
    is_processable = False

    if policy == "raise":
        for name, inspection in inspections.items():
            if inspection.has_non_finite:
                raise ValueError(
                    f"{group_path}/{name} contains {inspection.non_finite_count} non-finite samples. "
                    "Use missing_policy='interpolate', 'trim_edges', or 'drop' explicitly if you want to repair it."
                )
        components = raw_components
        is_processable = True

    elif policy == "drop":
        masks = [np.isfinite(np.asarray(values, dtype=float).reshape(-1)) for values in raw_components.values()]
        combined_mask = np.logical_and.reduce(masks)
        if not combined_mask.any():
            raise ValueError(f"{group_path} does not contain any finite samples for {signal_kind}")
        components = {
            name: np.asarray(values, dtype=float).reshape(-1)[combined_mask]
            for name, values in raw_components.items()
        }
        is_processable = _finite_indices_are_contiguous(combined_mask)
        if not is_processable:
            warnings.warn(
                f"{group_path}: dropping interior non-finite samples changes the sampling grid and is not "
                "recommended for FFT or wavelet analysis.",
                RuntimeWarning,
                stacklevel=3,
            )

    elif policy == "trim_edges":
        masks = [np.isfinite(np.asarray(values, dtype=float).reshape(-1)) for values in raw_components.values()]
        combined_mask = np.logical_and.reduce(masks)
        finite_indices = np.flatnonzero(combined_mask)
        if finite_indices.size == 0:
            raise ValueError(f"{group_path} does not contain any finite samples for {signal_kind}")
        start = int(finite_indices[0])
        end = int(finite_indices[-1])
        for name, values in raw_components.items():
            trimmed = np.asarray(values, dtype=float).reshape(-1)[start : end + 1]
            if not np.isfinite(trimmed).all():
                raise ValueError(
                    f"{group_path}/{name} contains interior non-finite samples. "
                    "Trimming edges only works when all invalid samples are at the start/end."
                )
            components[name] = trimmed
        is_processable = True

    elif policy == "interpolate":
        components = {}
        for name, values in raw_components.items():
            inspection = inspections[name]
            if inspection.non_finite_count == 0:
                components[name] = values
                continue
            cleaned, _ = _sanitize_signal(values, missing_policy, label=f"{group_path}/{name}")
            components[name] = cleaned
        is_processable = True

    else:
        raise ValueError(
            f"Unknown missing_policy='{missing_policy}'. Expected one of: raise, drop, trim_edges, interpolate."
        )

    return LoadedSignalBundle(
        components=components,
        source_group=group_path,
        source_paths=source_paths,
        component_names=component_names,
        signal_kind=signal_kind,
        inspections=inspections,
        missing_policy=policy,
        is_processable=is_processable,
    )


def resolve_imu_group_name(imu: str) -> str | None:
    """Map a logical limb label to a concrete Movella group name.

    Returns ``None`` for limbs that still need to be filled in.
    """

    normalized = _normalize_token(imu)
    if normalized in IMU_GROUP_IDS:
        return IMU_GROUP_IDS[normalized]
    return None


def load_signal(
    file_path: Path | str,
    imu: str,
    axis: str,
    *,
    dataset_path: str | None = None,
    missing_policy: str = "trim_edges",
) -> LoadedSignal:
    """Load a one-dimensional signal from an HDF5 file."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input HDF5 file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        normalized_imu = _normalize_token(imu)
        mapped_group = resolve_imu_group_name(imu)
        if normalized_imu in IMU_GROUP_IDS:
            if mapped_group is None:
                raise KeyError(
                    f"IMU '{imu}' is not mapped yet. "
                    "Fill src.hdf5_utils.IMU_GROUP_IDS with the Movella group ID."
                )

            candidate_group_path = f"/modalities/{mapped_group}"
            if candidate_group_path in handle:
                group_path = candidate_group_path
            else:
                raise KeyError(
                    f"Mapped IMU group '{candidate_group_path}' does not exist in the HDF5 file."
                )
        else:
            group_path = _resolve_group_path(handle, imu, dataset_path)
        group = handle[group_path]
        import numpy as np

        return _load_signal_from_group(group, group_path, axis, missing_policy, np)


def load_acceleration_axes(
    file_path: Path | str,
    imu: str,
    *,
    dataset_path: str | None = None,
    missing_policy: str = "raise",
) -> LoadedSignalBundle:
    """Load the three acceleration axes for one IMU."""

    import h5py

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input HDF5 file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        normalized_imu = _normalize_token(imu)
        mapped_group = resolve_imu_group_name(imu)
        if normalized_imu in IMU_GROUP_IDS:
            if mapped_group is None:
                raise KeyError(
                    f"IMU '{imu}' is not mapped yet. "
                    "Fill src.hdf5_utils.IMU_GROUP_IDS with the Movella group ID."
                )
            candidate_group_path = f"/modalities/{mapped_group}"
            if candidate_group_path in handle:
                group_path = candidate_group_path
            else:
                raise KeyError(
                    f"Mapped IMU group '{candidate_group_path}' does not exist in the HDF5 file."
                )
        else:
            group_path = _resolve_group_path(handle, imu, dataset_path)
        group = handle[group_path]
        import numpy as np

        return _load_bundle_from_group(
            group,
            group_path,
            ACCEL_COMPONENTS,
            "acceleration",
            missing_policy,
            np,
        )


