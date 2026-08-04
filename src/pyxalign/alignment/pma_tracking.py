"""
Data structures for tracking projection-matching alignment (PMA) runs.

`PMASnapshot` captures every factor that influenced a single call to
`LaminographyAlignmentTask.get_projection_matching_shift`. `PMASequence`
holds the ordered list of snapshots produced over the course of an
alignment session and can persist itself to (or be reloaded from) an
HDF5 file.
"""

import copy
import dataclasses
from dataclasses import field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any, List, Optional, TYPE_CHECKING

import h5py
import numpy as np

from pyxalign.api.options.alignment import (
    PMASequenceOptions,
    ProjectionMatchingOptions,
)
from pyxalign.api.options.options import ExperimentOptions
from pyxalign.api.options.projections import (
    ProbePositionMaskOptions,
    VolumeWidthOptions,
)
from pyxalign.api.options.reconstruct import ReconstructOptions
from pyxalign.api.options.roi import ROIOptions
from pyxalign.api.options.transform import CropOptions
from pyxalign.io.save import save_generic_data_structure_to_h5
from pyxalign.io.utils import (
    dict_to_dataclass,
    h5_to_dict,
    handle_null_type,
    is_null_type,
)
from pyxalign.transformations.classes import Cropper

if TYPE_CHECKING:
    from pyxalign.data_structures.projections import PhaseProjections


@dataclasses.dataclass(eq=False)
class PMASnapshot:
    """All inputs that shaped one `get_projection_matching_shift` call.

    `eq=False` keeps default identity-based equality so np.ndarray fields
    don't break `__eq__` and so we can safely look snapshots up by `is`
    when resolving parent links.
    """

    pma_options: ProjectionMatchingOptions

    initial_shift: Optional[np.ndarray]

    final_shift: Optional[np.ndarray]
    """Shift returned by the PMA call. None until the call finishes."""

    removed_scan_numbers: np.ndarray
    """Scan numbers that were dropped from `phase_projections` before this call."""

    removed_angles: np.ndarray
    """Angles (parallel to `removed_scan_numbers`) of the dropped scans."""

    volume: Optional[np.ndarray]
    """Recorded post-PMA volume (or a layer/cropped subset). None unless `pma_options.pma_sequence.record_volume` was enabled."""

    past_shift_sum: np.ndarray
    """Sum of the shifts already applied via the projections' ShiftManager."""

    angles: np.ndarray
    scan_numbers: np.ndarray
    center_of_rotation: np.ndarray

    mask_source: Optional[str]
    """String value of the `MaskSource` enum at the time of the call."""

    reconstruct: ReconstructOptions
    volume_width: VolumeWidthOptions
    experiment: ExperimentOptions
    mask_from_positions: ProbePositionMaskOptions
    masks_from_roi: ROIOptions

    timestamp: str = ""

    parent_index: Optional[int] = None
    """Index in the owning `PMASequence` of the snapshot whose `final_shift` was used as this run's `initial_shift`. None if no parent was recorded (manual or backward-compat path)."""

    def compute_shift_relative_to(
        self, phase_projections: "PhaseProjections"
    ) -> np.ndarray:
        """Re-express this snapshot's `final_shift` relative to a current `PhaseProjections`.

        The "absolute" alignment recorded in a snapshot is
        `past_shift_sum + final_shift` — that sum is invariant under any
        shifts that have been applied to the projections since the
        snapshot was recorded. To use the snapshot as an initial guess or
        to stage it as a new shift on the *current* projections we need
        to subtract whatever has been applied since then and align the
        rows to the current scan ordering.

        Raises ValueError if `final_shift` is None (the original PMA run
        never completed) or if `phase_projections` contains scans that
        weren't present when the snapshot was taken.
        """
        if self.final_shift is None:
            raise ValueError(
                "Cannot derive a shift from this PMASnapshot: its "
                "final_shift is None (the prior PMA run did not finish)."
            )
        snap_absolute_shift = (
            np.asarray(self.past_shift_sum) + np.asarray(self.final_shift)
        )
        snap_absolute_shift_current = _align_shift_to_current_scans(
            snap_absolute_shift,
            snapshot_scan_numbers=self.scan_numbers,
            current_scan_numbers=phase_projections.scan_numbers,
        )
        sm = phase_projections.shift_manager
        if len(sm.past_shifts) > 0:
            current_past_sum = np.sum(sm.past_shifts, axis=0)
        else:
            current_past_sum = np.zeros_like(sm.staged_shift)
        return snap_absolute_shift_current - current_past_sum

    @classmethod
    def from_phase_projections(
        cls,
        phase_projections: "PhaseProjections",
        pma_options: ProjectionMatchingOptions,
        initial_shift: Optional[np.ndarray],
        parent_index: Optional[int] = None,
    ) -> "PMASnapshot":
        sm = phase_projections.shift_manager
        if len(sm.past_shifts) > 0:
            past_shift_sum = np.sum(sm.past_shifts, axis=0)
        else:
            past_shift_sum = np.zeros_like(sm.staged_shift)

        dropped = list(getattr(phase_projections, "dropped_scan_numbers", []) or [])
        dropped_angles_map = getattr(phase_projections, "dropped_angles", {}) or {}
        removed_scan_numbers = np.array(dropped, dtype=int)
        removed_angles = np.array(
            [dropped_angles_map.get(s, np.nan) for s in dropped], dtype=float
        )

        po = phase_projections.options
        ms = phase_projections.mask_source
        # If the caller didn't provide an initial shift, record an explicit
        # zero array of the right shape so it shows up in the viewer's
        # plots (a flat line at 0) and snapshot info.
        if initial_shift is None:
            initial_shift_to_store = np.zeros_like(sm.staged_shift)
        else:
            initial_shift_to_store = np.asarray(initial_shift).copy()
        return cls(
            pma_options=copy.deepcopy(pma_options),
            initial_shift=initial_shift_to_store,
            final_shift=None,
            removed_scan_numbers=removed_scan_numbers,
            removed_angles=removed_angles,
            volume=None,
            past_shift_sum=np.asarray(past_shift_sum).copy(),
            angles=np.asarray(phase_projections.angles).copy(),
            scan_numbers=np.asarray(phase_projections.scan_numbers).copy(),
            center_of_rotation=np.asarray(phase_projections.center_of_rotation).copy(),
            mask_source=str(ms) if ms is not None else None,
            reconstruct=copy.deepcopy(po.reconstruct),
            volume_width=copy.deepcopy(po.volume_width),
            experiment=copy.deepcopy(po.experiment),
            mask_from_positions=copy.deepcopy(po.mask_from_positions),
            masks_from_roi=copy.deepcopy(po.masks_from_roi),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            parent_index=parent_index,
        )


def _align_shift_to_current_scans(
    shift: np.ndarray,
    snapshot_scan_numbers: np.ndarray,
    current_scan_numbers: np.ndarray,
) -> np.ndarray:
    """Reorder/subset `shift` so its rows line up with `current_scan_numbers`.

    Raises ValueError if a current scan wasn't present when the snapshot
    was recorded — the snapshot can't supply a shift for it.
    """
    shift = np.asarray(shift)
    snap_scans = np.asarray(snapshot_scan_numbers).astype(int)
    cur_scans = np.asarray(current_scan_numbers).astype(int)

    if shift.shape[0] != snap_scans.shape[0]:
        raise ValueError(
            f"Snapshot shift has {shift.shape[0]} rows but its "
            f"scan_numbers has {snap_scans.shape[0]} entries; cannot align."
        )

    snap_scan_to_row = {int(s): r for r, s in enumerate(snap_scans)}
    missing = [int(s) for s in cur_scans if int(s) not in snap_scan_to_row]
    if missing:
        raise ValueError(
            "Cannot use the passed PMASnapshot: the projections currently "
            f"include scan(s) {missing} that were not present when that "
            "snapshot was recorded."
        )
    rows = np.array([snap_scan_to_row[int(s)] for s in cur_scans], dtype=int)
    return shift[rows].copy()


def crop_volume_for_recording(
    volume_data: np.ndarray, options: PMASequenceOptions
) -> np.ndarray:
    """Apply the PMASequenceOptions layer slice + in-plane crop to a volume."""
    arr = np.asarray(volume_data)
    n_layers = arr.shape[0]
    start = int(np.clip(options.volume_start_layer_fractional, 0.0, 1.0) * n_layers)
    end = int(np.clip(options.volume_end_layer_fractional, 0.0, 1.0) * n_layers)
    end = max(end, start + 1)
    sliced = arr[start:end]
    crop_options = copy.deepcopy(options.volume_crop)
    if crop_options.enabled:
        # Cropper operates on (N, H, W); volume is (Z, Y, X) which matches.
        try:
            sliced = Cropper(crop_options).run(sliced)
        except ValueError:
            print(
                "WARNING: volume_crop values are invalid for the current volume shape "
                f"{sliced.shape}. Proceeding without cropping."
            )
    return np.asarray(sliced).copy()


def compute_chains(snapshots: list["PMASnapshot"]) -> dict[int, list[int]]:
    """
    Group snapshots into linear chains by following `parent_index` links.

    Returns a mapping from terminal-snapshot-index to the ordered list of
    indices (root → terminal) in the chain ending at that terminal. A
    "terminal" is an index that no later snapshot references as parent.
    Snapshots whose `parent_index` points outside the sequence are
    treated as roots.
    """
    n = len(snapshots)
    has_child = [False] * n
    for i, snap in enumerate(snapshots):
        p = getattr(snap, "parent_index", None)
        if p is not None and 0 <= p < n and p != i:
            has_child[p] = True

    chains: dict[int, list[int]] = {}
    for terminal in range(n):
        if has_child[terminal]:
            continue
        chain: list[int] = []
        seen: set[int] = set()
        cur: Optional[int] = terminal
        while cur is not None and cur not in seen:
            seen.add(cur)
            chain.append(cur)
            p = getattr(snapshots[cur], "parent_index", None)
            cur = p if (p is not None and 0 <= p < n and p != cur) else None
        chains[terminal] = list(reversed(chain))
    return chains


class PMASequencePlotType(StrEnum):
    """Display modes for the PMA sequence viewer."""

    INITIAL_FINAL_SHIFTS = auto()
    ANGLES = auto()
    VOLUME = auto()


class PMASequenceSortAxis(StrEnum):
    """X-axis ordering for the shift / angle plots."""

    BY_ANGLE = auto()
    BY_SCAN_NUMBER = auto()


@dataclasses.dataclass
class PMASequence:
    """Ordered history of `PMASnapshot` records for a single task."""

    snapshots: List[PMASnapshot] = field(default_factory=list)

    def append(self, snapshot: PMASnapshot) -> None:
        self.snapshots.append(snapshot)

    def clear(self) -> None:
        self.snapshots.clear()

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, idx) -> PMASnapshot:
        return self.snapshots[idx]

    def __iter__(self):
        return iter(self.snapshots)

    def save(self, file_path: str, include_volumes: bool = True) -> None:
        """Save the sequence to a standalone HDF5 file."""
        with h5py.File(file_path, "w") as F:
            self._save_to_group(F, include_volumes=include_volumes)
        print(f"PMASequence with {len(self.snapshots)} snapshot(s) saved to {file_path}")

    def _save_to_group(self, group: h5py.Group, include_volumes: bool = True) -> None:
        """Write the sequence into an existing h5 group (e.g. inside a task file)."""
        group.attrs["n_snapshots"] = len(self.snapshots)
        for i, snap in enumerate(self.snapshots):
            snap_to_save = snap
            if not include_volumes and snap.volume is not None:
                # Shallow copy with the volume stripped — keeps the live
                # snapshot intact while keeping the file small.
                snap_to_save = dataclasses.replace(snap, volume=None)
            save_generic_data_structure_to_h5(
                snap_to_save, group.create_group(_snapshot_key(i))
            )

    @classmethod
    def load(cls, file_path: str, include_volumes: bool = True) -> "PMASequence":
        """Load a sequence previously written with `save`."""
        with h5py.File(file_path, "r") as F:
            seq = cls._load_from_group(F, include_volumes=include_volumes)
        print(f"Loaded PMASequence with {len(seq)} snapshot(s) from {file_path}")
        return seq

    @classmethod
    def _load_from_group(
        cls, group: h5py.Group, include_volumes: bool = True
    ) -> "PMASequence":
        """Read a sequence out of an existing h5 group."""
        n = int(group.attrs["n_snapshots"])
        snapshots = [
            _load_snapshot_group(group[_snapshot_key(i)], include_volume=include_volumes)
            for i in range(n)
        ]
        return cls(snapshots=snapshots)


def _snapshot_key(i: int) -> str:
    return f"snapshot_{i:04d}"


def _load_snapshot_group(
    group: h5py.Group, include_volume: bool = True
) -> PMASnapshot:
    if include_volume or "volume" not in group:
        data = h5_to_dict(group)
    else:
        # Read every dataset / nested group except `volume` so we don't
        # pull a possibly-large array off disk just to discard it.
        data: dict[str, Any] = {}
        for key, item in group.items():
            if key == "volume":
                data[key] = None
                continue
            if isinstance(item, h5py.Group):
                data[key] = h5_to_dict(item)
            else:
                value = item[()]
                if isinstance(value, bytes):
                    value = handle_null_type(value) if is_null_type(value) else value.decode()
                data[key] = value
    return dict_to_dataclass(PMASnapshot, data)
