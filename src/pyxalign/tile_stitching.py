"""Stitch a 3D tile grid the way ImageJ's Grid/Collection plugin does
(type = "grid: column-by-column", order = "up & left").

In that convention tile index 0 is the BOTTOM-RIGHT tile of the grid; the
index walks UP through the rightmost column, then jumps to the bottom of
the next column to the LEFT, and so on.

All three axes (z, y, x) are handled. Phase correlation runs in 3D so a
tile can drift in z relative to its neighbours, and the stitched volume
is padded in z to cover the union of every tile's z extent.

Four overlap-fusion modes are supported: linear blending (the ImageJ
default, applied along all three axes), simple average, per-pixel max,
and last-tile-wins overwrite.

Memory notes
------------
* Offset estimation only transfers the overlap strip of each tile pair
  to the GPU (not the full tile), uses a real-input FFT to halve the
  complex working memory, and accepts a ``correlation_downsample``
  factor for tiles whose strips still won't fit.
* Stitching builds the working ``(cz, n_tiles, H, W)`` slab one output-z
  chunk at a time, directly from the original tiles, so the full padded
  ``(out_z, n_tiles, H, W)`` array is never materialised. The output
  ``(out_z, out_h, out_w)`` numpy array still has to fit in host RAM.
"""

from typing import List, Optional, Sequence, Tuple

import cupy as cp
import numpy as np
from tqdm import tqdm

from pyxalign.api.enums import BlendMode, DeviceType, OffsetMethod
from pyxalign.api.options.device import DeviceOptions
from pyxalign.api.types import ArrayType


_MODE_CODES = {
    BlendMode.LINEAR: 0,
    BlendMode.AVERAGE: 1,
    BlendMode.MAX: 2,
    BlendMode.OVERWRITE: 3,
}

# cuFFT's PlanNd construction fails with CUFFT_INVALID_SIZE once the total
# transform size exceeds roughly 2**31 elements (32-bit indexing). When the
# user lets `correlation_downsample` default to None, we'll auto-pick the
# smallest factor that keeps strips below this ceiling.
_CUFFT_MAX_ELEMENTS = 2**31 - 1


def _auto_downsample_factor(strip_elements: int, use_gpu: bool) -> int:
    """Smallest d>=1 such that decimating a strip of ``strip_elements``
    voxels by ``d`` on every axis keeps it under cuFFT's plan-size limit.
    Returns 1 on CPU.
    """
    if not use_gpu:
        return 1
    d = 1
    while strip_elements // (d ** 3) > _CUFFT_MAX_ELEMENTS:
        d += 1
    return d


def imagej_tile_grid_coords(n_cols: int, n_rows: int) -> List[Tuple[int, int]]:
    """Map tile index -> (col, row) for the ImageJ "column-by-column,
    up & left" ordering. col=0 is leftmost, row=0 is topmost.
    """
    coords: List[Tuple[int, int]] = []
    for k in range(n_cols * n_rows):
        col_from_right = k // n_rows
        row_from_bottom = k % n_rows
        coords.append((n_cols - 1 - col_from_right, n_rows - 1 - row_from_bottom))
    return coords


def _phase_correlation_shift_3d(a, b) -> Tuple[int, int, int]:
    """Integer (dz, dy, dx) shift that registers ``b`` onto ``a`` via 3D
    phase correlation. Uses ``rfftn`` instead of ``fftn`` so the complex
    spectrum is ~half the size of the equivalent full FFT.
    """
    xp = cp.get_array_module(a)
    a = a.astype(xp.float32, copy=False)
    b = b.astype(xp.float32, copy=False)
    A = xp.fft.rfftn(a)
    B = xp.fft.rfftn(b)
    R = A * xp.conj(B)
    R /= xp.maximum(xp.abs(R), 1e-12)
    corr = xp.fft.irfftn(R, s=a.shape)
    peak = int(corr.argmax())
    Z, H, W = a.shape
    dz, rem = divmod(peak, H * W)
    dy, dx = divmod(rem, W)
    if dz > Z // 2:
        dz -= Z
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return int(dz), int(dy), int(dx)


def _estimate_neighbor_offset(
    tile_a: np.ndarray,
    tile_b: np.ndarray,
    axis: str,
    overlap_fraction: float,
    use_gpu: bool,
    device: int,
    downsample: int = 1,
) -> Tuple[int, int, int]:
    """Return ``(dz, dy, dx)``: the offset of ``tile_b``'s origin relative
    to ``tile_a``'s. ``axis`` is 'x' (b is to the right of a) or 'y'
    (b is below a).

    Only the host-side overlap STRIP is transferred to GPU — the rest of
    each tile is never touched. If ``downsample > 1`` the strip is then
    decimated by that factor on every axis before the FFT, and the
    detected offsets are scaled back up. Increase ``downsample`` if even
    the strip's FFT won't fit on the GPU.
    """
    Z, H, W = tile_a.shape
    if axis == "x":
        ow = max(1, int(round(W * overlap_fraction)))
        a_strip_host = np.ascontiguousarray(tile_a[:, :, W - ow:])
        b_strip_host = np.ascontiguousarray(tile_b[:, :, :ow])
    else:
        oh = max(1, int(round(H * overlap_fraction)))
        a_strip_host = np.ascontiguousarray(tile_a[:, H - oh:, :])
        b_strip_host = np.ascontiguousarray(tile_b[:, :oh, :])

    if downsample > 1:
        a_strip_host = np.ascontiguousarray(
            a_strip_host[::downsample, ::downsample, ::downsample]
        )
        b_strip_host = np.ascontiguousarray(
            b_strip_host[::downsample, ::downsample, ::downsample]
        )

    if use_gpu:
        with cp.cuda.Device(device):
            dz, dy, dx = _phase_correlation_shift_3d(
                cp.asarray(a_strip_host), cp.asarray(b_strip_host)
            )
    else:
        dz, dy, dx = _phase_correlation_shift_3d(a_strip_host, b_strip_host)

    dz *= downsample
    dy *= downsample
    dx *= downsample

    if axis == "x":
        return dz, dy, (W - ow) + dx
    return dz, (H - oh) + dy, dx


def compute_tile_positions(
    tiles: Sequence[np.ndarray],
    grid_shape: Tuple[int, int],
    offset_method: OffsetMethod,
    overlap: Tuple[float, float],
    device_options: DeviceOptions,
    correlation_downsample: Optional[int] = None,
) -> np.ndarray:
    """Return an ``(n_tiles, 3)`` int array of ``(z, y, x)`` origins in
    the stitched volume, indexed in the user-supplied tile order. Origins
    are shifted so the minimum along each axis is 0.

    With ``offset_method="known_overlap"`` the z origin is 0 for every
    tile; only ``"phase_correlation"`` recovers z drift.

    ``correlation_downsample=None`` auto-picks the smallest strip-decimation
    factor that keeps the FFT under cuFFT's plan-size limit; pass an int to
    force a specific factor.
    """
    n_cols, n_rows = grid_shape
    coords = imagej_tile_grid_coords(n_cols, n_rows)
    by_grid = {(c, r): i for i, (c, r) in enumerate(coords)}
    Z, H, W = tiles[0].shape
    ov_y, ov_x = overlap

    if offset_method == OffsetMethod.KNOWN_OVERLAP:
        step_y = int(round(H * (1 - ov_y)))
        step_x = int(round(W * (1 - ov_x)))
        positions = np.zeros((len(tiles), 3), dtype=np.int64)
        for i, (c, r) in enumerate(coords):
            positions[i] = (0, r * step_y, c * step_x)
        return positions

    use_gpu = device_options.device_type == DeviceType.GPU
    device = device_options.gpu.gpu_indices[0]

    if correlation_downsample is None:
        ow = max(1, int(round(W * ov_x)))
        oh = max(1, int(round(H * ov_y)))
        max_strip_elements = max(Z * H * ow, Z * oh * W)
        ds = _auto_downsample_factor(max_strip_elements, use_gpu)
        if ds > 1:
            tqdm.write(
                f"tile_stitching: strip size {max_strip_elements:,} voxels "
                f"exceeds cuFFT plan limit; auto-set correlation_downsample={ds}"
            )
    else:
        ds = correlation_downsample

    pos: dict = {(0, 0): np.zeros(3, dtype=np.int64)}
    for c in range(1, n_cols):
        a, b = tiles[by_grid[(c - 1, 0)]], tiles[by_grid[(c, 0)]]
        dz, dy, dx = _estimate_neighbor_offset(
            a, b, "x", ov_x, use_gpu, device, ds,
        )
        pos[(c, 0)] = pos[(c - 1, 0)] + np.array([dz, dy, dx], dtype=np.int64)
    for c in range(n_cols):
        for r in range(1, n_rows):
            a, b = tiles[by_grid[(c, r - 1)]], tiles[by_grid[(c, r)]]
            dz, dy, dx = _estimate_neighbor_offset(
                a, b, "y", ov_y, use_gpu, device, ds,
            )
            pos[(c, r)] = pos[(c, r - 1)] + np.array([dz, dy, dx], dtype=np.int64)

    raw = np.stack([pos[(c, r)] for (c, r) in coords], axis=0)
    raw -= raw.min(axis=0, keepdims=True)
    return raw.astype(np.int64)


def _triangular(n: int, xp) -> ArrayType:
    return xp.minimum(
        xp.arange(1, n + 1, dtype=xp.float32),
        xp.arange(n, 0, -1, dtype=xp.float32),
    )


def _linear_blend_weight_yx(H: int, W: int, xp) -> ArrayType:
    """Separable triangular distance-from-edge weight in y and x. The z
    factor is supplied per-tile via the precomputed z-mask column.
    """
    return xp.outer(_triangular(H, xp), _triangular(W, xp))


def _stitched_shape(
    positions: np.ndarray, tile_z: int, tile_h: int, tile_w: int
) -> Tuple[int, int, int]:
    return (
        int(positions[:, 0].max()) + tile_z,
        int(positions[:, 1].max()) + tile_h,
        int(positions[:, 2].max()) + tile_w,
    )


def _build_z_mask(
    positions: np.ndarray, tile_z: int, out_z: int, blend_mode: BlendMode,
) -> np.ndarray:
    """For every output-z row and every tile, give the tile's z-axis
    contribution weight. Triangular along z for ``"linear"``; otherwise 1
    inside the tile's z range and 0 outside. Shape: ``(out_z, n_tiles)``.
    """
    n_tiles = positions.shape[0]
    z_mask = np.zeros((out_z, n_tiles), dtype=np.float32)
    column = (
        np.minimum(
            np.arange(1, tile_z + 1, dtype=np.float32),
            np.arange(tile_z, 0, -1, dtype=np.float32),
        )
        if blend_mode == BlendMode.LINEAR
        else np.ones(tile_z, dtype=np.float32)
    )
    for ti in range(n_tiles):
        z0 = int(positions[ti, 0])
        z_mask[z0:z0 + tile_z, ti] = column
    return z_mask


def _build_chunk_slab(
    tiles: Sequence[np.ndarray],
    z_origins: np.ndarray,
    z_start: int,
    z_end: int,
    Z: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Build ``(cz, n_tiles, H, W)`` by slicing each tile's contribution
    to output-z range ``[z_start, z_end)``. Pixels outside any tile's
    actual z extent stay zero and are gated out by ``z_mask``.
    """
    cz = z_end - z_start
    n_tiles = len(tiles)
    chunk = np.zeros((cz, n_tiles, H, W), dtype=tiles[0].dtype)
    for ti in range(n_tiles):
        z0 = int(z_origins[ti])
        tile_zs = max(0, z_start - z0)
        tile_ze = min(Z, z_end - z0)
        if tile_zs >= tile_ze:
            continue
        out_zs = max(0, z0 - z_start)
        chunk[out_zs:out_zs + (tile_ze - tile_zs), ti] = tiles[ti][tile_zs:tile_ze]
    return chunk


def _stitch_kernel(
    stack_chunk, z_mask_chunk, yx_origins,
    mode_code: int, H: int, W: int, out_h: int, out_w: int,
):
    """Blend one ``(cz, n_tiles, H, W)`` slab into ``(cz, out_h, out_w)``."""
    xp = cp.get_array_module(stack_chunk)
    cz, nt = stack_chunk.shape[0], stack_chunk.shape[1]
    yx_host = cp.asnumpy(yx_origins) if xp is cp else yx_origins

    if mode_code == 3:
        out = xp.zeros((cz, out_h, out_w), dtype=stack_chunk.dtype)
        for ti in range(nt):
            y0, x0 = int(yx_host[ti, 0]), int(yx_host[ti, 1])
            z_valid = (z_mask_chunk[:, ti] > 0)[:, None, None]
            region = out[:, y0:y0 + H, x0:x0 + W]
            region[:] = xp.where(z_valid, stack_chunk[:, ti], region)
        return out

    if mode_code == 2:
        acc = xp.full((cz, out_h, out_w), -xp.inf, dtype=xp.float32)
        for ti in range(nt):
            y0, x0 = int(yx_host[ti, 0]), int(yx_host[ti, 1])
            z_valid = (z_mask_chunk[:, ti] > 0)[:, None, None]
            tile_for_max = xp.where(
                z_valid, stack_chunk[:, ti].astype(xp.float32), -xp.inf,
            )
            region = acc[:, y0:y0 + H, x0:x0 + W]
            xp.maximum(region, tile_for_max, out=region)
        acc = xp.where(xp.isfinite(acc), acc, 0)
        return acc.astype(stack_chunk.dtype)

    # Weighted blends (linear, average)
    if mode_code == 0:
        yx_weight = _linear_blend_weight_yx(H, W, xp)
    else:
        yx_weight = xp.ones((H, W), dtype=xp.float32)
    accum = xp.zeros((cz, out_h, out_w), dtype=xp.float32)
    weight_sum = xp.zeros((cz, out_h, out_w), dtype=xp.float32)
    for ti in range(nt):
        y0, x0 = int(yx_host[ti, 0]), int(yx_host[ti, 1])
        w3d = z_mask_chunk[:, ti, None, None] * yx_weight[None, :, :]
        accum[:, y0:y0 + H, x0:x0 + W] += (
            stack_chunk[:, ti].astype(xp.float32) * w3d
        )
        weight_sum[:, y0:y0 + H, x0:x0 + W] += w3d
    return (accum / xp.maximum(weight_sum, 1e-12)).astype(stack_chunk.dtype)


def stitch_tiles(
    tiles: Sequence[np.ndarray],
    grid_shape: Tuple[int, int],
    device_options: DeviceOptions,
    offset_method: OffsetMethod = OffsetMethod.PHASE_CORRELATION,
    overlap: Tuple[float, float] = (0.2, 0.2),
    blend_mode: BlendMode = BlendMode.LINEAR,
    positions: Optional[np.ndarray] = None,
    pinned_results: Optional[np.ndarray] = None,
    correlation_downsample: Optional[int] = None,
) -> np.ndarray:
    """Stitch a grid of 3D tiles into one volume.

    The pattern the tiles are placed in is as follows: if the `grid_shape`
    is (2x2), the first entry in the list `tiles` will be on the lower 
    right, the second entry will be place above that tile, and then the
    third entry will start on the bottom of the next row to the left, and
    the fourth entry will be above that tile. This corresponds to ImageJ's 
    Grid/Collection stitching tool, where the type is "grid: column-by-column"
    and the order is "up and left"

    Parameters
    ----------
    tiles
        Sequence of 3D arrays ``(z, y, x)``. All tiles must share the
        same shape. Tile order follows ImageJ "grid: column-by-column,
        up & left": index 0 = bottom-right, indices step up then leftward.
    grid_shape
        ``(n_cols, n_rows)`` of the tile grid.
    device_options
        Selects CPU vs GPU. On GPU, ``device_options.gpu.chunk_length``
        sets the output-z chunk size that controls per-chunk VRAM use;
        ``device_options.gpu.gpu_indices[0]`` selects which GPU runs the
        work. (Multi-GPU pipelining is not currently used here.)
    offset_method
        ``"known_overlap"`` places tiles on a regular grid using
        ``overlap`` with z origin 0 for every tile. ``"phase_correlation"``
        runs a 3D FFT correlation on each adjacent pair so z drift is
        recovered too; ``overlap`` is then only used as a region hint
        for the strip correlated in y/x.
    overlap
        ``(overlap_y, overlap_x)`` as fractions in [0, 1).
    blend_mode
        Fusion in overlap regions. ``"linear"`` matches the ImageJ
        default and is applied along z as well when tiles z-overlap.
    positions
        Optional precomputed ``(n_tiles, 3)`` origins ``(z, y, x)``;
        skips offset estimation. Each axis must already be shifted so
        its min is 0.
    pinned_results
        Optional pre-allocated numpy buffer of shape
        ``(out_z, out_h, out_w)`` to receive the stitched volume.
    correlation_downsample
        Decimation factor applied to each overlap strip before FFT
        correlation in ``"phase_correlation"`` mode. ``None`` (default)
        auto-picks the smallest factor that keeps the FFT under cuFFT's
        plan-size limit (~2**31 voxels); an explicit int forces that
        factor exactly. The detected offsets are scaled back to original-
        tile pixels, so the only cost is offset precision
        (±``correlation_downsample`` pixels). Ignored in
        ``"known_overlap"`` mode.
    """
    n_tiles = len(tiles)
    expected = grid_shape[0] * grid_shape[1]
    assert n_tiles == expected, (
        f"Expected {expected} tiles for grid {grid_shape}, got {n_tiles}."
    )
    base_shape = tiles[0].shape
    assert all(t.shape == base_shape for t in tiles), \
        "All tiles must share the same shape."
    assert len(base_shape) == 3, "Tiles must be 3D (z, y, x)."

    Z, H, W = base_shape
    if positions is None:
        positions = compute_tile_positions(
            tiles, grid_shape, offset_method, overlap, device_options,
            correlation_downsample=correlation_downsample,
        )
    assert positions.shape == (n_tiles, 3), \
        f"positions must have shape ({n_tiles}, 3), got {positions.shape}"

    out_z, out_h, out_w = _stitched_shape(positions, Z, H, W)
    mode_code = _MODE_CODES[blend_mode]
    z_mask = _build_z_mask(positions, Z, out_z, blend_mode)
    yx_origins = positions[:, 1:].astype(np.int64)
    z_origins = positions[:, 0].astype(np.int64)

    if pinned_results is None:
        output = np.empty((out_z, out_h, out_w), dtype=tiles[0].dtype)
    else:
        assert pinned_results.shape == (out_z, out_h, out_w), (
            f"pinned_results must have shape ({out_z}, {out_h}, {out_w}), "
            f"got {pinned_results.shape}"
        )
        output = pinned_results

    use_gpu = device_options.device_type == DeviceType.GPU
    if use_gpu and device_options.gpu.chunking_enabled:
        chunk_length = device_options.gpu.chunk_length
    else:
        chunk_length = out_z
    n_chunks = (out_z + chunk_length - 1) // chunk_length

    if use_gpu:
        gpu = device_options.gpu.gpu_indices[0]
        with cp.cuda.Device(gpu):
            yx_origins_gpu = cp.asarray(yx_origins)
            for i in tqdm(range(n_chunks), desc="stitch_tiles"):
                z_start = i * chunk_length
                z_end = min(z_start + chunk_length, out_z)
                chunk_host = _build_chunk_slab(
                    tiles, z_origins, z_start, z_end, Z, H, W,
                )
                chunk_gpu = cp.asarray(chunk_host)
                del chunk_host
                mask_gpu = cp.asarray(z_mask[z_start:z_end])
                result_gpu = _stitch_kernel(
                    chunk_gpu, mask_gpu, yx_origins_gpu, mode_code,
                    H, W, out_h, out_w,
                )
                del chunk_gpu, mask_gpu
                result_gpu.get(out=output[z_start:z_end])
                del result_gpu
    else:
        for i in tqdm(range(n_chunks), desc="stitch_tiles"):
            z_start = i * chunk_length
            z_end = min(z_start + chunk_length, out_z)
            chunk_host = _build_chunk_slab(
                tiles, z_origins, z_start, z_end, Z, H, W,
            )
            result = _stitch_kernel(
                chunk_host, z_mask[z_start:z_end], yx_origins, mode_code,
                H, W, out_h, out_w,
            )
            output[z_start:z_end] = result

    return output
