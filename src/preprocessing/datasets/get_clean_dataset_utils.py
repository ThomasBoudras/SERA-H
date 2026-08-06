"""Utilities to build VRT mosaics from Sentinel-1/Sentinel-2 tiles and to select valid ones.

This module provides helpers to (1) regroup downloaded S1/S2 GeoTIFFs into per-date (or global)
VRT mosaics, and (2) find, for a given geometry and reference date, the combination of S2/S1
ascending/descending VRTs that actually contain valid (non-empty) data for that geometry.
"""

import logging
import shutil
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from geefetch.utils.rasterio import create_vrt
from tqdm import tqdm

from src.global_utils import get_window


def _create_vrts_from_dict(vrts_path: Path, dict_vrt: Dict[str, List[Path]]) -> None:
    """Create one VRT per date from lists of source tif paths, replacing any existing directory.

    Args:
        vrts_path: Directory where the VRT files will be written. If it already exists, it is
            removed and recreated.
        dict_vrt: Mapping from date string to the list of tif paths to mosaic for that date.
    """
    if vrts_path.exists():
        shutil.rmtree(vrts_path)
    vrts_path.mkdir(parents=True)

    for date, list_tif in tqdm(
        dict_vrt.items(), desc="Creating VRTs from dict", total=len(dict_vrt)
    ):
        if list_tif:
            vrt_path = vrts_path / (date + ".vrt")
            create_vrt(vrt_path, list_tif)


def create_vrts_timeseries(data_dir: Path) -> None:
    """Regroup S1 (by date and orbit) and S2 (by date) tifs into per-date VRT mosaics.

    Args:
        data_dir: Root data directory containing the `s1` and `s2` subdirectories with the
            downloaded tifs. The resulting VRTs are written to `s1/s1_asc_vrts`,
            `s1/s1_dsc_vrts`, and `s2/s2_vrts`.
    """
    s1_path = data_dir / "s1"
    s2_path = data_dir / "s2"

    s1_vrts_asc_path = s1_path / "s1_asc_vrts"
    s1_vrts_dsc_path = s1_path / "s1_dsc_vrts"
    s2_vrts_path = s2_path / "s2_vrts"

    logging.info("Regrouping S1 files by date and orbit")
    dict_vrt_s1_asc = defaultdict(list)
    dict_vrt_s1_dsc = defaultdict(list)
    list_path_s1 = list(s1_path.rglob("*tif"))
    for path in tqdm(
        list_path_s1, desc="Regrouping S1 files by date and orbit", total=len(list_path_s1)
    ):
        date_s1 = path.stem.split("_")[4][:8]
        with rasterio.open(path) as src:
            s1_orbit = src.tags()["orbitProperties_pass"]
        if s1_orbit == "ASCENDING":
            dict_vrt_s1_asc[date_s1].append(path)
        else:
            dict_vrt_s1_dsc[date_s1].append(path)

    logging.info("Regrouping S2 files by date")
    dict_vrt_s2 = defaultdict(list)
    list_path_s2 = list(s2_path.rglob("*tif"))
    for path in tqdm(list_path_s2, desc="Regrouping S2 files by date", total=len(list_path_s2)):
        date_s2 = path.stem[:8]
        dict_vrt_s2[date_s2].append(path)

    logging.info("Creating VRTs for S1 ASC")
    _create_vrts_from_dict(s1_vrts_asc_path, dict_vrt_s1_asc)
    logging.info("Creating VRTs for S1 DSC")
    _create_vrts_from_dict(s1_vrts_dsc_path, dict_vrt_s1_dsc)
    logging.info("Creating VRTs for S2")
    _create_vrts_from_dict(s2_vrts_path, dict_vrt_s2)


def create_vrts_composites(data_dir: Path) -> None:
    """Build a single global VRT mosaic per source (S1 ascending, S1 descending, S2).

    Args:
        data_dir: Root data directory containing the `s1` and `s2` subdirectories with the
            downloaded tifs. The resulting VRTs are written to `s1/s1_asc.vrt`, `s1/s1_dsc.vrt`,
            and `s2/s2.vrt`.

    Raises:
        ValueError: If an S1 file stem contains neither "asc" nor "desc".
    """
    s1_path = data_dir / "s1"
    s2_path = data_dir / "s2"

    s1_asc_vrt_path = s1_path / "s1_asc.vrt"
    s1_dsc_vrt_path = s1_path / "s1_dsc.vrt"
    s2_vrt_path = s2_path / "s2.vrt"

    list_tif_s1_asc = []
    list_tif_s1_dsc = []
    for path in s1_path.rglob("*tif"):
        if "asc" in path.stem:
            list_tif_s1_asc.append(path)
        elif "desc" in path.stem:
            list_tif_s1_dsc.append(path)
        else:
            raise ValueError(f"File {path.stem} is not a valid S1 file")

    list_tif_s2 = list(s2_path.rglob("*tif"))

    create_vrt(s1_asc_vrt_path, list_tif_s1_asc)
    create_vrt(s1_dsc_vrt_path, list_tif_s1_dsc)
    create_vrt(s2_vrt_path, list_tif_s2)


def _get_date_from_vrt_name(vrt_name: Path) -> datetime:
    """Parse the date encoded in a VRT file's stem (expected format: YYYYMMDD)."""
    return datetime.strptime(vrt_name.stem, "%Y%m%d")


def filter_files_by_date_gap(
    reference_date: str, file_list: List[Path], max_days_gap: int
) -> List[Path]:
    """Filters files by maximum allowed date gap from the reference date.

    Args:
        reference_date: Reference date string (expected format: YYYYMMDD).
        file_list: List of Path objects with date info encoded in their stem.
        max_days_gap: Maximum allowed number of days between a file's date and the reference
            date for it to be kept.

    Returns:
        The subset of `file_list` whose date is within `max_days_gap` days of `reference_date`.
    """
    ref_date = datetime.strptime(reference_date, "%Y%m%d")
    filtered_files = []
    for file in file_list:
        file_date = _get_date_from_vrt_name(file)
        days_diff = abs((file_date - ref_date).days)
        if days_diff <= max_days_gap:
            filtered_files.append(file)
    return filtered_files


def _sort_by_proximity(target_file: Path, file_list: List[Path]) -> List[Path]:
    """Sorts a list of files by proximity in days to the target_file.

    Args:
        target_file: Path-like, the reference file whose date is used.
        file_list: list of Path-like objects to compare.

    Returns:
        List of files sorted by date proximity to target_file.
    """
    target_date = _get_date_from_vrt_name(target_file)
    files_with_gap = [
        (file, abs((_get_date_from_vrt_name(file) - target_date).days)) for file in file_list
    ]
    sorted_files = sorted(files_with_gap, key=lambda x: x[1])
    return [file for file, gap in sorted_files]


def _verify_window(path: Path, bounds: Tuple[float, float, float, float]) -> bool:
    """Verify that the window of `path` covered by `bounds` contains finite (non-empty) data.

    Args:
        path: Path to the raster (or VRT) to read from.
        bounds: Bounding box `(left, bottom, right, top)` to read the window for.

    Returns:
        True if the window was read successfully and contains at least one finite value.
    """
    img, _ = get_window(
        image_path=path,
        bounds=bounds,
        resolution=None,
        resampling_method="bilinear",
    )
    return img is not None and bool(img.size) and np.isfinite(img).any()  # type: ignore


def get_valid_vrts_timeseries(
    data_dir: Path, geometry: Any, date: str, half_window_size: int
) -> Optional[List[List[str]]]:
    """Find valid combinations of (S2, S1 ascending, S1 descending) VRTs for a geometry.

    For each S2 VRT within `half_window_size` days of `date` that has valid data over
    `geometry`, the nearest (by date) S1 ascending and S1 descending VRTs with valid data are
    selected, without reusing an S1 VRT across multiple matches.

    Args:
        data_dir: Root data directory containing the `s1/s1_asc_vrts`, `s1/s1_dsc_vrts`, and
            `s2/s2_vrts` VRT directories.
        geometry: Shapely geometry to check for data validity.
        date: Reference date string (expected format: YYYYMMDD) used to filter candidate VRTs.
        half_window_size: Maximum allowed number of days between a VRT's date and `date`.

    Returns:
        A list of `[s2_vrt_stem, s1_asc_vrt_stem, s1_dsc_vrt_stem]` triplets, or None if no
        valid combination was found or an error occurred while processing the geometry.
    """
    try:
        bounds = geometry.bounds
        valid_vrts = []

        limit_date = datetime.strptime("20170328", "%Y%m%d") + timedelta(days=half_window_size)
        limit_date_str = limit_date.strftime("%Y%m%d")
        if date < limit_date_str:
            date = limit_date_str

        s1_asc_vrts_path = data_dir / "s1" / "s1_asc_vrts"
        s1_dsc_vrts_path = data_dir / "s1" / "s1_dsc_vrts"
        s2_vrts_path = data_dir / "s2" / "s2_vrts"

        s1_asc_vrts = [file for file in s1_asc_vrts_path.iterdir() if file.suffix == ".vrt"]
        s1_asc_vrts = filter_files_by_date_gap(date, s1_asc_vrts, half_window_size)

        s1_dsc_vrts = [file for file in s1_dsc_vrts_path.iterdir() if file.suffix == ".vrt"]
        s1_dsc_vrts = filter_files_by_date_gap(date, s1_dsc_vrts, half_window_size)

        s2_vrts = [file for file in s2_vrts_path.iterdir() if file.suffix == ".vrt"]
        s2_vrts = filter_files_by_date_gap(date, s2_vrts, half_window_size)

        for s2_vrt in s2_vrts:
            if _verify_window(s2_vrt, bounds):
                sorted_s1_asc_vrts = _sort_by_proximity(
                    s2_vrt, s1_asc_vrts
                )  # We are looking for the nearest tensor s1 in terms of date
                sorted_s1_dsc_vrts = _sort_by_proximity(s2_vrt, s1_dsc_vrts)

                for s1_asc_vrt in sorted_s1_asc_vrts:
                    if _verify_window(s1_asc_vrt, bounds):
                        for s1_dsc_vrt in sorted_s1_dsc_vrts:
                            if _verify_window(s1_dsc_vrt, bounds):
                                valid_vrts.append([s2_vrt.stem, s1_asc_vrt.stem, s1_dsc_vrt.stem])
                                # Remove the selected s1_asc and s1_dsc from the original lists to avoid re-selection
                                s1_asc_vrts.remove(s1_asc_vrt)
                                s1_dsc_vrts.remove(s1_dsc_vrt)
                                break
                        break

        if len(valid_vrts) > 0:
            return valid_vrts
        return None
    except Exception as e:
        logging.warning(f"Skipping geometry {geometry.bounds}: {e}")
        return None


def get_valid_vrts_composites(
    data_dir: Path, geometry: Any, date: str, half_window_size: int
) -> Optional[bool]:
    """Check whether the global S1/S2 composite VRTs have valid data over a geometry.

    Args:
        data_dir: Root data directory containing the `s1/s1_asc.vrt`, `s1/s1_dsc.vrt`, and
            `s2/s2.vrt` composite VRTs.
        geometry: Shapely geometry to check for data validity.
        date: Unused, kept for a signature consistent with `get_valid_vrts_timeseries`.
        half_window_size: Unused, kept for a signature consistent with
            `get_valid_vrts_timeseries`.

    Returns:
        True if all three composite VRTs have valid data over `geometry`, otherwise None.
    """
    bounds = geometry.bounds

    s1_asc_vrt = data_dir / "s1" / "s1_asc.vrt"
    s1_dsc_vrt = data_dir / "s1" / "s1_dsc.vrt"
    s2_vrt = data_dir / "s2" / "s2.vrt"

    valid_vrt = (
        _verify_window(s2_vrt, bounds)
        and _verify_window(s1_asc_vrt, bounds)
        and _verify_window(s1_dsc_vrt, bounds)
    )

    if valid_vrt:
        return valid_vrt
    return None
