"""Utilities to download Sentinel-1/Sentinel-2 imagery around a reference date via geefetch.

This module wraps `geefetch`'s `download_s1`/`download_s2` to download imagery for a bounding
box and date window centered on a reference date, optionally restricted to a country and/or an
arbitrary polygon.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, List, Optional, Tuple

from geefetch.cli.download_implementation import load_country_filter_polygon
from geefetch.data.get import download_s1, download_s2
from geefetch.utils.enums import CompositeMethod, S1Orbit
from geefetch.utils.rasterio import WGS84
from geobbox import GeoBoundingBox
from pyproj import Transformer
from rasterio.crs import CRS
from shapely.ops import transform as shapely_transform


def extemity_dates_calculation(reference_date: str, duration: int) -> Tuple[str, str]:
    """Compute the start/end dates of a window of `duration` days on each side of a reference date.

    If the computed start date would fall before 2017-03-28 (the start of Sentinel data
    availability used here), the window is shifted so that it starts at 2017-03-28 while keeping
    the same total window length.

    Args:
        reference_date: Reference date string, formatted as YYYYMMDD.
        duration: Number of days to extend before and after the reference date.

    Returns:
        A tuple `(start_date, end_date)`, each formatted as YYYY-MM-DD.
    """
    reference_date = (
        f"{int(reference_date[:4])}-{int(reference_date[4:6])}-{int(reference_date[6:])}"
    )
    reference_date = datetime.strptime(reference_date, "%Y-%m-%d")

    start_date = reference_date - timedelta(days=abs(duration))
    final_date = reference_date + timedelta(days=abs(duration))

    start_date = start_date.strftime("%Y-%m-%d")
    final_date = final_date.strftime("%Y-%m-%d")

    if start_date < "2017-03-28":
        start_date = "2017-03-28"
        final_date = (
            datetime.strptime("2017-03-28", "%Y-%m-%d") + 2 * timedelta(days=abs(duration))
        ).strftime("%Y-%m-%d")

    return start_date, final_date


def download_s1_s2(
    data_dir: Path,
    ee_project_ids: List[str],
    resolution: int,
    tile_shape: int,
    max_tile_size: int,
    cloudless_portion: int,
    cloud_prb: int,
    country: Optional[str],
    composite_method_s2: str,
    composite_method_s1: str,
    bounds: Tuple[float, float, float, float],
    reference_date: str,
    duration: int,
    s1_orbit: str = "BOTH",
    filter_polygon: Optional[Any] = None,
    bounds_crs: int = 2154,
) -> None:
    """Download Sentinel-2 then Sentinel-1 imagery for a bounding box around a reference date.

    The date window is computed from `reference_date` and `duration` via
    `extemity_dates_calculation`. If provided, `filter_polygon` (in `bounds_crs`) is reprojected
    to WGS84 and, if `country` is also provided, intersected with the country's filter polygon,
    to restrict the tiles that are actually downloaded.

    Args:
        data_dir: Directory where the downloaded imagery is saved; created if missing.
        ee_project_ids: Google Earth Engine project ids to use.
        resolution: Output pixel resolution, in the units of `bounds_crs`.
        tile_shape: Tile size (in pixels) used to split the bounding box for download.
        max_tile_size: Maximum tile size allowed for a single download request.
        cloudless_portion: Minimum cloudless portion (percent) required for S2 tiles.
        cloud_prb: Cloud probability threshold (percent) used for S2 cloud masking.
        country: Country name used to further restrict the download area, or None to disable.
        composite_method_s2: Name of the `CompositeMethod` enum member to use for S2.
        composite_method_s1: Name of the `CompositeMethod` enum member to use for S1.
        bounds: Bounding box `(left, bottom, right, top)` in `bounds_crs`.
        reference_date: Reference date string, formatted as YYYYMMDD.
        duration: Number of days to extend the download window before and after
            `reference_date`.
        s1_orbit: Name of the `S1Orbit` enum member selecting which orbit(s) to download.
        filter_polygon: Optional geometry (in `bounds_crs`) further restricting which tiles are
            downloaded.
        bounds_crs: EPSG code of the CRS in which `bounds` and `filter_polygon` are expressed.
    """
    start_date, end_date = extemity_dates_calculation(
        reference_date=reference_date, duration=duration
    )

    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    left, bottom, right, top = bounds[0], bounds[1], bounds[2], bounds[3]
    bbox_crs = CRS.from_user_input(bounds_crs)
    bbox = GeoBoundingBox(left=left, right=right, top=top, bottom=bottom, crs=bbox_crs)

    # filter_polygon (expected in bbox_crs) restricts which tiles are downloaded to the ones
    # that actually intersect it, once reprojected to WGS84 below.
    if country is not None:
        country_polygon = load_country_filter_polygon(country)

    if filter_polygon is not None:
        # filter_polygon is given in bbox_crs (EPSG:2154 by default); reproject it to WGS84
        # so it can be compared to the WGS84 tile bounds in Tiler.split.
        to_wgs84 = Transformer.from_crs(bbox_crs, WGS84, always_xy=True).transform
        filter_polygon = shapely_transform(to_wgs84, filter_polygon)

    if country is not None and filter_polygon is not None:
        filter_polygon = filter_polygon.intersection(country_polygon)

    composite_method_s1 = getattr(CompositeMethod, composite_method_s1)
    composite_method_s2 = getattr(CompositeMethod, composite_method_s2)
    s1_orbit = getattr(S1Orbit, s1_orbit)

    logging.info(
        f"downloading s2 and s1 from {start_date} to {end_date}, tile shape: {tile_shape}"
    )

    download_s2(
        data_dir=data_dir,
        ee_project_ids=ee_project_ids,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        composite_method=composite_method_s2,
        crs=bbox_crs,
        resolution=resolution,
        tile_shape=tile_shape,
        max_tile_size=max_tile_size,
        cloudless_portion=cloudless_portion,
        cloud_prb_thresh=cloud_prb,
        filter_polygon=filter_polygon,
    )

    download_s1(
        data_dir=data_dir,
        ee_project_ids=ee_project_ids,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        crs=bbox_crs,
        resolution=resolution,
        tile_shape=tile_shape,
        max_tile_size=max_tile_size,
        composite_method=composite_method_s1,
        filter_polygon=filter_polygon,
        orbit=s1_orbit,
    )
