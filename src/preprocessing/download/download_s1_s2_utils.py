from pathlib import Path
from datetime import datetime, timedelta
from geefetch.cli.download_implementation import load_country_filter_polygon
from geefetch.utils.rasterio import WGS84
from geobbox import GeoBoundingBox
from geefetch.data.get import download_s1, download_s2
from geefetch.utils.enums import CompositeMethod, S1Orbit
from pyproj import Transformer
from rasterio.crs import CRS
from shapely.ops import transform as shapely_transform
import logging


def extemity_dates_calculation (reference_date, duration):
    reference_date= f"{int(reference_date[:4])}-{int(reference_date[4:6])}-{int(reference_date[6:])}"
    reference_date = datetime.strptime(reference_date, "%Y-%m-%d")

    start_date =  reference_date - timedelta(days=abs(duration))
    final_date = reference_date + timedelta(days=abs(duration))

    start_date= start_date.strftime("%Y-%m-%d")
    final_date = final_date.strftime("%Y-%m-%d")

    if start_date < "2017-03-28":
        start_date = "2017-03-28"
        final_date = (datetime.strptime("2017-03-28", "%Y-%m-%d") + 2*timedelta(days=abs(duration))).strftime("%Y-%m-%d")

    return start_date, final_date

def download_s1_s2(
    data_dir : Path,
    ee_project_ids,
    resolution,
    tile_shape,
    max_tile_size,
    cloudless_portion,
    cloud_prb,
    country,
    composite_method_s2,
    composite_method_s1,
    bounds,
    reference_date,
    duration,
    s1_orbit="BOTH",
    filter_polygon=None,
    bounds_crs=2154,
):

    start_date, end_date = extemity_dates_calculation(reference_date=reference_date, duration=duration)

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

    logging.info(f"downloading s2 and s1 from {start_date} to {end_date}, tile shape: {tile_shape}")

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
        orbit=s1_orbit
    )
