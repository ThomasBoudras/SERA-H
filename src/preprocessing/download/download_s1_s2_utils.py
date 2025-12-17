from pathlib import Path
from datetime import datetime, timedelta
from geefetch.cli.download_implementation import load_country_filter_polygon
from geobbox import GeoBoundingBox
from geefetch.data.get import download_s1, download_s2
from geefetch.utils.enums import CompositeMethod, S1Orbit
from rasterio.crs import CRS
import traceback


def extemity_dates_calculation (reference_date, duration):
    reference_date= f"{int(reference_date[:4])}-{int(reference_date[4:6])}-{int(reference_date[6:])}"
    reference_date = datetime.strptime(reference_date, "%Y-%m-%d")

    start_date =  reference_date - timedelta(days=abs(duration))
    final_date = reference_date + timedelta(days=abs(duration))

    start_date= start_date.strftime("%Y-%m-%d")
    final_date = final_date.strftime("%Y-%m-%d")
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
    s1_orbit="BOTH"
):

    start_date, end_date = extemity_dates_calculation(reference_date=reference_date, duration=duration)
    data_dir = data_dir / f"lidar_date_{reference_date}"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    left, bottom, right, top = bounds[0], bounds[1], bounds[2], bounds[3]
    tile_shape = int(min(tile_shape, max(abs(right - left), abs(top - bottom)))) # either tile_size is larger than WH: we want the whole area in one tile without necessarily downloading more, or tile size is smaller: we follow the size of tile_size
    bbox = GeoBoundingBox(left=left, right=right, top=top, bottom=bottom, crs=CRS.from_epsg(2154))

    composite_method_s1 = getattr(CompositeMethod, composite_method_s1)
    composite_method_s2 = getattr(CompositeMethod, composite_method_s2)
    s1_orbit = getattr(S1Orbit, s1_orbit)
    
    
    try:
        download_s2(
            data_dir=data_dir,
            ee_project_ids=ee_project_ids,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            composite_method=composite_method_s2,
            crs=CRS.from_epsg(2154),
            resolution=resolution,
            tile_shape=tile_shape,
            max_tile_size=max_tile_size,
            cloudless_portion=cloudless_portion,
            cloud_prb_thresh=cloud_prb,
            filter_polygon=(None if country is None else load_country_filter_polygon(country)),
        )

        download_s1(
            data_dir=data_dir,
            ee_project_ids=ee_project_ids,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            crs=CRS.from_epsg(2154),
            resolution=resolution,
            tile_shape=tile_shape,
            max_tile_size=max_tile_size,
            composite_method=composite_method_s1,
            filter_polygon=(None if country is None else load_country_filter_polygon(country)),
            orbit=s1_orbit
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        raise Exception(f"Download failed from {start_date} to {end_date}:\n{error_trace}") from e
 