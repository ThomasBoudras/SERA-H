import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import rasterio
import rasterio
from collections import defaultdict
from pathlib import Path
import shutil
from src.global_utils  import get_window
from geefetch.utils.rasterio import create_vrt
from tqdm import tqdm
from datetime import timedelta
import logging

def _create_vrts_from_dict(vrts_path: Path, dict_vrt: dict):
    if vrts_path.exists() :
        shutil.rmtree(vrts_path)
    vrts_path.mkdir(parents=True)

    for date, list_tif in tqdm(dict_vrt.items(), desc="Creating VRTs from dict", total=len(dict_vrt)):
        if list_tif :
            vrt_path = vrts_path / (date + ".vrt")
            create_vrt(vrt_path, list_tif)

def create_vrts_timeseries(data_dir):
    
    s1_path = data_dir / "s1"
    s2_path = data_dir / "s2" 

    s1_vrts_asc_path = s1_path / "s1_asc_vrts"
    s1_vrts_dsc_path = s1_path / "s1_dsc_vrts" 
    s2_vrts_path = s2_path / "s2_vrts"

    logging.info(f"Regrouping S1 files by date and orbit")
    dict_vrt_s1_asc = defaultdict(list) 
    dict_vrt_s1_dsc = defaultdict(list) 
    list_path_s1 = list(s1_path.rglob("*tif"))
    for path in tqdm(list_path_s1, desc="Regrouping S1 files by date and orbit", total=len(list_path_s1)) :
        date_s1 = path.stem.split("_")[4][:8]
        with rasterio.open(path) as src:
            s1_orbit = src.tags()["orbitProperties_pass"]
        if s1_orbit == "ASCENDING" :
            dict_vrt_s1_asc[date_s1].append(path)
        else :
            dict_vrt_s1_dsc[date_s1].append(path)

    logging.info(f"Regrouping S2 files by date")
    dict_vrt_s2 = defaultdict(list)
    list_path_s2 = list(s2_path.rglob("*tif"))
    for path in tqdm(list_path_s2, desc="Regrouping S2 files by date", total=len(list_path_s2)) :
        date_s2 = path.stem[:8]
        dict_vrt_s2[date_s2].append(path)

    logging.info(f"Creating VRTs for S1 ASC")
    _create_vrts_from_dict(s1_vrts_asc_path, dict_vrt_s1_asc)
    logging.info(f"Creating VRTs for S1 DSC")
    _create_vrts_from_dict(s1_vrts_dsc_path, dict_vrt_s1_dsc)
    logging.info(f"Creating VRTs for S2")
    _create_vrts_from_dict(s2_vrts_path, dict_vrt_s2)


def create_vrts_composites(data_dir):
    
    s1_path = data_dir / "s1"
    s2_path = data_dir / "s2" 

    s1_asc_vrt_path = s1_path / "s1_asc.vrt"
    s1_dsc_vrt_path = s1_path / "s1_dsc.vrt"
    s2_vrt_path = s2_path / "s2.vrt"
    
    list_tif_s1_asc = []
    list_tif_s1_dsc = []   
    for path in s1_path.rglob("*tif") :
        if "asc" in  path.stem :
            list_tif_s1_asc.append(path)
        elif "desc" in  path.stem :
            list_tif_s1_dsc.append(path)
        else:
            raise ValueError(f"File {path.stem} is not a valid S1 file")

    list_tif_s2 = list(s2_path.rglob("*tif")) 

    create_vrt(s1_asc_vrt_path, list_tif_s1_asc)
    create_vrt(s1_dsc_vrt_path, list_tif_s1_dsc)
    create_vrt(s2_vrt_path, list_tif_s2)



def _get_date_from_vrt_name(vrt_name) :
    return datetime.strptime(vrt_name.stem, "%Y%m%d")

def filter_files_by_date_gap(reference_date, file_list, max_days_gap):
    """
    Filters files by maximum allowed date gap from the reference date.
    reference_date: str (expected format: YYYYMMDD)
    file_list: list of Path objects with date info in their names
    max_days_gap: int
    """
    ref_date = datetime.strptime(reference_date, "%Y%m%d")
    filtered_files = []
    for file in file_list:
        file_date = _get_date_from_vrt_name(file)
        days_diff = abs((file_date - ref_date).days)
        if days_diff <= max_days_gap:
            filtered_files.append(file)
    return filtered_files

def _sort_by_proximity(target_file, file_list):
    """
    Sorts a list of files by proximity in days to the target_file.
    Args:
        target_file: Path-like, the reference file whose date is used.
        file_list: list of Path-like objects to compare.
        max_days_gap: int or None. Maximum allowed days between dates to be included, else ignores. 
            E.g., if 10, only files within +/-10 days are considered.

    Returns:
        List of files sorted by date proximity to target_file (filtered if max_days_gap given).
    """
    target_date = _get_date_from_vrt_name(target_file)
    files_with_gap = [
        (file, abs((_get_date_from_vrt_name(file) - target_date).days))
        for file in file_list
    ]
    sorted_files = sorted(files_with_gap, key=lambda x: x[1])
    return [file for file, gap in sorted_files]

def _verify_window(path: Path, bounds) -> bool:
    """
    Verify if we have data for this windows
    """
    img, _  = get_window(
        image_path=path, 
        bounds=bounds,
        resolution=None,
        resampling_method="bilinear",
    )
    return img is not None and bool(img.size) and np.isfinite(img).any() # type: ignore

def get_valid_vrts_timeseries(data_dir, geometry, date, half_window_size):
        bounds = geometry.bounds 
        valid_vrts = []
        
        limit_date = datetime.strptime("20170328", "%Y%m%d") + timedelta(days=half_window_size)
        limit_date_str = limit_date.strftime("%Y%m%d")
        if date < limit_date_str:
            date = limit_date_str


        s1_asc_vrts_path = data_dir / "s1" / "s1_asc_vrts"
        s1_dsc_vrts_path = data_dir / "s1" / "s1_dsc_vrts"
        s2_vrts_path = data_dir / "s2" / "s2_vrts"

        s1_asc_vrts = [file for file in s1_asc_vrts_path.iterdir() if file.suffix == '.vrt']
        s1_asc_vrts = filter_files_by_date_gap(date, s1_asc_vrts, half_window_size)
        
        s1_dsc_vrts = [file for file in s1_dsc_vrts_path.iterdir() if file.suffix == '.vrt']
        s1_dsc_vrts = filter_files_by_date_gap(date, s1_dsc_vrts, half_window_size)
        
        s2_vrts = [file for file in s2_vrts_path.iterdir() if file.suffix == '.vrt']
        s2_vrts = filter_files_by_date_gap(date, s2_vrts, half_window_size)

        for s2_vrt in s2_vrts:
            if _verify_window(s2_vrt, bounds): 
                sorted_s1_asc_vrts = _sort_by_proximity(s2_vrt, s1_asc_vrts)  # We are looking for the nearest tensor s1 in terms of date
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
        
        if len(valid_vrts) > 0 :
            return valid_vrts
        return None
    
def get_valid_vrts_composites(data_dir, geometry, date, half_window_size):
    bounds = geometry.bounds 

    s1_asc_vrt = data_dir / "s1" / "s1_asc.vrt"
    s1_dsc_vrt = data_dir / "s1" / "s1_dsc.vrt"
    s2_vrt = data_dir / "s2" / "s2.vrt"

    valid_vrt = (_verify_window(s2_vrt, bounds) and
                 _verify_window(s1_asc_vrt, bounds) and
                 _verify_window(s1_dsc_vrt, bounds))

    if valid_vrt :
        return valid_vrt
    return None
