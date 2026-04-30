import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import json
from src.global_utils  import get_window
from src.global_utils import get_logger
import random

log = get_logger(__name__)


class getS1S2Timeseries :
    def __init__(
        self,
        inputs_path,
        resolution_input,
        nb_timeseries_image,
        nb_timeseries_image_min,
        nb_acquisition_days_max,
        duplication_level_noise,
        vrts_column,
        date_column,
        resampling_method,
        open_even_oob,
        input_combination_mode,
    ):
        """
        nb_timeseries_image : Number of tuple (S2, S1 asc, S1 dsc)
        """
        
        self.inputs_path = Path(inputs_path).resolve()
        self.resolution_input = resolution_input
        self.nb_timeseries_image = nb_timeseries_image
        self.nb_timeseries_image_min = nb_timeseries_image_min
        self.nb_acquisition_days_max = nb_acquisition_days_max
        self.duplication_level_noise = duplication_level_noise
        self.vrts_column = vrts_column
        self.date_column = date_column
        self.resampling_method = resampling_method
        self.open_even_oob = open_even_oob
        self.mean = None
        self.std = None
        self.input_combination_mode = input_combination_mode


    def get_date(self, str_date) :
        return datetime.strptime(str_date, "%Y%m%d")

    def filter_dates(self, vrts_list, ref_date_str):
        ref_date = self.get_date(ref_date_str)
        filtered = [
            dates for dates in vrts_list
            if abs((self.get_date(dates[0]) - ref_date).days) <= self.nb_acquisition_days_max/2 # nb_acquisition_days_max is the total number of days of acquisition, before and after the reference date
        ]
        return filtered

    def prepare_gdf_for_inputs(self, gdf) :        
        # For each row, filter the list in vrts_column to keep only dates within nb_acquisition_days_max days of the reference date (date_column)
        if self.nb_acquisition_days_max is not None:
            gdf[self.vrts_column] = [
                self.filter_dates(vrts, ref_date)
                for vrts, ref_date in zip(gdf[self.vrts_column], gdf[self.date_column])
            ]
        
        # After filtering, keep only rows with at least nb_timeseries_image_min valid tuples
        gdf = gdf[gdf[self.vrts_column].map(len) >= self.nb_timeseries_image_min].reset_index(drop=True)
        return gdf
    
    def get_n_closest_dates(
        self,
        dates_list,
        reference_date,
    ) :
        """
        Returns the nb_timeseries_image tuples (S2, S1 asc, S1 dsc) whose S2 date is closest to the reference date,
        sorted by ascending order of the S2 date in each tuple.
        """
        n_dates = min(self.nb_timeseries_image, len(dates_list))
        ref_date = self.get_date(reference_date)
        
        # Sort the list by proximity to the reference date
        dates_list = sorted(
            dates_list,
            key=lambda dates: abs(self.get_date(dates[0]) - ref_date)
        )
        
        # Take the n_dates closest elements and sort them by ascending order of date
        dates_list = sorted(
        dates_list[:n_dates],
        key=lambda dates: self.get_date(dates[0])
        )
        
        return dates_list

    def __len__(self) :
        return len(self.gdf)
    
    def get_input_combination_processing(self) :
        if self.input_combination_mode == "s2_s1asc_s1dsc" :
            return True, True, True
        elif self.input_combination_mode == "s2":
            return True, False, False
        elif self.input_combination_mode == "s1asc_s1dsc" :
            return False, True, True
        elif self.input_combination_mode == "s2_s1rdm" :
            heads_or_tails = random.choice([1, 2])
            if heads_or_tails == 1 :
                return True, True, False
            else :
                return True, False, True
        else :
            raise ValueError(f"Invalid input combination mode: {self.input_combination_mode}")

    def __call__(self, bounds, row, transform) :        
        vrts = row[self.vrts_column]
        lidar_date = row[self.date_column]

        file_lidar_date = lidar_date[:6] + "15" # files are stored with the date of the middle of the month
        s2_vrts_path = self.inputs_path / "s2" / "s2_vrts"
        s1_asc_vrts_path = self.inputs_path / "s1" / "s1_asc_vrts"
        s1_dsc_vrts_path = self.inputs_path / "s1" / "s1_dsc_vrts"
        
        vrts = self.get_n_closest_dates(vrts, lidar_date)

        #Load input
        inputs = []
        inputs_dates = []
        for s2_s1_vrt in vrts :
            s2_vrt = s2_vrts_path / f"{s2_s1_vrt[0]}.vrt"
            s1_asc_vrt = s1_asc_vrts_path / f"{s2_s1_vrt[1]}.vrt"
            s1_dsc_vrt = s1_dsc_vrts_path / f"{s2_s1_vrt[2]}.vrt"

            images = []

            get_s2_image, get_s1_asc_image, get_s1_dsc_image = self.get_input_combination_processing()

            if get_s2_image:
                s2_image, _ = get_window(
                    image_path=s2_vrt,
                    bounds=bounds,
                    resolution=self.resolution_input,
                    resampling_method = self.resampling_method,
                    open_even_oob = self.open_even_oob,
                )
                images.append(s2_image)
                if s2_image is None:
                    raise ValueError(f"s2_image is None for bounds: {bounds}, file_lidar_date: {file_lidar_date}, s2_s1_vrt: {s2_s1_vrt}")
            
            if get_s1_asc_image:
                s1_asc_image, _ = get_window(
                    image_path=s1_asc_vrt,
                    bounds=bounds,
                    resolution=self.resolution_input,
                    resampling_method = self.resampling_method,
                    open_even_oob = self.open_even_oob,
                )
                images.append(s1_asc_image)
                if s1_asc_image is None:
                    raise ValueError(f"s1_asc_image is None for bounds: {bounds}, file_lidar_date: {file_lidar_date}, s2_s1_vrt: {s2_s1_vrt}")
            
            if get_s1_dsc_image:
                s1_dsc_image, _ = get_window(
                    image_path=s1_dsc_vrt,
                    bounds=bounds,
                    resolution=self.resolution_input,
                    resampling_method = self.resampling_method,
                    open_even_oob = self.open_even_oob,
                )
                images.append(s1_dsc_image)
                if s1_dsc_image is None:
                    raise ValueError(f"s1_dsc_image is None for bounds: {bounds}, file_lidar_date: {file_lidar_date}, s2_s1_vrt: {s2_s1_vrt}")
            
            images = np.concatenate(images, axis=0)
            images = images.astype(np.float32).transpose(1, 2, 0)
            images[~np.isfinite(images)] = 0
            
            inputs.append(images)
            inputs_dates.append(int(self.get_date(s2_s1_vrt[0]).timetuple().tm_yday)) # The date is given as the number of the day in the year of s2

        # If the number of images is less than the required number, we randomly repeat 
        # the images in the time series until we reach the right number.
        while len(inputs) < self.nb_timeseries_image :
            idx = np.random.randint(0, len(inputs))
            image_added = inputs[idx] 
            date_added = inputs_dates[idx]

            if self.duplication_level_noise is not None and self.std is not None: 
                noise = np.random.normal(
                    0,
                    self.duplication_level_noise * self.std,
                    size=image_added.shape
                ).astype(np.float32)
                image_added = image_added + noise 

            inputs.insert(idx + 1, image_added)
            inputs_dates.insert(idx + 1, date_added)
        
        if transform:
            inputs = torch.stack([transform(images) for images in inputs], dim=0)
        else : 
            inputs = torch.stack([torch.from_numpy(images) for images in inputs], dim=0)

        metadata = {"inputs_dates": torch.tensor(inputs_dates), "nb_real_timeseries_images": torch.tensor([len(vrts)])} 
        return inputs, metadata
    

