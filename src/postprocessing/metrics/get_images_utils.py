import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import binary_dilation, binary_erosion
from src.global_utils  import get_window
from shapely.geometry import shape
from shapely.geometry import box
from rasterio.features import rasterize
from pathlib import Path
from datetime import datetime

class get_images:
    def __init__(self, image_loaded_set, image_computed_set) :
        self.image_loaded_set = image_loaded_set
        self.image_computed_set = image_computed_set

    def __call__(self, row):
        bounds = row["geometry"].bounds

        res_images = {}

        #Retrieving images to be loaded 
        for name_image, image_loader in self.image_loaded_set.items() :
            if image_loader is not None:
                image = image_loader.load_image(bounds, row)
                res_images[name_image] = image 

        #Retrieving images to be calculated if they are not None
        if self.image_computed_set is not None:         
            for name_image, image_computer in self.image_computed_set.items() :
                if image_computer is not None:
                    images_value = image_computer.compute_image(res_images, row)
                    if isinstance(images_value, dict):
                        for key, value in images_value.items():
                            res_images[f"{name_image}_{key}"] = value
                    else:
                        res_images[name_image] = images_value

        return res_images

def get_delta_date(date_ref, date_vrt):
    """
    Retourne le nombre de jours entre deux dates (YYYYMMDD ou YYYY-MM-DD).
    """
    date_vrt = Path(date_vrt).stem
    date_ref = datetime.strptime(date_ref, "%Y%m%d")
    date_vrt = datetime.strptime(date_vrt, "%Y%m%d")
    delta = abs((date_vrt - date_ref).days)
    return delta

class input_image_loader :
    def __init__(self, path, resolution, resampling_method, open_even_oob, channel_to_keep, grouping_dates):
        self.path = path
        self.resolution = resolution
        self.resampling_method = resampling_method
        self.open_even_oob = open_even_oob
        self.channel_to_keep = channel_to_keep
        self.grouping_dates = grouping_dates

    def load_image(self, bounds, row):
        if self.grouping_dates:
            date = row[self.grouping_dates]
            date = date[:6] + "15"

            if "<date>" in self.path :
                path = self.path.replace("<date>", date)

            elif "<year>" in self.path :
                path = self.path.replace("<year>", date[:4])

            else:
                path = self.path
        else:
            path = self.path

        if Path(path).is_dir():
            paths = list(Path(path).iterdir())
        else:
            paths = [path]

        images_input = []
        for path in paths:

            if self.grouping_dates and get_delta_date(date, path) > 60:
                continue
            
            image, _ = get_window(
                path,
                bounds=bounds,
                resolution=self.resolution,
                resampling_method=self.resampling_method,
                open_even_oob=self.open_even_oob
                )

            if image is not None and np.isfinite(image).any():   
                images_input.append(image)
            
        image = np.median(images_input, axis=0)

        if self.channel_to_keep is not None:
            image = image[self.channel_to_keep, ...]
        image = image.astype(np.float32)
        # Normalisation min-max
        image[~np.isfinite(image)] = 0
        image_min = np.nanmin(image)
        image_max = np.nanmax(image)
        image = (image - image_min) / (image_max - image_min)

        return image


class output_image_loader :
    def __init__(self, path, resolution, resampling_method, scaling_factor, min_image, max_image, open_even_oob, max_date, min_date, grouping_dates):
        self.path = path
        self.resolution = resolution
        self.resampling_method = resampling_method
        self.scaling_factor = scaling_factor
        self.min_image = int(min_image) if min_image is not None else None
        self.max_image = int(max_image) if max_image is not None else None
        self.open_even_oob = open_even_oob
        self.max_date = max_date 
        self.min_date = min_date
        self.grouping_dates = grouping_dates

    def load_image(self, bounds, row):
        if self.grouping_dates:
            date = row[self.grouping_dates]
                
            if self.max_date is not None:
                date = str(min(int(date), int(self.max_date)))

            if self.min_date is not None:
                date = str(max(int(date), int(self.min_date)))
            
            date = date[:6] + "15"

        path = self.path
        if "<year>" in self.path :
            path = path.replace("<year>", date[:4])
        if "<date>" in self.path :
            path = path.replace("<date>", date)
        if "<area>" in self.path :
            path = path.replace("<area>", row["area_name"])

        image, profile = get_window(
            path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob
            )
        if image is None:
            print(f"Image is None for path {path}")
            print(f"Bounds: {bounds}")
        image = image.astype(np.float32)*self.scaling_factor
        image = np.clip(image, self.min_image, self.max_image).squeeze()
        return image


class mask_image_loader :
    def __init__(self, classification_mask_path, forest_mask_path, resolution, classes_to_keep, grouping_dates):
        self.classification_mask_path = classification_mask_path
        self.forest_mask_gdf = gpd.read_parquet(forest_mask_path) if forest_mask_path is not None else None
        self.resolution = resolution
        self.classes_to_keep = classes_to_keep
        self.grouping_dates = grouping_dates
        
    def load_image(self, bounds, row):
        date = row[self.grouping_dates]          
        year = date[:4]
        raster_bounds = box(*bounds)
        mask_path = self.classification_mask_path.replace("<year>", year)
        classification, profile = get_window(
            mask_path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method="nearest",
        )
        classification = classification.squeeze()

        classif_mask = classification == self.classes_to_keep[0]
        if len(self.classes_to_keep) > 1:
            for aclass in self.classes_to_keep[1::]:
                classif_mask = classif_mask | (classification == aclass)
    
        if self.forest_mask_gdf is not None:
            clipped_gdf = gpd.clip(self.forest_mask_gdf, raster_bounds)     
            geometries = [(geom, 1) for geom in clipped_gdf.geometry]
            if len(geometries):
                mask_forest = rasterize(
                    geometries,
                    out_shape=classification.shape,
                    transform=profile["transform"],
                    fill=0,
                    default_value=1,
                    dtype=np.uint8,
                ).astype(bool)
            else:
                mask_forest = np.zeros_like(classif_mask, dtype=bool)
        else :
            mask_forest = np.zeros_like(classif_mask, dtype=bool)
        
        final_mask = classif_mask | mask_forest
        return final_mask


class masked_image_computer :
    def __init__(self, input_name, mask_name):
        self.input_name = input_name
        self.mask_name = mask_name

    def compute_image(self, res_images, row):
        if self.input_name not in res_images:
            Exception(f"You must first load {self.input_name}")
            
        image = res_images[self.input_name].copy()
        mask = res_images[self.mask_name].copy()

        image[~mask] = np.nan
        return image


class difference_computer :
    def __init__(self, input_name_1, input_name_2, min_height, max_height, min_difference, max_difference, threshold_forest):
        self.input_name_1 = input_name_1
        self.input_name_2 = input_name_2
        self.max_height = max_height
        self.min_height = min_height
        self.min_difference = int(min_difference) if min_difference is not None else None
        self.max_difference = int(max_difference) if max_difference is not None else None
        self.threshold_forest = threshold_forest

    def compute_image(self, res_images, row):
        if self.input_name_1 not in res_images or  self.input_name_2 not in res_images :
            Exception(f"You must first load {self.input_name_1} and {self.input_name_2}")
        
        image_1 = res_images[self.input_name_1].copy()
        image_2 = res_images[self.input_name_2].copy()

        difference = image_2 - image_1

        if self.max_height is not None and self.min_height is not None:
            range_mask = (image_1 >= self.min_height) & (image_2 >= self.min_height) & (image_1 <= self.max_height) & (image_2 <= self.max_height)
            difference[~range_mask] = np.nan

        if self.threshold_forest is not None:
            non_forest_mask = (image_1 < self.threshold_forest) & (image_2 < self.threshold_forest)
            difference[non_forest_mask & np.isfinite(difference)] = 0
        
        # Mask the difference outside the range of valid differences
        mask_diff_range = (difference < self.min_difference) | (difference > self.max_difference)
        difference[mask_diff_range] = np.nan
        return difference
  