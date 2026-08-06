"""Loaders and computers that build the per-sample image dictionary used for metrics/plots.

Image loaders read rasters/masks from disk for a given sample bounds, while
image computers derive additional images (differences, masks, evaluation
masks) from already loaded/computed images.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import box
from skimage.measure import label

from src.global_utils import get_window


class get_images:
    """Builds the dictionary of loaded and computed images for a single sample.

    Args:
        image_loaded_set: Mapping from image name to a loader object exposing a
            `load_image(bounds, row)` method (or None to skip).
        image_computed_set: Mapping from image name to a computer object
            exposing a `compute_image(res_images, row)` method (or None), or
            None to skip all computed images.
    """

    def __init__(
        self, image_loaded_set: Dict[str, Any], image_computed_set: Optional[Dict[str, Any]]
    ) -> None:
        self.image_loaded_set = image_loaded_set
        self.image_computed_set = image_computed_set

    def __call__(self, row: pd.Series) -> Dict[str, np.ndarray]:
        """Loads and computes all configured images for a sample.

        Args:
            row: GeoDataFrame row corresponding to the sample; its `geometry`
                column bounds are used to load images.

        Returns:
            Dictionary mapping image names to their loaded/computed arrays.
            When a computer returns a dict, its keys are prefixed with the
            image name.
        """
        bounds = row["geometry"].bounds

        res_images = {}

        # Retrieving images to be loaded
        for name_image, image_loader in self.image_loaded_set.items():
            if image_loader is not None:
                image = image_loader.load_image(bounds, row)
                res_images[name_image] = image

        # Retrieving images to be calculated if they are not None
        if self.image_computed_set is not None:
            for name_image, image_computer in self.image_computed_set.items():
                if image_computer is not None:
                    images_value = image_computer.compute_image(res_images, row)
                    if isinstance(images_value, dict):
                        for key, value in images_value.items():
                            res_images[f"{name_image}_{key}"] = value
                    else:
                        res_images[name_image] = images_value

        return res_images


def get_delta_date(date_ref: str, date_vrt: Union[str, Path]) -> int:
    """Returns the number of days between two dates (YYYYMMDD format).

    Args:
        date_ref: Reference date as a `YYYYMMDD` string.
        date_vrt: Path to a VRT/raster file whose stem is a `YYYYMMDD` date, or
            the date string itself.

    Returns:
        The absolute number of days between the two dates.
    """
    date_vrt = Path(date_vrt).stem
    date_ref = datetime.strptime(date_ref, "%Y%m%d")
    date_vrt = datetime.strptime(date_vrt, "%Y%m%d")
    delta = abs((date_vrt - date_ref).days)
    return delta


class input_image_loader:
    """Loads and normalizes an input raster image (e.g. Sentinel imagery) for a sample.

    Args:
        path: Path (possibly a directory, or containing `<date>`/`<year>`
            placeholders) to the raster(s) to load.
        resolution: Target pixel resolution used when reading the window.
        resampling_method: Resampling method used when reading the window.
        open_even_oob: Whether to allow opening windows that are out of bounds.
        channel_to_keep: Indices of the channels to keep, or None to keep all.
        grouping_dates: Name of the row column holding the date used to resolve
            `<date>`/`<year>` placeholders, or falsy to disable date grouping.
    """

    def __init__(
        self,
        path: str,
        resolution: float,
        resampling_method: str,
        open_even_oob: bool,
        channel_to_keep: Optional[list],
        grouping_dates: Optional[str],
    ) -> None:
        self.path = path
        self.resolution = resolution
        self.resampling_method = resampling_method
        self.open_even_oob = open_even_oob
        self.channel_to_keep = channel_to_keep
        self.grouping_dates = grouping_dates

    def load_image(self, bounds: Tuple[float, float, float, float], row: pd.Series) -> np.ndarray:
        """Loads, aggregates (median) and min-max normalizes the input image.

        Args:
            bounds: `(minx, miny, maxx, maxy)` bounding box of the sample.
            row: GeoDataFrame row corresponding to the sample; used to resolve
                date-based path placeholders when `grouping_dates` is set.

        Returns:
            The normalized image array with values in `[0, 1]`.
        """
        if self.grouping_dates:
            date = row[self.grouping_dates]
            date = date[:6] + "15"

            if "<date>" in self.path:
                path = self.path.replace("<date>", date)

            elif "<year>" in self.path:
                path = self.path.replace("<year>", date[:4])

            else:
                path = self.path
        else:
            path = self.path

        path = Path(path).resolve()
        if path.is_dir():
            paths = list(path.iterdir())
        else:
            paths = [path]

        images_input = []
        for path in paths:
            if self.grouping_dates and "sentinel" in str(path) and get_delta_date(date, path) > 60:
                continue

            image, _ = get_window(
                path,
                bounds=bounds,
                resolution=self.resolution,
                resampling_method=self.resampling_method,
                open_even_oob=self.open_even_oob,
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


class output_image_loader:
    """Loads and rescales a model output raster (e.g. a height prediction) for a sample.

    Args:
        path: Path to the raster, may contain `<date>`, `<year>` and/or `<area>`
            placeholders.
        resolution: Target pixel resolution used when reading the window.
        resampling_method: Resampling method used when reading the window.
        scaling_factor: Multiplicative factor applied to the raw raster values.
        min_image: Minimum value used to clip the scaled image, or None.
        max_image: Maximum value used to clip the scaled image, or None.
        open_even_oob: Whether to allow opening windows that are out of bounds.
        max_date: If set, caps the resolved date at this value.
        min_date: If set, floors the resolved date at this value.
        grouping_dates: Name of the row column holding the date used to resolve
            `<date>`/`<year>` placeholders, or falsy to disable date grouping.
    """

    def __init__(
        self,
        path: str,
        resolution: float,
        resampling_method: str,
        scaling_factor: float,
        min_image: Optional[float],
        max_image: Optional[float],
        open_even_oob: bool,
        max_date: Optional[str],
        min_date: Optional[str],
        grouping_dates: Optional[str],
    ) -> None:
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

    def load_image(self, bounds: Tuple[float, float, float, float], row: pd.Series) -> np.ndarray:
        """Loads, scales and clips the output raster for the sample.

        Args:
            bounds: `(minx, miny, maxx, maxy)` bounding box of the sample.
            row: GeoDataFrame row corresponding to the sample; used to resolve
                date/area-based path placeholders.

        Returns:
            The scaled and clipped image array.
        """
        if self.grouping_dates:
            date = row[self.grouping_dates]

            if self.max_date is not None:
                date = str(min(int(date), int(self.max_date)))

            if self.min_date is not None:
                date = str(max(int(date), int(self.min_date)))

            date = date[:6] + "15"

        path = self.path
        if "<year>" in self.path:
            path = path.replace("<year>", date[:4])
        if "<date>" in self.path:
            path = path.replace("<date>", date)
        if "<area>" in self.path:
            path = path.replace("<area>", row["area_name"])

        image, profile = get_window(
            path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method=self.resampling_method,
            open_even_oob=self.open_even_oob,
        )
        image = image.astype(np.float32) * self.scaling_factor
        image = np.clip(image, self.min_image, self.max_image).squeeze()
        return image


class mask_image_loader:
    """Builds a validity mask from a LidarHD/height classification raster and an optional forest mask.

    Args:
        lidarhd_classification_mask_path: Path to the LidarHD classification
            raster (may contain a `<year>` placeholder), takes precedence over
            `height_classification_mask_path` if both are set.
        height_classification_mask_path: Fallback path used when
            `lidarhd_classification_mask_path` is None.
        ign_forest_mask_path: Optional path to an IGN forest mask GeoParquet
            file, unioned with the classification mask.
        resolution: Target pixel resolution used when reading the window.
        classes_to_keep: List of classification values considered valid.
        grouping_dates: Name of the row column holding the date used to
            determine the year for path resolution.
    """

    def __init__(
        self,
        lidarhd_classification_mask_path: Optional[str],
        height_classification_mask_path: Optional[str],
        ign_forest_mask_path: Optional[str],
        resolution: float,
        classes_to_keep: list,
        grouping_dates: str,
    ) -> None:
        self.lidarhd_classification_mask_path = (
            lidarhd_classification_mask_path
            if lidarhd_classification_mask_path is not None
            else height_classification_mask_path
        )
        self.ign_forest_mask_gdf = (
            gpd.read_parquet(ign_forest_mask_path) if ign_forest_mask_path is not None else None
        )
        self.resolution = resolution
        self.classes_to_keep = classes_to_keep
        self.grouping_dates = grouping_dates

    def load_image(self, bounds: Tuple[float, float, float, float], row: pd.Series) -> np.ndarray:
        """Builds the boolean validity mask for the sample.

        Args:
            bounds: `(minx, miny, maxx, maxy)` bounding box of the sample.
            row: GeoDataFrame row corresponding to the sample; used to resolve
                the year from `grouping_dates`.

        Returns:
            Boolean array, True where the LidarHD classification is in
            `classes_to_keep` or the IGN forest mask applies.
        """
        date = row[self.grouping_dates]
        year = date[:4]
        raster_bounds = box(*bounds)

        # LidarHD classification mask
        lidarhd_mask_path = self.lidarhd_classification_mask_path.replace("<year>", year)
        lidarhd_classification, lidarhd_profile = get_window(
            lidarhd_mask_path,
            bounds=bounds,
            resolution=self.resolution,
            resampling_method="nearest",
        )
        lidarhd_classification = lidarhd_classification.squeeze()

        lidarhd_classif_mask = lidarhd_classification == self.classes_to_keep[0]
        if len(self.classes_to_keep) > 1:
            for aclass in self.classes_to_keep[1::]:
                lidarhd_classif_mask = lidarhd_classif_mask | (lidarhd_classification == aclass)

        # Ign forest mask
        if self.ign_forest_mask_gdf is not None:
            clipped_gdf = gpd.clip(self.ign_forest_mask_gdf, raster_bounds)
            geometries = [(geom, 1) for geom in clipped_gdf.geometry]
            if len(geometries):
                ign_forest_mask = rasterize(
                    geometries,
                    out_shape=lidarhd_classification.shape,
                    transform=lidarhd_profile["transform"],
                    fill=0,
                    default_value=1,
                    dtype=np.uint8,
                ).astype(bool)
            else:
                ign_forest_mask = np.zeros_like(lidarhd_classif_mask, dtype=bool)
        else:
            ign_forest_mask = np.zeros_like(lidarhd_classif_mask, dtype=bool)

        # Final mask: union of lidarhd classification mask and ign forest mask
        final_mask = lidarhd_classif_mask | ign_forest_mask
        return final_mask


class type_forest_mask_loader:
    """Builds a boolean mask for a given forest type (deciduous, coniferous, or mixed).

    Args:
        forest_mask_path: Path to the forest type GeoPackage, filtered by
            `CODE_TFV` prefix.
        resolution: Target pixel resolution used to rasterize the geometries.
        type_forest: One of `"deciduous"`, `"coniferous"` or `"mixed"`.
    """

    def __init__(self, forest_mask_path: str, resolution: float, type_forest: str) -> None:
        self.forest_mask_path = forest_mask_path
        self.resolution = resolution
        self.type_forest = type_forest

    def load_image(self, bounds: Tuple[float, float, float, float], row: pd.Series) -> np.ndarray:
        """Rasterizes the geometries of the selected forest type over the sample bounds.

        Args:
            bounds: `(minx, miny, maxx, maxy)` bounding box of the sample.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            Boolean array, True where the selected forest type is present.
        """
        # Calculate shape and transform from bounds and resolution
        minx, miny, maxx, maxy = bounds
        width = int(np.round((maxx - minx) / self.resolution))
        height = int(np.round((maxy - miny) / self.resolution))

        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)
        out_shape = (height, width)

        # Read only the geometries intersecting the bounding box directly from the GPKG
        clipped_gdf = gpd.read_file(self.forest_mask_path, bbox=bounds)

        if self.type_forest == "deciduous":
            clipped_gdf = clipped_gdf[clipped_gdf["CODE_TFV"].str.startswith(("FF1", "FO1"))]

        elif self.type_forest == "coniferous":
            clipped_gdf = clipped_gdf[clipped_gdf["CODE_TFV"].str.startswith(("FF2", "FO2"))]

        elif self.type_forest == "mixed":
            clipped_gdf = clipped_gdf[clipped_gdf["CODE_TFV"].str.startswith(("FF3", "FO3"))]

        if not clipped_gdf.empty:
            geometries = [(geom, 1) for geom in clipped_gdf.geometry]
            forest_mask = rasterize(
                geometries,
                out_shape=out_shape,
                transform=transform,
                fill=0,
                default_value=1,
                dtype=np.uint8,
            ).astype(bool)
        else:
            forest_mask = np.zeros(out_shape, dtype=bool)

        return forest_mask


class masked_image_computer:
    """Sets pixels of an image to NaN wherever a boolean mask is False.

    Args:
        input_name: Name of the image to mask in `res_images`.
        mask_name: Name of the boolean mask image in `res_images`.
    """

    def __init__(self, input_name: str, mask_name: str) -> None:
        self.input_name = input_name
        self.mask_name = mask_name

    def compute_image(self, res_images: Dict[str, np.ndarray], row: pd.Series) -> np.ndarray:
        """Applies the mask to the input image.

        Args:
            res_images: Dictionary of already loaded/computed images, must
                contain `input_name` and `mask_name`.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            A copy of the input image with masked-out pixels set to NaN.

        Raises:
            Exception: If `input_name` is not present in `res_images`.
        """
        if self.input_name not in res_images:
            raise Exception(f"You must first load {self.input_name}")

        image = res_images[self.input_name].copy()
        mask = res_images[self.mask_name].copy()

        image[~mask] = np.nan
        return image


class detection_edge_mask_computer:
    """Builds pred/ref evaluation masks based on edge-to-edge distance tolerance.

    For each mask (pred and ref), boundary pixels are marked 1 if they lie
    within `tolerance` of a boundary pixel of the other mask, and -1 otherwise.

    Args:
        name_pred: Name of the predicted boolean mask image in `res_images`.
        name_ref: Name of the reference boolean mask image in `res_images`.
        tolerance: Distance tolerance in the same unit as `resolution`.
        resolution: Spatial resolution of a pixel, used to convert `tolerance`
            to pixels.
    """

    def __init__(self, name_pred: str, name_ref: str, tolerance: float, resolution: float) -> None:
        self.name_pred = name_pred
        self.name_ref = name_ref
        self.tolerance_pixels = tolerance / resolution
        self.resolution = resolution

    def compute_image(
        self, res_images: Dict[str, np.ndarray], row: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Computes the pred and ref edge evaluation masks.

        Args:
            res_images: Dictionary of already loaded/computed images, must
                contain `name_pred` and `name_ref`.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            Dictionary with `"pred"` and `"ref"` evaluation masks (values in
            `{-1, 0, 1}`).

        Raises:
            Exception: If `name_pred` or `name_ref` is not present in `res_images`.
        """
        if (self.name_pred not in res_images) or (self.name_ref not in res_images):
            raise Exception(f"You must first load {self.name_pred} and {self.name_ref}")

        mask_pred = res_images[self.name_pred].astype(bool)
        mask_ref = res_images[self.name_ref].astype(bool)

        # Inner contour: boundary pixels adjacent to a background pixel
        edge_pred = mask_pred & ~binary_erosion(mask_pred)
        edge_ref = mask_ref & ~binary_erosion(mask_ref)

        # Euclidean distance (in pixels) from each pixel to the nearest edge pixel
        dist_to_ref_edges = distance_transform_edt(~edge_ref)
        dist_to_pred_edges = distance_transform_edt(~edge_pred)

        # Pred edge pixels: 1 if within tolerance of a ref edge, -1 otherwise
        eval_mask_pred = np.zeros_like(mask_pred, dtype=np.float32)
        eval_mask_pred[edge_pred & (dist_to_ref_edges <= self.tolerance_pixels)] = 1
        eval_mask_pred[edge_pred & (dist_to_ref_edges > self.tolerance_pixels)] = -1

        # Ref edge pixels: 1 if within tolerance of a pred edge, -1 otherwise
        eval_mask_ref = np.zeros_like(mask_ref, dtype=np.float32)
        eval_mask_ref[edge_ref & (dist_to_pred_edges <= self.tolerance_pixels)] = 1
        eval_mask_ref[edge_ref & (dist_to_pred_edges > self.tolerance_pixels)] = -1

        return {"pred": eval_mask_pred, "ref": eval_mask_ref}


class difference_computer:
    """Computes a masked difference image between two input images.

    Args:
        input_name_1: Name of the first (baseline) image in `res_images`.
        input_name_2: Name of the second image in `res_images`; the difference
            is `input_name_2 - input_name_1`.
        min_height: If set (with `max_height`), pixels outside `[min_height,
            max_height]` in either image are masked to NaN.
        max_height: See `min_height`.
        min_difference: Minimum valid difference value; outside values are set to NaN.
        max_difference: Maximum valid difference value; outside values are set to NaN.
        threshold_forest: If set, pixels below this height in both images are
            treated as non-forest and their difference is forced to 0.
    """

    def __init__(
        self,
        input_name_1: str,
        input_name_2: str,
        min_height: Optional[float],
        max_height: Optional[float],
        min_difference: Optional[float],
        max_difference: Optional[float],
        threshold_forest: Optional[float],
    ) -> None:
        self.input_name_1 = input_name_1
        self.input_name_2 = input_name_2
        self.max_height = max_height
        self.min_height = min_height
        self.min_difference = int(min_difference) if min_difference is not None else None
        self.max_difference = int(max_difference) if max_difference is not None else None
        self.threshold_forest = threshold_forest

    def compute_image(self, res_images: Dict[str, np.ndarray], row: pd.Series) -> np.ndarray:
        """Computes the masked difference `input_name_2 - input_name_1`.

        Args:
            res_images: Dictionary of already loaded/computed images, must
                contain `input_name_1` and `input_name_2`.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            The difference image, with out-of-range/non-forest pixels handled
            as described in the class docstring.

        Raises:
            Exception: If `input_name_1` or `input_name_2` is not present in `res_images`.
        """
        if self.input_name_1 not in res_images or self.input_name_2 not in res_images:
            raise Exception(f"You must first load {self.input_name_1} and {self.input_name_2}")

        image_1 = res_images[self.input_name_1].copy()
        image_2 = res_images[self.input_name_2].copy()

        difference = image_2 - image_1

        if self.max_height is not None and self.min_height is not None:
            range_mask = (
                (image_1 >= self.min_height)
                & (image_2 >= self.min_height)
                & (image_1 <= self.max_height)
                & (image_2 <= self.max_height)
            )
            difference[~range_mask] = np.nan

        if self.threshold_forest is not None:
            non_forest_mask = (image_1 < self.threshold_forest) & (image_2 < self.threshold_forest)
            difference[non_forest_mask & np.isfinite(difference)] = 0

        # Mask the difference outside the range of valid differences
        mask_diff_range = (difference < self.min_difference) | (difference > self.max_difference)
        difference[mask_diff_range] = np.nan
        return difference


class min_max_height_mask_computer:
    """Builds a validity mask for pixels of an image within an optional height range.

    Args:
        name_image: Name of the image to threshold in `res_images`.
        min_height: Minimum valid value, or None to disable the lower bound.
        max_height: Maximum valid value, or None to disable the upper bound.
    """

    def __init__(
        self, name_image: str, min_height: Optional[float], max_height: Optional[float]
    ) -> None:
        self.name_image = name_image
        self.min_height = min_height
        self.max_height = max_height

    def compute_image(self, res_images: Dict[str, np.ndarray], row: pd.Series) -> np.ndarray:
        """Computes the float mask (1/0/NaN) of pixels within the configured height range.

        Args:
            res_images: Dictionary of already loaded/computed images, must
                contain `name_image`.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            Float array with 1 where the pixel is valid and within range, 0
            where valid but out of range, and NaN where the input was NaN.

        Raises:
            Exception: If `name_image` is not present in `res_images`.
        """
        if self.name_image not in res_images:
            raise Exception(f"You must first load {self.name_image}")

        image = res_images[self.name_image].copy()
        nan_mask = np.isnan(image)

        mask = ~np.isnan(image)
        if self.min_height is not None:
            mask = mask & (image >= self.min_height)
        if self.max_height is not None:
            mask = mask & (image <= self.max_height)

        mask[nan_mask] = np.nan
        return mask.astype(np.float32)


def apply_erosion_dilation(changes: np.ndarray) -> np.ndarray:
    """Denoises a binary change mask with an erosion-dilation-erosion sequence.

    Args:
        changes: Binary (or boolean) change mask.

    Returns:
        The denoised mask as a float32 array.
    """
    size = 3
    # Apply morphology to filter out noise
    changes = binary_erosion(changes, structure=np.ones((size, size)))
    changes = binary_dilation(changes, structure=np.ones((size, size)), iterations=2)
    changes = binary_erosion(changes, structure=np.ones((size, size)))
    return changes.astype(np.float32)


def apply_min_max_area(
    changes_dependent: np.ndarray,
    changes_primary: np.ndarray,
    resolution: float,
    min_area: Optional[float],
    max_area: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters connected components based on their area in the primary mask and propagates the removal to the dependent mask.

    This function identifies connected components in `changes_primary`. If a component's area (calculated using `resolution`)
    is strictly less than `min_area` or strictly greater than `max_area`, it is removed (set to NaN).
    Additionally, the overlapping region in `changes_dependent` is also removed (set to NaN).
    Finally, if a dependent component had an overlap with a primary component before filtering,
    but lost all its primary overlaps after filtering, the entire dependent component is removed.

    Args:
        changes_dependent: The dependent mask where overlapping regions will be removed.
        changes_primary: The primary mask used to compute component areas and decide on removal.
        resolution: The spatial resolution of the pixels (used to convert pixel count to area).
        min_area: The minimum allowed area, or None to disable the lower bound.
            Components smaller than this are removed.
        max_area: The maximum allowed area, or None to disable the upper bound.
            Components larger than this are removed.

    Returns:
        A tuple containing the filtered (changes_dependent, changes_primary) arrays.
    """

    changes_dependent = changes_dependent.copy()
    changes_primary = changes_primary.copy()

    nan_mask = np.isnan(changes_dependent) | np.isnan(changes_primary)
    changes_dependent[nan_mask] = 0
    changes_primary[nan_mask] = 0

    pixel_area = resolution**2
    min_area_pixels = np.ceil(min_area / pixel_area) if min_area is not None else None
    max_area_pixels = np.floor(max_area / pixel_area) if max_area is not None else None

    # Label the components
    labeled_primary, num_labels_primary = label(changes_primary, return_num=True)
    labeled_dependent, num_labels_dependent = label(changes_dependent, return_num=True)

    # Find dependent components that overlap with ANY primary component BEFORE filtering
    mask_any_primary_before = labeled_primary > 0
    dependent_with_ref_before = np.unique(labeled_dependent[mask_any_primary_before])
    dependent_with_ref_before = dependent_with_ref_before[dependent_with_ref_before > 0]

    if num_labels_primary > 0:
        for label_id in range(1, num_labels_primary + 1):
            mask_label_primary = labeled_primary == label_id

            component_area = np.sum(mask_label_primary)

            if min_area_pixels is not None and component_area < min_area_pixels:
                changes_primary[mask_label_primary] = np.nan
                changes_dependent[mask_label_primary] = np.nan
            elif max_area_pixels is not None and component_area > max_area_pixels:
                changes_primary[mask_label_primary] = np.nan
                changes_dependent[mask_label_primary] = np.nan

    # Find dependent components that overlap with ANY primary component AFTER filtering
    mask_any_primary_after = ~np.isnan(changes_primary) & (changes_primary != 0)
    dependent_with_ref_after = np.unique(labeled_dependent[mask_any_primary_after])
    dependent_with_ref_after = dependent_with_ref_after[dependent_with_ref_after > 0]

    # Remove the entire dependent component if it had a reference before, but has none left after
    dependent_to_remove = np.setdiff1d(dependent_with_ref_before, dependent_with_ref_after)
    if len(dependent_to_remove) > 0:
        mask_dependent_to_remove = np.isin(labeled_dependent, dependent_to_remove)
        changes_dependent[mask_dependent_to_remove] = np.nan

    changes_dependent[nan_mask] = np.nan
    changes_primary[nan_mask] = np.nan
    return changes_dependent, changes_primary


class detection_object_scale_mask_computer:
    """Matches predicted and reference connected components at the object scale.

    Labels connected components in the pred and ref masks, clusters components
    that overlap (via a bipartite graph of overlapping labels), and evaluates
    each cluster's IoU to decide whether it counts as a match (1) or not (-1).
    Components with no overlap on the other side are always marked as
    unmatched (-1). Optionally filters resulting components by area.

    Args:
        name_pred: Name of the predicted boolean mask image in `res_images`.
        name_ref: Name of the reference boolean mask image in `res_images`.
        min_area: Minimum area (post-matching) for a component to be kept, or
            None to disable.
        max_area: Maximum area (post-matching) for a component to be kept, or
            None to disable.
        resolution: Spatial resolution of a pixel, used to convert pixel counts
            to area.
        iou_threshold: Minimum IoU for a matched cluster to be considered a
            correct detection.
        apply_erosion_dilation: If True, denoises the masks with
            `apply_erosion_dilation` before labeling connected components.
    """

    def __init__(
        self,
        name_pred: str,
        name_ref: str,
        min_area: Optional[float],
        max_area: Optional[float],
        resolution: float,
        iou_threshold: float,
        apply_erosion_dilation: bool,
    ) -> None:
        self.name_pred = name_pred
        self.name_ref = name_ref
        self.min_area = min_area
        self.max_area = max_area
        self.resolution = resolution
        self.iou_threshold = iou_threshold
        self.apply_erosion_dilation = apply_erosion_dilation

    def compute_image(
        self, res_images: Dict[str, np.ndarray], row: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Computes the pred and ref object-scale evaluation masks.

        Args:
            res_images: Dictionary of already loaded/computed images, must
                contain `name_pred` and `name_ref`.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            Dictionary with `"pred"` and `"ref"` evaluation masks (values in
            `{-1, 0, 1}`; NaN is preserved where the inputs were NaN).

        Raises:
            Exception: If `name_pred` or `name_ref` is not present in `res_images`.
        """
        if (self.name_pred not in res_images) or (self.name_ref not in res_images):
            raise Exception(f"You must first load {self.name_pred} and {self.name_ref}")

        mask_pred = res_images[self.name_pred].copy()
        mask_ref = res_images[self.name_ref].copy()

        nan_mask = np.isnan(mask_pred) | np.isnan(mask_ref)
        mask_pred[nan_mask] = 0
        mask_ref[nan_mask] = 0

        mask_pred = mask_pred.astype(np.float32)
        mask_ref = mask_ref.astype(np.float32)

        # Denoise the binary masks before labeling so isolated pixel-level noise
        # doesn't fragment a true object into spurious objects
        if self.apply_erosion_dilation:
            mask_pred = apply_erosion_dilation(mask_pred)
            mask_ref = apply_erosion_dilation(mask_ref)

        # 2. Label connected components
        labeled_pred, num_pred = label(mask_pred, return_num=True)
        labeled_ref, num_ref = label(mask_ref, return_num=True)

        # Find overlapping pixels
        overlap_mask = (mask_pred == 1) & (mask_ref == 1)
        overlapping_pred_labels = labeled_pred[overlap_mask]
        overlapping_ref_labels = labeled_ref[overlap_mask]

        # Filter out background
        valid_overlaps = (overlapping_pred_labels > 0) & (overlapping_ref_labels > 0)
        p_labels = overlapping_pred_labels[valid_overlaps]
        r_labels = overlapping_ref_labels[valid_overlaps]

        # 3. Create Bipartite Graph for clustering
        if len(p_labels) > 0:
            unique_pairs = np.unique(np.vstack((p_labels, r_labels)), axis=1)
        else:
            unique_pairs = np.array([[], []])

        if unique_pairs.size > 0:
            p_unique = unique_pairs[0]
            r_unique = unique_pairs[1]

            total_nodes = num_pred + num_ref
            row_indices = p_unique - 1
            col_indices = (r_unique - 1) + num_pred

            data = np.ones(len(row_indices), dtype=int)
            rows = np.concatenate([row_indices, col_indices])
            cols = np.concatenate([col_indices, row_indices])
            vals = np.concatenate([data, data])

            graph = csr_matrix((vals, (rows, cols)), shape=(total_nodes, total_nodes))
            n_components, labels_graph = connected_components(
                csgraph=graph, directed=False, return_labels=True
            )

            clusters = []
            for comp_id in range(n_components):
                nodes_in_comp = np.where(labels_graph == comp_id)[0]
                pred_nodes = [n + 1 for n in nodes_in_comp if n < num_pred]
                ref_nodes = [n - num_pred + 1 for n in nodes_in_comp if n >= num_pred]
                clusters.append({"pred_elements": pred_nodes, "ref_elements": ref_nodes})
        else:
            # If there are absolutely no overlaps, every component is isolated
            clusters = []
            for p_idx in range(1, num_pred + 1):
                clusters.append({"pred_elements": [p_idx], "ref_elements": []})
            for r_idx in range(1, num_ref + 1):
                clusters.append({"pred_elements": [], "ref_elements": [r_idx]})

        # 4. Create clustered masks
        pred_lookup = np.zeros(num_pred + 1, dtype=np.int32)
        ref_lookup = np.zeros(num_ref + 1, dtype=np.int32)

        current_cluster_id = 1
        for cluster in clusters:
            if cluster["pred_elements"] and cluster["ref_elements"]:
                pred_lookup[cluster["pred_elements"]] = current_cluster_id
                ref_lookup[cluster["ref_elements"]] = current_cluster_id
                current_cluster_id += 1

        clustered_mask_pred = pred_lookup[labeled_pred]
        clustered_mask_ref = ref_lookup[labeled_ref]

        # 5. Create Evaluation Masks (0=Background, 1=Good, -1=Bad)
        eval_mask_pred = np.zeros_like(mask_pred)
        eval_mask_ref = np.zeros_like(mask_ref)

        # Mark isolated components as bad (-1)
        for cluster in clusters:
            if cluster["pred_elements"] and not cluster["ref_elements"]:
                for p_idx in cluster["pred_elements"]:
                    eval_mask_pred[labeled_pred == p_idx] = -1
            elif cluster["ref_elements"] and not cluster["pred_elements"]:
                for r_idx in cluster["ref_elements"]:
                    eval_mask_ref[labeled_ref == r_idx] = -1

        # Evaluate IoU for overlapping clusters
        for cid in range(1, current_cluster_id):
            c_pred = clustered_mask_pred == cid
            c_ref = clustered_mask_ref == cid

            intersection = np.sum(c_pred & c_ref)
            union = np.sum(c_pred | c_ref)

            if union > 0:
                iou = intersection / union
                if iou >= self.iou_threshold:
                    eval_mask_pred[c_pred] = 1
                    eval_mask_ref[c_ref] = 1
                else:
                    eval_mask_pred[c_pred] = -1
                    eval_mask_ref[c_ref] = -1

        eval_mask_pred, eval_mask_ref = apply_min_max_area(
            eval_mask_pred, eval_mask_ref, self.resolution, self.min_area, self.max_area
        )

        eval_mask_pred[np.isnan(eval_mask_pred)] = 0
        eval_mask_ref[np.isnan(eval_mask_ref)] = 0

        eval_mask_pred[nan_mask] = np.nan
        eval_mask_ref[nan_mask] = np.nan

        dict_changes = {"pred": eval_mask_pred, "ref": eval_mask_ref}

        return dict_changes


class binned_by_object_scale_mask_computer:
    """Computes object-scale evaluation masks binned by connected-component area.

    Runs `detection_object_scale_mask_computer` once globally, then re-filters
    the resulting pred/ref masks by area bin using `apply_min_max_area`.

    Args:
        name_pred: Name of the predicted boolean mask image in `res_images`.
        name_ref: Name of the reference boolean mask image in `res_images`.
        bins_range_area: Sorted list of area bin edges.
        resolution: Spatial resolution of a pixel, used to convert pixel counts
            to area.
        iou_threshold: Minimum IoU for a matched cluster to be considered a
            correct detection (forwarded to the underlying detector).
        reduction_basis: Either "pred" or "ref", selecting which mask's area
            drives the bin filtering; the other mask is filtered based on the
            resulting overlap.
        include_outer_bins: If True, adds an unbounded bin below the first edge
            and above the last edge.
        apply_erosion_dilation: If True, denoises the masks before labeling
            connected components (forwarded to the underlying detector).

    Raises:
        AssertionError: If `reduction_basis` is not "pred" or "ref".
    """

    def __init__(
        self,
        name_pred: str,
        name_ref: str,
        bins_range_area: list,
        resolution: float,
        iou_threshold: float,
        reduction_basis: str,
        include_outer_bins: bool,
        apply_erosion_dilation: bool,
    ) -> None:
        self.name_pred = name_pred
        self.name_ref = name_ref
        self.bins_range_area = (
            [None] + bins_range_area + [None] if include_outer_bins else bins_range_area
        )
        self.resolution = resolution
        self.detection_object_scale_mask_computer = detection_object_scale_mask_computer(
            name_pred, name_ref, None, None, resolution, iou_threshold, apply_erosion_dilation
        )
        self.reduction_basis = reduction_basis
        self.include_outer_bins = include_outer_bins
        assert self.reduction_basis in ["pred", "ref"], "reduction_basis must be 'pred' or 'ref'"

    def compute_image(
        self, res_images: Dict[str, np.ndarray], row: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Computes per-bin pred/ref evaluation masks from the global object-scale matching.

        Args:
            res_images: Dictionary of already loaded/computed images, must
                contain `name_pred` and `name_ref`.
            row: GeoDataFrame row corresponding to the sample (unused).

        Returns:
            Dictionary with `"{bin}_pred"` and `"{bin}_ref"` evaluation masks
            for each area bin.

        Raises:
            Exception: If `name_pred` or `name_ref` is not present in `res_images`.
        """
        if (self.name_pred not in res_images) or (self.name_ref not in res_images):
            raise Exception(f"You must first load {self.name_pred} and {self.name_ref}")

        changes_global = self.detection_object_scale_mask_computer.compute_image(res_images, row)

        pred_changes = changes_global["pred"]
        ref_changes = changes_global["ref"]

        dict_changes = {}
        for bin_idx in range(len(self.bins_range_area) - 1):
            min_area_current = self.bins_range_area[bin_idx]
            max_area_current = self.bins_range_area[bin_idx + 1]

            max_area_filter = (
                max_area_current - 1e-6 if max_area_current is not None else None
            )  # we subtract epsilon to have the bins [min,max[ with apply_min_max_area

            if self.reduction_basis == "pred":
                ref_change_binned, pred_change_binned = apply_min_max_area(
                    ref_changes, pred_changes, self.resolution, min_area_current, max_area_filter
                )
            elif self.reduction_basis == "ref":
                pred_change_binned, ref_change_binned = apply_min_max_area(
                    pred_changes, ref_changes, self.resolution, min_area_current, max_area_filter
                )

            pred_change_binned[np.isnan(pred_change_binned)] = 0
            ref_change_binned[np.isnan(ref_change_binned)] = 0

            max_area_current_str = "" if max_area_current is None else str(int(max_area_current))
            min_area_current_str = "" if min_area_current is None else str(int(min_area_current))
            name_bin = f"{min_area_current_str}-{max_area_current_str}"

            dict_changes[f"{name_bin}_pred"] = pred_change_binned
            dict_changes[f"{name_bin}_ref"] = ref_change_binned
        return dict_changes
