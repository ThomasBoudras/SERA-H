import hydra
import shutil
from pathlib import Path
import geopandas as gpd 
import numpy as np
import rasterio
import torch
from omegaconf import DictConfig, OmegaConf
from osgeo import gdal
from shapely.geometry import box
import glob
from rasterio.transform import from_bounds
from lightning import seed_everything
from tqdm import tqdm
from src import global_utils as utils
from src.global_utils import get_window
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box
from joblib import Parallel, delayed
from src.datamodule.datamodule_utils import expand_gdf
import time

log = utils.get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def predict(config: DictConfig) -> None:
   
    # Print the config
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    OmegaConf.set_struct(config, True)

    # Set the seed
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning loggers
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    # Prepare initialisation of datamodule and module 
    # Define paths for all processes
    save_dir = Path(config.save_dir).resolve()
    save_dir_tmp = save_dir / "tmp"
    save_dir_data = save_dir / "data"
    aoi_gdf_path = save_dir / "predict_gdf.geojson"
    grouped_aoi_gdf_path = save_dir / "grouped_aoi_gdf.geojson"
    
    # Create directories and files, just for the master process (to avoid race conditions)
    if trainer.is_global_zero:
        if save_dir.exists() :
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True)
        save_dir_data.mkdir()
        save_dir_tmp.mkdir()

        # Create the subdirectories
        (save_dir_tmp / "patch_tifs").mkdir()

        # Get the aoi from the config
        if config.via_bounds is not None and config.via_geojson is None:
            row = {name : data for name, data in config.via_bounds.items() if name!="bounds"}
            bounds = config.via_bounds["bounds"]
            bounds = bounds["left"], bounds["bottom"], bounds["right"], bounds["top"]
            row["geometry"] = box(*bounds)
            aoi_gdf = gpd.GeoDataFrame([row], crs="EPSG:2154")

        elif config.via_geojson is not None and config.via_bounds is None :
            aoi_gdf = gpd.read_file(config.via_geojson["path"])
            if config.via_geojson.get("split", None) is not None:
                aoi_gdf = aoi_gdf[aoi_gdf['split'] == config.via_geojson["split"]].reset_index(drop=True) 
        else:
            Exception("via_bounds and via_geojson : one of them must be provided and the other must be null")

            
        if config.get("adapt_gdf", None) is not None: 
            adapt_gdf = hydra.utils.instantiate(config.adapt_gdf)
            aoi_gdf = adapt_gdf(aoi_gdf)
        
        # Save the aoi to a geojson file for datamodule
        aoi_gdf.to_file(
                aoi_gdf_path,
                driver="GeoJSON",
            )

        # Group geometries that are adjacent or touching each other
        unioned = unary_union(aoi_gdf['geometry'])
        if isinstance(unioned, MultiPolygon):
            polygons = list(unioned.geoms)
        elif isinstance(unioned, Polygon):
            polygons = [unioned]
        else:
            polygons = []
        grouped_aoi_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=aoi_gdf.crs)        

        # Subdivide geometries into patches if patch_size_max is provided
        if config.get("patch_size_max", None) is not None:
            patch_size_max = config.patch_size_max
            resolution = config.datamodule.dataset.resolution_input
            
            log.info(f"Subdividing geometries according to patch_size_max={patch_size_max}")
            grouped_aoi_gdf = expand_gdf(grouped_aoi_gdf, patch_size=patch_size_max, margin_size=0, resolution=resolution)
            # Ne garder que les géométries de grouped_aoi_gdf qui intersectent avec aoi_gdf
            grouped_aoi_gdf = grouped_aoi_gdf[grouped_aoi_gdf.geometry.intersection(aoi_gdf.unary_union).area > 0].reset_index(drop=True)

        else:
            log.info("patch_size_max is null, keeping geometries as is after unary_union")


        grouped_aoi_gdf.to_file(
            grouped_aoi_gdf_path,
            driver="GeoJSON",
        )
    # Wait for the master process to be ready, it's not pretty but trainer.startegy.barrier does not work here
    if trainer.is_global_zero:
        (save_dir_tmp / "rank_0_ready").touch()

    while not (save_dir_tmp / "rank_0_ready").exists() :
        time.sleep(1)

    if trainer.is_global_zero:
        time.sleep(3)
        (save_dir_tmp / "rank_0_ready").unlink()
        

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule.instance._target_}>")
    config.datamodule.dataset.gdf_path = aoi_gdf_path # we take the new gdf_path
    # TO avoid useless initialization of the datasets
    config.datamodule.dataset.train_dataset = None
    config.datamodule.dataset.val_dataset = None
    config.datamodule.dataset.test_dataset = None
    datamodule = hydra.utils.instantiate(config.datamodule.instance)

    # Init lightning module
    log.info(f"Instantiating module <{config.module.instance._target_}>")
    module = hydra.utils.instantiate(config.module.instance)
    module.predictions_save_dir = save_dir_tmp

    # Load the model
    if config.get("ckpt_path") is not None or config.get("ckpt_path") != "last":
        ckpt_path = config.get("ckpt_path")
        if config.load_just_weights :
            log.info(f"Start of training from checkpoint {ckpt_path} using only the weights !")
            checkpoint = torch.load(ckpt_path)

            if "state_dict" in checkpoint:
                missing_keys, unexpected_keys = module.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                missing_keys, unexpected_keys = module.load_state_dict(checkpoint, strict=False)
            
                log.warning(f"Missing keys in checkpoint: {missing_keys}")
                log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            
            ckpt_path = None
        
        else :
            log.info(f"Start of training from checkpoint {ckpt_path} !")
    
    elif ckpt_path == "last" :
        log.info(f"Starting training from last checkpoint {ckpt_path} !")
    
    else :
        log.info("Starting training from scratch!")
        ckpt_path = None

    # Predict the model
    trainer.predict(module, ckpt_path=ckpt_path, datamodule=datamodule)

    if trainer.is_global_zero:
        rank_dirs = [file for file in save_dir_tmp.iterdir() if file.is_dir() and file.name.startswith("rank")]
        for rank_dir in rank_dirs:

            # Get the number of predicted files
            nb_predicted_files = len([
                file
                for file in (rank_dir / "preds").iterdir()
                if file.suffix == ".npy"
            ])

            # Save numpy predictions as tifs
            Parallel(n_jobs=-1)(
                delayed(save_images_as_tifs)(i_file, rank_dir, save_dir_tmp) 
                for i_file in tqdm(range(nb_predicted_files), desc="Saving preds for each batch")
            )

        # Get the list of all patch GeoTIFF files in the temporary directory
        patches_path = save_dir_tmp / f"patch_tifs" 
        list_subtif = [file for file in patches_path.iterdir() if file.suffix == ".tif"]  
        Parallel(n_jobs=-1)(
            delayed(merge_tifs)(tif_poly, list_subtif, save_dir_data, config.run_name) 
            for tif_poly in tqdm(grouped_aoi_gdf["geometry"], total=len(grouped_aoi_gdf), desc="Merging tifs")
        )

        # Create vrt
        create_vrts(save_dir_data, config.run_name)

        # Remove the temporary directory
        shutil.rmtree(save_dir_tmp)


def save_images_as_tifs(i_file, rank_dir, save_dir_tmp):
    """
    Save individual prediction patches as GeoTIFF files.

    Args:
        i_file (int): Index of the batch file to process. This corresponds to the batch number used in the prediction saving step.
        save_dir_tmp (Path): Path to the temporary directory where the prediction and bounds .npy files are stored. The function will also save the resulting GeoTIFF files in a subdirectory of this path.
    """
    pred_file = rank_dir / "preds" / f"batch_{i_file}.npy"
    bounds_file = rank_dir / "bounds" / f"batch_{i_file}.npy"
    
    # Load the prediction and bounds for the current batch
    batch_preds = np.load(pred_file)
    batch_bounds = np.load(bounds_file)

    # Save each prediction patch as a GeoTIFF file
    for idx, preds in enumerate(batch_preds):
        bounds = batch_bounds[idx]

        # Prepare the georeferencing information for the current patch
        count, height, width = preds.shape
        transform = from_bounds(*bounds, width, height)

        # Create the path to save the GeoTIFF file
        save_image_path = save_dir_tmp / f"patch_tifs" / f"{int(bounds[0])}_{int(bounds[1])}_{int(bounds[2])}_{int(bounds[3])}.tif"
        
        # Save the GeoTIFF file
        with rasterio.open(
            save_image_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=preds.dtype,
            crs="EPSG:2154",
            transform=transform,
        ) as dst:
            dst.write(preds)
    pred_file.unlink()
    bounds_file.unlink()


def merge_tifs(tif_poly, list_subtif, save_dir_data, model_name):
    """
    Merge adjacent GeoTIFF files into a single image.

    Args:
        tif_poly (Polygon): Polygon for the current patch.
        list_subtif (list): List of GeoTIFF files to merge.
        save_dir_data (Path): Path to the directory where the merged image will be saved.
        model_name (str): Name of the model used for the predictions.
    """
    tif_bounds = list(tif_poly.bounds)
        
    # Filter the tifs that intersect the current patch's bounding box
    list_subtif_intersect = []
    for subtif in list_subtif:
        subtif_bounds = subtif.stem.split("_")
        if (
            tif_bounds[0] <= int(subtif_bounds[2]) and
            tif_bounds[1] <= int(subtif_bounds[3]) and
            int(subtif_bounds[0]) <= tif_bounds[2] and
            int(subtif_bounds[1]) <= tif_bounds[3]
        ) :
            list_subtif_intersect.append(subtif)

    # To avoid out-of-bounds issues, we extend the tif bounds to the global bounds of all subtifs.
    for subtif in list_subtif_intersect:
        bounds = subtif.stem.split("_")
        bounds = [int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])]
        tif_bounds[0] = min(tif_bounds[0], bounds[0])
        tif_bounds[1] = min(tif_bounds[1], bounds[1])
        tif_bounds[2] = max(tif_bounds[2], bounds[2])
        tif_bounds[3] = max(tif_bounds[3], bounds[3])

    # Get resolution from the first subtif
    with rasterio.open(list_subtif_intersect[0]) as src:
        resolution = src.transform.a  
    
    # Create a window from the tif_bounds with the correct resolution
    window_width = round((tif_bounds[2] - tif_bounds[0]) / resolution)
    window_height = round((tif_bounds[3] - tif_bounds[1]) / resolution)
    window_transform = from_bounds(*tif_bounds, window_width, window_height)
    
    mean_image = np.zeros((1,window_height, window_width), dtype=np.float32)
    nb_values_image = np.zeros((1,window_height, window_width), dtype=np.float32)
    for subtif_file in list_subtif_intersect:
        bounds = subtif_file.stem.split("_")
        bounds = [int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])]

        image, _ = get_window(
            subtif_file,
            bounds,
            resolution=None,
            resampling_method=None,
        )

        # Convert world coordinates to pixel coordinates using the window transform
        subtif_col_start, subtif_row_start = ~window_transform * (bounds[0], bounds[3])
        subtif_col_end, subtif_row_end = ~window_transform * (bounds[2], bounds[1])

        # Convert to integers and ensure they are within bounds
        subtif_col_start = int(round(subtif_col_start))
        subtif_row_start = int(round(subtif_row_start))
        subtif_col_end = int(round(subtif_col_end))
        subtif_row_end = int(round(subtif_row_end))
        
        # Update the mean_image and nb_values_image for the specific region
        mask_nan = np.isnan(image)
        mean_image[:,subtif_row_start:subtif_row_end, subtif_col_start:subtif_col_end] = np.nansum([
            mean_image[:,subtif_row_start:subtif_row_end, subtif_col_start:subtif_col_end],
            image.astype(np.float32)
        ], axis=0)
        nb_values_image[:,subtif_row_start:subtif_row_end, subtif_col_start:subtif_col_end][~mask_nan] += 1
        
    valid_mask = nb_values_image != 0
    mean_image[valid_mask] = mean_image[valid_mask] / nb_values_image[valid_mask]
    mean_image[~valid_mask] = np.nan

    # Prepare the georeferencing information for the merged image
    count, height, width = mean_image.shape
    transform = from_bounds(*tif_bounds, width, height)

    # Create the path to save the merged image
    output_path = save_dir_data / f"{model_name}_{int(tif_bounds[0])}_{int(tif_bounds[1])}.tif"

    # Write the merged image to the output path using rasterio
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        count=count,
        height=height,
        width=width,
        dtype=mean_image.dtype,
        crs="EPSG:2154",
        transform=transform,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(mean_image)

def create_vrts(save_dir_data, model_name):
    vrt_path = save_dir_data / f"{model_name}_full.vrt"
    files_list = [
         str(file) for file in (save_dir_data).iterdir() if file.suffix == ".tif"
    ]

    gdal.UseExceptions()
    vrt_options = gdal.BuildVRTOptions(
        separate=False,
        srcNodata="nan",
        VRTNodata="nan",
    )  
   
    vrt = gdal.BuildVRT(vrt_path, files_list, options=vrt_options)
    vrt.FlushCache()

if __name__ == "__main__":
    predict()




        
        

