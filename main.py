# -*- coding: utf-8 -*-

"""
Entry point for the application.
"""

# **** IMPORTS ****
import time
import sqlite3
import logging
import datetime
from tqdm import tqdm
from typing import Type
from pathlib import Path

# **** LOCAL IMPORTS ****
from cse5525_final_project.metric import Metric, MetricComputationTimes
from cse5525_final_project.metrics import generate_iqa_metric_classes
from cse5525_final_project.util import (
    load_image_from_uuid,
    load_json_documents,
    combine_dictionaries,
    discover_classes_in_file,
)
from cse5525_final_project.database import get_db_connection
from cse5525_final_project.config import (
    DB_PATH,
    LOGGER_CONFIG,
    IMAGES_DIR,
    METRICS_FILE_PATH,
    IMAGE_METRIC_COMPUTATION_TIME_LOGGING_THRESHOLD,
)

# **** LOGGING ****
logger = logging.getLogger(__name__)

# **** MAIN FUNCTION ****
def main():
    app_start_time = time.perf_counter()
    logger.info("Starting the application...")
    
    # ****
    # Initialize database
    db_path = Path(DB_PATH)
    logger.info(f"Connecting to database at {db_path}")

    # Create database if not present
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch()
        logger.info(f"Database created at {db_path}")
        
    # Connect to the database
    conn: sqlite3.Connection = get_db_connection(db_path)
    
    try:
        # ****
        # Find metric classes
        logger.info("Finding metrics...")
        metric_classes = discover_classes_in_file(METRICS_FILE_PATH, Metric)
        if not metric_classes:
            logger.warning(f"No metrics found in the file: {METRICS_FILE_PATH}")
            return

        # ****
        # Generate pyiqa metrics
        logger.info("Generating pyiqa metrics...")
        iqa_metric_classes = generate_iqa_metric_classes(
            blacklist=[
                "ahiq",  # Full Reference metric
                "ckdn",  # Full Reference metric
                "clipscore",  # Needs a caption list
                "cw_ssim",  # Full Reference metric
                "dists",  # Full Reference metric
                "fid",  # TODO: Figure out error
                "fsim",  # Full Reference metric
                "gmsd",  # Full Reference metric
                "inception_score",  # TODO: Figure out error
                "lpips",  # Full Reference metric
                "lpips+",  # Full Reference metric
                "lpips-vgg",  # Full Reference metric
                "lpips-vgg+",  # Full Reference metric
                "mad",  # Full Reference metric
                "ms_ssim",  # Full Reference metric
                "msswd",  # Full Reference metric
                "nlpd",  # Full Reference metric
                "pieapp",  # Full Reference metric
                "psnr",  # Full Reference metric
                "psnry",  # Full Reference metric
                "qalign_4bit",  # TODO: Figure out error
                "ssim",  # Full Reference metric
                "ssimc",  # Full Reference metric
                "stlpips",  # Full Reference metric
                "stlpips-vgg",  # Full Reference metric
                "topiq_fr",  # Full Reference metric
                "topiq_fr-pipal",  # Full Reference metric
                "vif",  # Full Reference metric
                "vsi",  # Full Reference metric
                "wadiqam_fr",  # Full Reference metric
            ]
        ).values()
        if not iqa_metric_classes:
            logger.warning("No pyiqa metrics found")
            return
        metric_classes.extend(iqa_metric_classes)
        
        # ****
        # Load images
        logger.info(f"Searching for json files in {IMAGES_DIR}")
        json_files = load_json_documents(IMAGES_DIR)
        if not json_files:
            logger.warning(f"No json files found in {IMAGES_DIR}")
            return
        
        # Combine dictionaries from all json files
        combined_dict = combine_dictionaries(json_files)
        
        # Generate file paths for each image
        image_paths = {}
        combined_dict = list(combined_dict.keys())[:2]  # TODO: Remove this line to process all images
        for image_uuid in combined_dict:
            image_uuid: str = image_uuid.strip()
            # Keys are image names, so we need to remove the extension to get the UUID
            image_uuid = image_uuid.split(".")[0]  # Remove the file extension
            
            # Attempt to load the image using its UUID
            image_path = load_image_from_uuid(IMAGES_DIR, image_uuid)
            if image_path:
                image_paths[image_uuid] = image_path
            else:
                logger.warning(f"Image not found for UUID: {image_uuid}")
        
        # ****
        # Compute and store metrics for each image
        MetricComputationTimes.create_table(conn)  # Ensure table is created before loops

        for image_uuid in tqdm(image_paths.keys(), desc="Processing Images"):
            image_path: Path = image_paths[image_uuid]

            # Check if image path exists
            if not image_path.exists():
                logger.warning(f"Image path does not exist: {image_path}")
                continue

            logger.debug(f"Processing image: {image_uuid} at {image_path}")

            # Attempt each metric for this image
            for metric_class in metric_classes:
                metric_class: Type[Metric]

                # Use datetime to capture start and end times in ISO format
                metric_start = datetime.datetime.now()

                try:
                    metric_class.compute_and_store(conn, image_path)
                except Exception as e:
                    logger.error(f"Failed to compute/store {metric_class.__name__} for {image_uuid}: {e}")

                metric_end = datetime.datetime.now()

                # Calculate elapsed time in seconds
                elapsed = (metric_end - metric_start).total_seconds()

                # Insert record with ISO 8601 times
                MetricComputationTimes.insert_record(
                    conn,
                    {
                        "uuid": image_uuid,
                        "metric_name": metric_class.__name__,
                        "start_time": metric_start.isoformat(),
                        "end_time": metric_end.isoformat()
                    }
                )

                if elapsed > IMAGE_METRIC_COMPUTATION_TIME_LOGGING_THRESHOLD:
                    logger.info(
                        f"Metric {metric_class.__name__} for {image_uuid} took {elapsed:.3f} seconds"
                    )


    except Exception as e:
        logger.error(f"Caught exception during execution", exc_info=e)

    # ****
    # Clean up
    conn.close()
    app_end_time = time.perf_counter()
    logger.info(f"Application completed in {app_end_time - app_start_time:.2f} seconds")
    
    


# ****
if __name__ == "__main__":
    # ****
    # Suppress warnings
    import warnings

    # Suppress all UserWarnings and FutureWarnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # (Optional) suppress specific messages
    warnings.filterwarnings("ignore", message=".*resume_download.*")
    warnings.filterwarnings("ignore", message=".*do_sample.*")
    warnings.filterwarnings("ignore", message=".*top_p.*")
    warnings.filterwarnings("ignore", message=".*meshgrid.*")
    
    # ****
    # Set up logging
    import logging.config
    logging.disable(logging.DEBUG)

    # Suppress specific loggers
    logging.getLogger("pyiqa").setLevel(logging.WARNING)
    logging.getLogger("timm").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("load_pretrained").setLevel(logging.WARNING)
    logging.getLogger("load_state_dict_from_hf").setLevel(logging.WARNING)

    logging.config.dictConfig(LOGGER_CONFIG)
    
    # ****
    main()
