#!/usr/bin/env python3
"""
Data preprocessing utility for GeoAI Research project
Handles various data preprocessing tasks for satellite imagery and building footprint data
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
from rasterio.mask import mask
from shapely.geometry import shape, Polygon, MultiPolygon
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import shutil
import glob
from datetime import datetime
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_preprocessing')


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        description="Data preprocessing utility for GeoAI Research project"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input directory with raw data"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for processed data"
    )
    parser.add_argument(
        "--mode", "-m", choices=["all", "imagery", "footprints", "validation", "augmentation"],
        default="all", help="Processing mode"
    )
    parser.add_argument(
        "--bands", "-b", default="RGB", help="Bands to include in processed imagery (e.g., 'RGB', 'RGBNIR')"
    )
    parser.add_argument(
        "--resolution", "-r", type=float, default=0.5, help="Output resolution in meters per pixel"
    )
    parser.add_argument(
        "--tile-size", "-t", type=int, default=512, help="Tile size in pixels"
    )
    parser.add_argument(
        "--overlap", "-v", type=int, default=64, help="Overlap between tiles in pixels"
    )
    parser.add_argument(
        "--augment", "-a", action="store_true", help="Apply data augmentation"
    )
    parser.add_argument(
        "--clean", "-c", action="store_true", help="Clean output directory before processing"
    )
    parser.add_argument(
        "--format", "-f", choices=["geojson", "shapefile", "csv"], default="geojson",
        help="Output format for vector data"
    )
    parser.add_argument(
        "--crs", default="EPSG:4326", help="Target coordinate reference system"
    )
    parser.add_argument(
        "--validate-fraction", type=float, default=0.1, 
        help="Fraction of data to use for validation"
    )
    return parser


def clean_output_directory(output_dir: str) -> None:
    """Clean output directory"""
    if os.path.exists(output_dir):
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)


def create_directory_structure(base_dir: str) -> Dict[str, str]:
    """Create directory structure for processed data"""
    dirs = {
        'imagery': os.path.join(base_dir, 'imagery'),
        'footprints': os.path.join(base_dir, 'footprints'),
        'tiles': os.path.join(base_dir, 'tiles'),
        'validation': os.path.join(base_dir, 'validation'),
        'augmented': os.path.join(base_dir, 'augmented'),
        'metadata': os.path.join(base_dir, 'metadata'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def find_image_files(input_dir: str) -> List[str]:
    """Find satellite imagery files in the input directory"""
    extensions = ['.tif', '.tiff', '.jp2', '.png', '.jpg', '.jpeg']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', f'*{ext}'), recursive=True))
    
    logger.info(f"Found {len(image_files)} image files")
    return image_files


def find_vector_files(input_dir: str) -> List[str]:
    """Find vector data files in the input directory"""
    extensions = ['.geojson', '.shp', '.json', '.kml', '.kmz']
    
    vector_files = []
    for ext in extensions:
        vector_files.extend(glob.glob(os.path.join(input_dir, '**', f'*{ext}'), recursive=True))
    
    logger.info(f"Found {len(vector_files)} vector files")
    return vector_files


def process_imagery(
    image_files: List[str], 
    output_dirs: Dict[str, str],
    bands: str = 'RGB',
    resolution: float = 0.5,
    target_crs: str = 'EPSG:4326'
) -> List[Dict[str, Any]]:
    """Process satellite imagery"""
    logger.info(f"Processing {len(image_files)} satellite images")
    
    processed_images = []
    
    for image_file in image_files:
        try:
            logger.info(f"Processing image: {os.path.basename(image_file)}")
            
            # Open the image
            with rasterio.open(image_file) as src:
                # Get metadata
                metadata = src.meta.copy()
                
                # Determine bands to extract
                if bands == 'RGB':
                    if src.count >= 3:
                        band_indices = [1, 2, 3]  # Typically RGB bands
                    else:
                        band_indices = list(range(1, min(src.count + 1, 4)))
                        logger.warning(f"Image has fewer than 3 bands, using available bands: {band_indices}")
                elif bands == 'RGBNIR':
                    if src.count >= 4:
                        band_indices = [1, 2, 3, 4]  # Typically RGB+NIR bands
                    else:
                        band_indices = list(range(1, src.count + 1))
                        logger.warning(f"Image has fewer than 4 bands, using available bands: {band_indices}")
                else:
                    # Custom band selection
                    band_indices = [int(b) for b in bands.split(',') if b.isdigit() and 1 <= int(b) <= src.count]
                    if not band_indices:
                        band_indices = list(range(1, min(src.count + 1, 4)))
                        logger.warning(f"Invalid band selection, using default: {band_indices}")
                
                # Calculate output transform and dimensions
                dst_crs = target_crs
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds,
                    resolution=(resolution, resolution)
                )
                
                # Update metadata for output
                metadata.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height,
                    'count': len(band_indices),
                    'dtype': src.dtypes[0]
                })
                
                # Create output filename
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                output_file = os.path.join(output_dirs['imagery'], f"{base_name}_processed.tif")
                
                # Create output raster
                with rasterio.open(output_file, 'w', **metadata) as dst:
                    # Read and reproject each band
                    for idx, band_idx in enumerate(band_indices, 1):
                        reproject(
                            source=rasterio.band(src, band_idx),
                            destination=rasterio.band(dst, idx),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear
                        )
                
                # Create metadata entry
                image_metadata = {
                    'filename': output_file,
                    'original_file': image_file,
                    'crs': dst_crs,
                    'resolution': resolution,
                    'width': width,
                    'height': height,
                    'bands': band_indices,
                    'bounds': [float(b) for b in src.bounds],
                    'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                processed_images.append(image_metadata)
                
        except Exception as e:
            logger.error(f"Error processing image {image_file}: {e}")
    
    # Save metadata
    metadata_file = os.path.join(output_dirs['metadata'], 'imagery_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(processed_images, f, indent=2)
    
    logger.info(f"Processed {len(processed_images)} images")
    return processed_images


def process_footprints(
    vector_files: List[str], 
    output_dirs: Dict[str, str],
    output_format: str = 'geojson',
    target_crs: str = 'EPSG:4326'
) -> List[Dict[str, Any]]:
    """Process building footprints"""
    logger.info(f"Processing {len(vector_files)} vector files")
    
    processed_footprints = []
    
    for vector_file in vector_files:
        try:
            logger.info(f"Processing vector file: {os.path.basename(vector_file)}")
            
            # Read vector file
            try:
                gdf = gpd.read_file(vector_file)
            except Exception as e:
                logger.error(f"Error reading vector file {vector_file}: {e}")
                continue
            
            # Check if it contains polygons
            if 'geometry' not in gdf.columns:
                logger.warning(f"No geometry column found in {vector_file}")
                continue
                
            # Filter to only keep polygons
            gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
            
            if len(gdf) == 0:
                logger.warning(f"No polygon features found in {vector_file}")
                continue
            
            # Reproject to target CRS
            if gdf.crs is None:
                logger.warning(f"No CRS found in {vector_file}, assuming {target_crs}")
                gdf.set_crs(target_crs, inplace=True)
            else:
                gdf = gdf.to_crs(target_crs)
            
            # Clean invalid geometries
            gdf = gdf[gdf.geometry.is_valid]
            if len(gdf) == 0:
                logger.warning(f"No valid geometries found in {vector_file} after cleaning")
                continue
                
            # Create output filename
            base_name = os.path.splitext(os.path.basename(vector_file))[0]
            
            if output_format == 'geojson':
                output_file = os.path.join(output_dirs['footprints'], f"{base_name}_processed.geojson")
                gdf.to_file(output_file, driver='GeoJSON')
            elif output_format == 'shapefile':
                output_file = os.path.join(output_dirs['footprints'], f"{base_name}_processed.shp")
                gdf.to_file(output_file)
            elif output_format == 'csv':
                output_file = os.path.join(output_dirs['footprints'], f"{base_name}_processed.csv")
                # Convert geometries to WKT for CSV storage
                gdf['geometry'] = gdf.geometry.to_wkt()
                gdf.to_csv(output_file, index=False)
            
            # Create metadata entry
            footprint_metadata = {
                'filename': output_file,
                'original_file': vector_file,
                'crs': target_crs,
                'feature_count': len(gdf),
                'format': output_format,
                'bounds': [float(b) for b in gdf.total_bounds],
                'processed_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            processed_footprints.append(footprint_metadata)
            
        except Exception as e:
            logger.error(f"Error processing vector file {vector_file}: {e}")
    
    # Save metadata
    metadata_file = os.path.join(output_dirs['metadata'], 'footprints_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(processed_footprints, f, indent=2)
    
    logger.info(f"Processed {len(processed_footprints)} vector files")
    return processed_footprints


def create_tiles(
    processed_images: List[Dict[str, Any]],
    processed_footprints: List[Dict[str, Any]],
    output_dirs: Dict[str, str],
    tile_size: int = 512,
    overlap: int = 64,
    validate_fraction: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create image tiles with corresponding footprints"""
    logger.info("Creating image tiles")
    
    all_tiles = []
    validation_tiles = []
    
    # Process each image
    for img_metadata in processed_images:
        try:
            image_file = img_metadata['filename']
            logger.info(f"Creating tiles for image: {os.path.basename(image_file)}")
            
            # Find corresponding footprints
            img_bounds = img_metadata['bounds']
            matching_footprints = []
            
            for fp_metadata in processed_footprints:
                fp_bounds = fp_metadata['bounds']
                
                # Check if bounding boxes overlap
                if (img_bounds[0] < fp_bounds[2] and img_bounds[2] > fp_bounds[0] and
                    img_bounds[1] < fp_bounds[3] and img_bounds[3] > fp_bounds[1]):
                    matching_footprints.append(fp_metadata['filename'])
            
            if not matching_footprints:
                logger.warning(f"No matching footprints found for {image_file}")
                continue
                
            # Open the image
            with rasterio.open(image_file) as src:
                # Get image data and transform
                img_data = src.read()
                transform = src.transform
                crs = src.crs
                
                # Calculate the number of tiles
                n_bands, height, width = img_data.shape
                effective_tile_size = tile_size - overlap
                
                n_tiles_x = max(1, int((width - overlap) / effective_tile_size))
                n_tiles_y = max(1, int((height - overlap) / effective_tile_size))
                
                logger.info(f"Creating {n_tiles_x}x{n_tiles_y} = {n_tiles_x * n_tiles_y} tiles")
                
                # Create tiles
                base_name = os.path.splitext(os.path.basename(image_file))[0]
                tile_count = 0
                
                # Decide which tiles go to validation set
                total_tiles = n_tiles_x * n_tiles_y
                n_validation = int(total_tiles * validate_fraction)
                validation_indices = set(np.random.choice(total_tiles, n_validation, replace=False))
                
                for i in range(n_tiles_x):
                    for j in range(n_tiles_y):
                        # Calculate tile bounds
                        x_start = i * effective_tile_size
                        y_start = j * effective_tile_size
                        x_end = min(x_start + tile_size, width)
                        y_end = min(y_start + tile_size, height)
                        
                        # Extract tile data
                        tile_data = img_data[:, y_start:y_end, x_start:x_end]
                        
                        # Skip tiles that are too small
                        if tile_data.shape[1] < tile_size/2 or tile_data.shape[2] < tile_size/2:
                            continue
                            
                        # Calculate new geotransform for the tile
                        tile_transform = rasterio.transform.from_origin(
                            transform.c + transform.a * x_start,
                            transform.f + transform.e * y_start,
                            transform.a,
                            transform.e
                        )
                        
                        # Create output filename
                        tile_id = f"{base_name}_tile_{i}_{j}"
                        
                        # Determine if this tile goes to validation set
                        is_validation = (i * n_tiles_y + j) in validation_indices
                        output_dir = output_dirs['validation'] if is_validation else output_dirs['tiles']
                        
                        tile_file = os.path.join(output_dir, f"{tile_id}.tif")
                        
                        # Create metadata for the tile
                        tile_metadata = {
                            'tile_id': tile_id,
                            'filename': tile_file,
                            'source_image': image_file,
                            'x_start': x_start,
                            'y_start': y_start,
                            'x_end': x_end,
                            'y_end': y_end,
                            'width': x_end - x_start,
                            'height': y_end - y_start,
                            'validation': is_validation
                        }
                        
                        # Calculate bounds of the tile in the source CRS
                        minx = tile_transform.c
                        miny = tile_transform.f + tile_transform.e * (y_end - y_start)
                        maxx = tile_transform.c + tile_transform.a * (x_end - x_start)
                        maxy = tile_transform.f
                        
                        tile_bbox = [minx, miny, maxx, maxy]
                        tile_metadata['bounds'] = tile_bbox
                        
                        # Write the tile
                        tile_profile = src.profile.copy()
                        tile_profile.update({
                            'height': y_end - y_start,
                            'width': x_end - x_start,
                            'transform': tile_transform
                        })
                        
                        with rasterio.open(tile_file, 'w', **tile_profile) as dst:
                            dst.write(tile_data)
                        
                        # Extract footprints for this tile
                        tile_footprints = []
                        
                        for fp_file in matching_footprints:
                            try:
                                # Create output filename for footprint
                                fp_base = os.path.splitext(os.path.basename(fp_file))[0]
                                fp_output_file = os.path.join(
                                    output_dir, f"{tile_id}_{fp_base}.geojson"
                                )
                                
                                # Read footprints
                                footprint_gdf = gpd.read_file(fp_file)
                                
                                # Create bounding box polygon
                                bbox = box(minx, miny, maxx, maxy)
                                
                                # Clip footprints to the tile
                                clipped_gdf = gpd.clip(footprint_gdf, bbox)
                                
                                if not clipped_gdf.empty:
                                    # Save clipped footprints
                                    clipped_gdf.to_file(fp_output_file, driver='GeoJSON')
                                    tile_metadata['footprint_file'] = fp_output_file
                                    tile_metadata['building_count'] = len(clipped_gdf)
                                    
                                    # Generate mask if footprints exist
                                    mask_file = os.path.join(output_dir, f"{tile_id}_mask.tif")
                                    
                                    # Create binary mask
                                    mask_profile = tile_profile.copy()
                                    mask_profile.update(count=1, dtype=rasterio.uint8)
                                    
                                    with rasterio.open(mask_file, 'w', **mask_profile) as dst:
                                        # Rasterize footprints
                                        shapes_list = ((geom, 1) for geom in clipped_gdf.geometry)
                                        burned = rasterio.features.rasterize(
                                            shapes=shapes_list,
                                            out_shape=(y_end - y_start, x_end - x_start),
                                            transform=tile_transform,
                                            fill=0,
                                            dtype=rasterio.uint8
                                        )
                                        dst.write(burned, 1)
                                    
                                    tile_metadata['mask_file'] = mask_file
                                
                            except Exception as e:
                                logger.error(f"Error processing footprints for tile {tile_id}: {e}")
                        
                        # Add to the appropriate list
                        if is_validation:
                            validation_tiles.append(tile_metadata)
                        else:
                            all_tiles.append(tile_metadata)
                        
                        tile_count += 1
                
                logger.info(f"Created {tile_count} tiles from {image_file}")
                
        except Exception as e:
            logger.error(f"Error creating tiles for {img_metadata['filename']}: {e}")
    
    # Save metadata
    tiles_metadata_file = os.path.join(output_dirs['metadata'], 'tiles_metadata.json')
    with open(tiles_metadata_file, 'w') as f:
        json.dump(all_tiles, f, indent=2)
    
    validation_metadata_file = os.path.join(output_dirs['metadata'], 'validation_metadata.json')
    with open(validation_metadata_file, 'w') as f:
        json.dump(validation_tiles, f, indent=2)
    
    logger.info(f"Created {len(all_tiles)} training tiles and {len(validation_tiles)} validation tiles")
    return all_tiles, validation_tiles


def create_augmentations(
    tiles: List[Dict[str, Any]], 
    output_dirs: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Create augmented versions of the tiles"""
    logger.info("Creating data augmentations")
    
    augmented_tiles = []
    
    for tile_metadata in tiles:
        try:
            tile_file = tile_metadata['filename']
            logger.info(f"Augmenting tile: {os.path.basename(tile_file)}")
            
            # Open the image
            with rasterio.open(tile_file) as src:
                tile_data = src.read()
                tile_profile = src.profile.copy()
            
            # Check if we have a mask
            mask_data = None
            if 'mask_file' in tile_metadata and os.path.exists(tile_metadata['mask_file']):
                with rasterio.open(tile_metadata['mask_file']) as mask_src:
                    mask_data = mask_src.read(1)
            
            # Define augmentations
            augmentations = [
                ('flip_h', lambda img, mask: (np.flip(img, axis=2), np.fliplr(mask) if mask is not None else None)),
                ('flip_v', lambda img, mask: (np.flip(img, axis=1), np.flipud(mask) if mask is not None else None)),
                ('rot90', lambda img, mask: (np.rot90(img, k=1, axes=(1, 2)), np.rot90(mask, k=1) if mask is not None else None)),
                ('rot180', lambda img, mask: (np.rot90(img, k=2, axes=(1, 2)), np.rot90(mask, k=2) if mask is not None else None)),
                ('rot270', lambda img, mask: (np.rot90(img, k=3, axes=(1, 2)), np.rot90(mask, k=3) if mask is not None else None))
            ]
            
            for aug_name, aug_func in augmentations:
                try:
                    # Apply augmentation
                    aug_img, aug_mask = aug_func(tile_data, mask_data)
                    
                    # Create output filenames
                    tile_id = tile_metadata['tile_id']
                    aug_tile_id = f"{tile_id}_{aug_name}"
                    aug_tile_file = os.path.join(output_dirs['augmented'], f"{aug_tile_id}.tif")
                    
                    # Write augmented image
                    with rasterio.open(aug_tile_file, 'w', **tile_profile) as dst:
                        dst.write(aug_img)
                    
                    # Create metadata for the augmented tile
                    aug_metadata = tile_metadata.copy()
                    aug_metadata.update({
                        'tile_id': aug_tile_id,
                        'filename': aug_tile_file,
                        'augmentation': aug_name,
                        'source_tile': tile_file
                    })
                    
                    # Write augmented mask if available
                    if aug_mask is not None:
                        aug_mask_file = os.path.join(output_dirs['augmented'], f"{aug_tile_id}_mask.tif")
                        mask_profile = tile_profile.copy()
                        mask_profile.update(count=1, dtype=rasterio.uint8)
                        
                        with rasterio.open(aug_mask_file, 'w', **mask_profile) as dst:
                            dst.write(aug_mask, 1)
                        
                        aug_metadata['mask_file'] = aug_mask_file
                    
                    augmented_tiles.append(aug_metadata)
                    
                except Exception as e:
                    logger.error(f"Error creating {aug_name} augmentation for {tile_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error augmenting {tile_metadata['filename']}: {e}")
    
    # Save metadata
    augmented_metadata_file = os.path.join(output_dirs['metadata'], 'augmented_metadata.json')
    with open(augmented_metadata_file, 'w') as f:
        json.dump(augmented_tiles, f, indent=2)
    
    logger.info(f"Created {len(augmented_tiles)} augmented tiles")
    return augmented_tiles


def box(minx: float, miny: float, maxx: float, maxy: float) -> Polygon:
    """Helper function to create a box polygon from bounds"""
    return Polygon([
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny)
    ])


def main() -> int:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directory structure
    if args.clean:
        clean_output_directory(args.output)
    
    output_dirs = create_directory_structure(args.output)
    
    # Find input files
    image_files = find_image_files(args.input)
    vector_files = find_vector_files(args.input)
    
    processed_images = []
    processed_footprints = []
    
    # Process imagery
    if args.mode in ['all', 'imagery']:
        processed_images = process_imagery(
            image_files, 
            output_dirs,
            bands=args.bands,
            resolution=args.resolution,
            target_crs=args.crs
        )
    
    # Process footprints
    if args.mode in ['all', 'footprints']:
        processed_footprints = process_footprints(
            vector_files, 
            output_dirs,
            output_format=args.format,
            target_crs=args.crs
        )
    
    # Create tiles
    if args.mode in ['all', 'imagery', 'footprints']:
        all_tiles, validation_tiles = create_tiles(
            processed_images,
            processed_footprints,
            output_dirs,
            tile_size=args.tile_size,
            overlap=args.overlap,
            validate_fraction=args.validate_fraction
        )
        
        # Create augmentations
        if args.augment or args.mode == 'augmentation':
            augmented_tiles = create_augmentations(all_tiles, output_dirs)
    
    logger.info("Processing complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)