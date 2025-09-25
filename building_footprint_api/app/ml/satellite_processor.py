"""
Satellite Image Processing Module
Advanced preprocessing pipeline for satellite imagery including tiling, normalization, and CRS handling
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.windows import Window
from rasterio.enums import Resampling as RioResampling
from osgeo import gdal, osr
import cv2
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging
from shapely.geometry import box, Point
import geopandas as gpd
from pyproj import Transformer, CRS
import json

logger = logging.getLogger(__name__)


class SatelliteImageProcessor:
    """
    Comprehensive satellite image processing with geospatial capabilities
    """
    
    def __init__(self, target_crs: str = "EPSG:4326", tile_size: int = 512, overlap: int = 64):
        """
        Initialize satellite image processor
        
        Args:
            target_crs: Target coordinate reference system
            tile_size: Size of image tiles for processing
            overlap: Overlap between adjacent tiles
        """
        self.target_crs = target_crs
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Standard normalization parameters for satellite imagery
        self.normalization_params = {
            'sentinel2': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'scale_factor': 10000  # Sentinel-2 scaling
            },
            'landsat8': {
                'mean': [0.4, 0.4, 0.4],
                'std': [0.2, 0.2, 0.2], 
                'scale_factor': 65535  # Landsat 8 scaling
            },
            'rgb': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'scale_factor': 255  # Standard RGB
            }
        }
        
    def load_satellite_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Load satellite image with metadata
        
        Args:
            image_path: Path to satellite image file
            
        Returns:
            Dictionary containing image data and metadata
        """
        try:
            image_path = Path(image_path)
            
            with rasterio.open(image_path) as src:
                # Read image data
                image_data = src.read()
                
                # Get metadata
                metadata = {
                    'shape': image_data.shape,
                    'dtype': str(image_data.dtype),
                    'crs': str(src.crs) if src.crs else None,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'nodata': src.nodata,
                    'count': src.count,
                    'width': src.width,
                    'height': src.height,
                    'indexes': src.indexes
                }
                
                # Detect satellite type based on metadata
                satellite_type = self._detect_satellite_type(metadata, image_path)
                
                return {
                    'image': image_data,
                    'metadata': metadata,
                    'satellite_type': satellite_type,
                    'file_path': str(image_path)
                }
                
        except Exception as e:
            logger.error(f"Failed to load satellite image {image_path}: {e}")
            raise
            
    def _detect_satellite_type(self, metadata: Dict, image_path: Path) -> str:
        """Detect satellite type from metadata and filename"""
        filename = image_path.name.lower()
        
        if 'sentinel' in filename or 's2' in filename:
            return 'sentinel2'
        elif 'landsat' in filename or 'l8' in filename or 'lc08' in filename:
            return 'landsat8'
        elif metadata['count'] == 3:  # Standard RGB
            return 'rgb'
        else:
            return 'unknown'
            
    def normalize_image(self, image_data: np.ndarray, satellite_type: str = 'rgb') -> np.ndarray:
        """
        Normalize satellite image for ML processing
        
        Args:
            image_data: Image array (C, H, W) or (H, W, C)
            satellite_type: Type of satellite imagery
            
        Returns:
            Normalized image array
        """
        try:
            # Get normalization parameters
            norm_params = self.normalization_params.get(satellite_type, self.normalization_params['rgb'])
            
            # Ensure image is in (C, H, W) format
            if image_data.ndim == 3 and image_data.shape[-1] <= 4:
                # Convert (H, W, C) to (C, H, W)
                image_data = np.transpose(image_data, (2, 0, 1))
                
            # Scale to [0, 1]
            normalized = image_data.astype(np.float32) / norm_params['scale_factor']
            
            # Apply standardization
            mean = np.array(norm_params['mean']).reshape(-1, 1, 1)
            std = np.array(norm_params['std']).reshape(-1, 1, 1)
            
            # Handle multi-channel images
            if normalized.shape[0] > len(mean):
                # Use first 3 channels or repeat mean/std
                if normalized.shape[0] == 3:
                    pass  # Use as is
                else:
                    # Take first 3 bands or use NIR-R-G for false color
                    normalized = normalized[:3]
                    
            # Apply normalization
            normalized = (normalized - mean) / std
            
            return normalized
            
        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            raise
            
    def create_tiles(self, image_data: np.ndarray, metadata: Dict) -> List[Dict]:
        """
        Create overlapping tiles from satellite image
        
        Args:
            image_data: Image array (C, H, W)
            metadata: Image metadata with geospatial information
            
        Returns:
            List of tile dictionaries with image data and coordinates
        """
        try:
            _, height, width = image_data.shape
            tiles = []
            
            # Calculate tile positions
            step_size = self.tile_size - self.overlap
            
            for y in range(0, height - self.tile_size + 1, step_size):
                for x in range(0, width - self.tile_size + 1, step_size):
                    # Extract tile
                    tile_data = image_data[:, y:y+self.tile_size, x:x+self.tile_size]
                    
                    # Calculate geographic coordinates if transform available
                    tile_bounds = None
                    if 'transform' in metadata and metadata['transform']:
                        transform = metadata['transform']
                        
                        # Calculate tile bounds in image CRS
                        left = transform[2] + x * transform[0]
                        top = transform[5] + y * transform[4]
                        right = left + self.tile_size * transform[0]
                        bottom = top + self.tile_size * transform[4]
                        
                        tile_bounds = {
                            'left': left,
                            'bottom': bottom,
                            'right': right,
                            'top': top,
                            'crs': metadata.get('crs')
                        }
                    
                    tile_info = {
                        'image': tile_data,
                        'x_offset': x,
                        'y_offset': y,
                        'tile_id': f"tile_{x}_{y}",
                        'bounds': tile_bounds
                    }
                    
                    tiles.append(tile_info)
                    
            logger.info(f"Created {len(tiles)} tiles from image")
            return tiles
            
        except Exception as e:
            logger.error(f"Tile creation failed: {e}")
            raise
            
    def reproject_image(self, src_path: Union[str, Path], dst_crs: str, output_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Reproject satellite image to target CRS
        
        Args:
            src_path: Source image path
            dst_crs: Destination CRS (e.g., 'EPSG:4326')
            output_path: Optional output path for reprojected image
            
        Returns:
            Dictionary with reprojected image info
        """
        try:
            src_path = Path(src_path)
            
            with rasterio.open(src_path) as src:
                # Calculate transform and dimensions for target CRS
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                
                # Define output profile
                dst_profile = src.profile.copy()
                dst_profile.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })
                
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Reproject and save to file
                    with rasterio.open(output_path, 'w', **dst_profile) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest
                            )
                    
                    return {
                        'status': 'success',
                        'output_path': str(output_path),
                        'dst_profile': dst_profile
                    }
                else:
                    # Reproject in memory
                    reprojected_data = np.zeros((src.count, height, width), dtype=src.dtypes[0])
                    
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=reprojected_data[i-1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )
                    
                    return {
                        'status': 'success',
                        'image': reprojected_data,
                        'profile': dst_profile
                    }
                    
        except Exception as e:
            logger.error(f"Image reprojection failed: {e}")
            raise
            
    def enhance_image(self, image_data: np.ndarray, enhancement_type: str = 'histogram') -> np.ndarray:
        """
        Apply image enhancement techniques
        
        Args:
            image_data: Image array
            enhancement_type: Type of enhancement ('histogram', 'clahe', 'gamma')
            
        Returns:
            Enhanced image array
        """
        try:
            # Convert to uint8 for processing
            if image_data.dtype != np.uint8:
                # Normalize to 0-255
                img_norm = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)
            else:
                img_norm = image_data.copy()
                
            if enhancement_type == 'histogram':
                # Histogram equalization
                if len(img_norm.shape) == 3:
                    # Multi-channel
                    enhanced = np.zeros_like(img_norm)
                    for i in range(img_norm.shape[0]):
                        enhanced[i] = cv2.equalizeHist(img_norm[i])
                else:
                    enhanced = cv2.equalizeHist(img_norm)
                    
            elif enhancement_type == 'clahe':
                # Contrast Limited Adaptive Histogram Equalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                if len(img_norm.shape) == 3:
                    enhanced = np.zeros_like(img_norm)
                    for i in range(img_norm.shape[0]):
                        enhanced[i] = clahe.apply(img_norm[i])
                else:
                    enhanced = clahe.apply(img_norm)
                    
            elif enhancement_type == 'gamma':
                # Gamma correction
                gamma = 1.2
                enhanced = np.power(img_norm / 255.0, gamma) * 255
                enhanced = enhanced.astype(np.uint8)
                
            else:
                enhanced = img_norm
                
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image_data


class GeoDataProcessor:
    """
    Process geospatial vector data (building footprints) alongside satellite imagery
    """
    
    def __init__(self):
        self.supported_formats = ['.shp', '.geojson', '.gpkg', '.kml']
        
    def load_ground_truth(self, vector_path: Union[str, Path], image_bounds: Dict) -> gpd.GeoDataFrame:
        """
        Load ground truth building footprints that intersect with image bounds
        
        Args:
            vector_path: Path to vector file containing building footprints
            image_bounds: Image bounds dictionary with CRS info
            
        Returns:
            GeoDataFrame with building footprints
        """
        try:
            vector_path = Path(vector_path)
            
            # Load vector data
            gdf = gpd.read_file(vector_path)
            
            # Create bounds geometry
            if image_bounds and 'crs' in image_bounds:
                bounds_geom = box(
                    image_bounds['left'],
                    image_bounds['bottom'], 
                    image_bounds['right'],
                    image_bounds['top']
                )
                
                # Convert to same CRS if needed
                if gdf.crs != image_bounds['crs']:
                    target_crs = CRS.from_string(image_bounds['crs'])
                    gdf = gdf.to_crs(target_crs)
                    
                # Filter to image bounds
                gdf = gdf[gdf.geometry.intersects(bounds_geom)]
                
            logger.info(f"Loaded {len(gdf)} building footprints from {vector_path}")
            return gdf
            
        except Exception as e:
            logger.error(f"Failed to load ground truth data: {e}")
            raise
            
    def rasterize_vectors(self, gdf: gpd.GeoDataFrame, reference_image: Dict, burn_value: int = 1) -> np.ndarray:
        """
        Rasterize vector geometries to match reference image
        
        Args:
            gdf: GeoDataFrame with building footprints
            reference_image: Reference image metadata for rasterization
            burn_value: Value to burn into raster for buildings
            
        Returns:
            Binary raster mask
        """
        try:
            from rasterio.features import rasterize
            
            metadata = reference_image['metadata']
            
            # Create shapes for rasterization
            shapes = [(geom, burn_value) for geom in gdf.geometry]
            
            # Rasterize
            raster_mask = rasterize(
                shapes,
                out_shape=(metadata['height'], metadata['width']),
                transform=metadata['transform'],
                fill=0,
                dtype=np.uint8
            )
            
            logger.info(f"Rasterized {len(gdf)} buildings to {raster_mask.shape} mask")
            return raster_mask
            
        except Exception as e:
            logger.error(f"Vector rasterization failed: {e}")
            raise
            
    def convert_pixels_to_geo(self, pixel_coords: List[Tuple[int, int]], transform: rasterio.transform.Affine) -> List[Tuple[float, float]]:
        """
        Convert pixel coordinates to geographic coordinates
        
        Args:
            pixel_coords: List of (x, y) pixel coordinates
            transform: Rasterio transform object
            
        Returns:
            List of (lon, lat) geographic coordinates
        """
        try:
            geo_coords = []
            
            for x, y in pixel_coords:
                # Apply affine transform
                lon = transform[2] + x * transform[0] + y * transform[1]
                lat = transform[5] + x * transform[3] + y * transform[4]
                geo_coords.append((lon, lat))
                
            return geo_coords
            
        except Exception as e:
            logger.error(f"Pixel to geo conversion failed: {e}")
            raise


class SatelliteDataset:
    """
    Dataset class for managing satellite imagery and corresponding building footprints
    """
    
    def __init__(self, data_dir: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        self.processor = SatelliteImageProcessor()
        self.geo_processor = GeoDataProcessor()
        
    def scan_dataset(self) -> Dict:
        """
        Scan data directory for satellite images and vector data
        
        Returns:
            Dictionary with dataset information
        """
        try:
            # Find satellite images
            image_extensions = ['.tif', '.tiff', '.jp2', '.img', '.hdf']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(self.data_dir.glob(f'**/*{ext}'))
                image_files.extend(self.data_dir.glob(f'**/*{ext.upper()}'))
                
            # Find vector files
            vector_extensions = ['.shp', '.geojson', '.gpkg', '.kml']
            vector_files = []
            
            for ext in vector_extensions:
                vector_files.extend(self.data_dir.glob(f'**/*{ext}'))
                vector_files.extend(self.data_dir.glob(f'**/*{ext.upper()}'))
                
            dataset_info = {
                'image_files': [str(f) for f in image_files],
                'vector_files': [str(f) for f in vector_files],
                'total_images': len(image_files),
                'total_vectors': len(vector_files)
            }
            
            logger.info(f"Found {len(image_files)} images and {len(vector_files)} vector files")
            return dataset_info
            
        except Exception as e:
            logger.error(f"Dataset scanning failed: {e}")
            raise
            
    def prepare_training_data(self, image_path: Union[str, Path], vector_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Prepare training data from satellite image and optional ground truth
        
        Args:
            image_path: Path to satellite image
            vector_path: Optional path to ground truth vector data
            
        Returns:
            Dictionary with prepared training data
        """
        try:
            # Load satellite image
            sat_data = self.processor.load_satellite_image(image_path)
            
            # Normalize image
            normalized_image = self.processor.normalize_image(
                sat_data['image'], 
                sat_data['satellite_type']
            )
            
            # Create tiles
            tiles = self.processor.create_tiles(normalized_image, sat_data['metadata'])
            
            # Load ground truth if available
            ground_truth_masks = []
            if vector_path:
                try:
                    # Create image bounds for filtering
                    bounds = {
                        'left': sat_data['metadata']['bounds'].left,
                        'bottom': sat_data['metadata']['bounds'].bottom,
                        'right': sat_data['metadata']['bounds'].right,
                        'top': sat_data['metadata']['bounds'].top,
                        'crs': sat_data['metadata']['crs']
                    }
                    
                    # Load ground truth
                    gdf = self.geo_processor.load_ground_truth(vector_path, bounds)
                    
                    # Create mask for each tile
                    for tile in tiles:
                        if tile['bounds']:
                            # Filter buildings to tile bounds
                            tile_bounds_geom = box(
                                tile['bounds']['left'],
                                tile['bounds']['bottom'],
                                tile['bounds']['right'],
                                tile['bounds']['top']
                            )
                            
                            tile_buildings = gdf[gdf.geometry.intersects(tile_bounds_geom)]
                            
                            if len(tile_buildings) > 0:
                                # Create dummy metadata for tile
                                tile_metadata = {
                                    'height': self.processor.tile_size,
                                    'width': self.processor.tile_size,
                                    'transform': rasterio.transform.from_bounds(
                                        tile['bounds']['left'],
                                        tile['bounds']['bottom'],
                                        tile['bounds']['right'],
                                        tile['bounds']['top'],
                                        self.processor.tile_size,
                                        self.processor.tile_size
                                    )
                                }
                                
                                tile_ref = {'metadata': tile_metadata}
                                mask = self.geo_processor.rasterize_vectors(tile_buildings, tile_ref)
                            else:
                                mask = np.zeros((self.processor.tile_size, self.processor.tile_size), dtype=np.uint8)
                        else:
                            mask = np.zeros((self.processor.tile_size, self.processor.tile_size), dtype=np.uint8)
                            
                        ground_truth_masks.append(mask)
                        
                except Exception as e:
                    logger.warning(f"Could not load ground truth: {e}")
                    ground_truth_masks = [None] * len(tiles)
            else:
                ground_truth_masks = [None] * len(tiles)
                
            return {
                'satellite_data': sat_data,
                'tiles': tiles,
                'ground_truth_masks': ground_truth_masks,
                'training_ready': True
            }
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = SatelliteImageProcessor(target_crs="EPSG:4326", tile_size=512)
    
    # Test with dummy data
    dummy_image = np.random.randint(0, 255, (3, 1024, 1024), dtype=np.uint8)
    dummy_metadata = {
        'shape': (3, 1024, 1024),
        'transform': rasterio.transform.from_bounds(-180, -90, 180, 90, 1024, 1024),
        'crs': 'EPSG:4326'
    }
    
    print("Satellite Image Processing Test Results:")
    
    # Test normalization
    normalized = processor.normalize_image(dummy_image, 'rgb')
    print(f"Original shape: {dummy_image.shape}, dtype: {dummy_image.dtype}")
    print(f"Normalized shape: {normalized.shape}, dtype: {normalized.dtype}")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # Test tiling
    tiles = processor.create_tiles(normalized, dummy_metadata)
    print(f"Created {len(tiles)} tiles from {dummy_image.shape} image")
    
    if tiles:
        print(f"First tile shape: {tiles[0]['image'].shape}")
        print(f"First tile ID: {tiles[0]['tile_id']}")
        if tiles[0]['bounds']:
            print(f"First tile bounds: {tiles[0]['bounds']}")
            
    print("Satellite image processing pipeline ready!")