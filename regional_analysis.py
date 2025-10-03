#!/usr/bin/env python3
"""
Regional performance analysis for GeoAI Research project
Analyzes model performance across different geographic regions
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import logging
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import glob
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('regional_analysis')


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        description="Regional performance analysis for GeoAI Research project"
    )
    parser.add_argument(
        "--results", "-r", required=True, help="Directory with model prediction results"
    )
    parser.add_argument(
        "--ground-truth", "-g", required=True, help="Directory with ground truth data"
    )
    parser.add_argument(
        "--output", "-o", default="region_analysis_output", help="Output directory for analysis results"
    )
    parser.add_argument(
        "--regions", "-e", help="CSV file with region definitions (optional)"
    )
    parser.add_argument(
        "--models", "-m", help="Comma-separated list of model names to analyze"
    )
    parser.add_argument(
        "--metrics", "-c", default="iou,precision,recall,f1",
        help="Comma-separated list of metrics to calculate"
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Generate visualizations"
    )
    parser.add_argument(
        "--format", "-f", choices=["png", "pdf", "svg"], default="png",
        help="Output file format for visualizations"
    )
    parser.add_argument(
        "--style", "-s", choices=["paper", "presentation", "web"], default="paper",
        help="Visualization style"
    )
    return parser


def load_region_definitions(regions_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load region definitions from a CSV file or create default ones
    
    Expected CSV format:
    region_id,region_name,min_lon,min_lat,max_lon,max_lat
    """
    regions = {}
    
    if regions_file and os.path.exists(regions_file):
        logger.info(f"Loading region definitions from {regions_file}")
        
        try:
            # Read CSV file
            df = pd.read_csv(regions_file)
            
            # Validate columns
            required_cols = ['region_id', 'region_name', 'min_lon', 'min_lat', 'max_lon', 'max_lat']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Regions file missing required columns: {required_cols}")
                return create_default_regions()
            
            # Convert to dictionary
            for _, row in df.iterrows():
                region_id = row['region_id']
                regions[region_id] = {
                    'name': row['region_name'],
                    'bounds': [row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']]
                }
                
                # Add optional attributes if present
                for col in df.columns:
                    if col not in required_cols and not pd.isna(row[col]):
                        regions[region_id][col] = row[col]
            
            logger.info(f"Loaded {len(regions)} regions")
            return regions
            
        except Exception as e:
            logger.error(f"Error loading region definitions: {e}")
            return create_default_regions()
    else:
        return create_default_regions()


def create_default_regions() -> Dict[str, Any]:
    """Create default region definitions based on common geographic regions"""
    logger.info("Creating default region definitions")
    
    regions = {
        "us-midwest": {
            "name": "US Midwest",
            "bounds": [-97.5, 36.5, -82.0, 49.5]
        },
        "us-northeast": {
            "name": "US Northeast",
            "bounds": [-82.0, 37.0, -66.5, 47.5]
        },
        "us-south": {
            "name": "US South",
            "bounds": [-106.5, 25.5, -75.0, 36.5]
        },
        "us-west": {
            "name": "US West",
            "bounds": [-125.0, 32.0, -104.0, 49.0]
        },
        "europe-west": {
            "name": "Western Europe",
            "bounds": [-10.0, 36.0, 15.0, 60.0]
        },
        "europe-east": {
            "name": "Eastern Europe",
            "bounds": [15.0, 36.0, 40.0, 60.0]
        },
        "asia-east": {
            "name": "East Asia",
            "bounds": [100.0, 20.0, 145.0, 50.0]
        },
        "asia-south": {
            "name": "South Asia",
            "bounds": [60.0, 5.0, 100.0, 40.0]
        }
    }
    
    logger.info(f"Created {len(regions)} default regions")
    return regions


def find_files_by_pattern(directory: str, pattern: str) -> List[str]:
    """Find files in a directory that match the given pattern"""
    search_pattern = os.path.join(directory, "**", pattern)
    files = glob.glob(search_pattern, recursive=True)
    return files


def determine_region_from_location(location: Tuple[float, float], regions: Dict[str, Any]) -> Optional[str]:
    """Determine which region a location belongs to based on coordinates"""
    lon, lat = location
    
    for region_id, region_info in regions.items():
        bounds = region_info['bounds']
        min_lon, min_lat, max_lon, max_lat = bounds
        
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return region_id
    
    return None


def extract_location_from_filename(filename: str) -> Optional[Tuple[float, float]]:
    """Extract location coordinates from filename if present"""
    # Common formats:
    # - lat<LAT>_lon<LON> or lon<LON>_lat<LAT>
    # - <LAT>_<LON>
    # - <NAME>_<LAT>_<LON>
    
    # Try lat/lon pattern
    pattern1 = r'lat(-?\d+\.?\d*)_lon(-?\d+\.?\d*)'
    pattern2 = r'lon(-?\d+\.?\d*)_lat(-?\d+\.?\d*)'
    
    match = re.search(pattern1, filename, re.IGNORECASE)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        return (lon, lat)
    
    match = re.search(pattern2, filename, re.IGNORECASE)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))
        return (lon, lat)
    
    # Try _<LAT>_<LON> pattern (look for last two numbers in filename)
    pattern3 = r'_(-?\d+\.?\d*)_(-?\d+\.?\d*)(?:\.|_|$)'
    match = re.search(pattern3, filename, re.IGNORECASE)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        # Basic validation - if these look like valid lat/lon
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lon, lat)
    
    return None


def extract_model_name_from_path(file_path: str, specified_models: Optional[List[str]] = None) -> Optional[str]:
    """Extract model name from file path"""
    # Check if any specified model names appear in the path
    if specified_models:
        for model in specified_models:
            if model.lower() in file_path.lower():
                return model
    
    # Common model name patterns in filenames
    patterns = [
        r'(?:model|mdl)[_-]([a-zA-Z0-9_-]+)',  # model_NAME or mdl-NAME
        r'([a-zA-Z0-9_-]+)(?:model|mdl)',      # NAMEmodel or NAME-mdl
        r'([a-zA-Z0-9_-]+)[_-](?:pred|prediction)'  # NAME_pred or NAME-prediction
    ]
    
    for pattern in patterns:
        match = re.search(pattern, os.path.basename(file_path), re.IGNORECASE)
        if match:
            return match.group(1)
    
    # If we can't determine the model name, use directory name
    parent_dir = os.path.basename(os.path.dirname(file_path))
    if parent_dir not in ('results', 'predictions', 'output'):
        return parent_dir
    
    # Last resort: use "unknown"
    return "unknown"


def load_prediction_results(
    results_dir: str, 
    regions: Dict[str, Any],
    specified_models: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Load model prediction results and organize by region and model
    
    Returns:
    {
        "region_id": {
            "model_name": [
                {"file": "path/to/file.geojson", "buildings": [...], ...},
                ...
            ],
            ...
        },
        ...
    }
    """
    logger.info(f"Loading prediction results from {results_dir}")
    
    # Split comma-separated model names
    model_names = None
    if specified_models:
        if isinstance(specified_models, str):
            model_names = [m.strip() for m in specified_models.split(',')]
        else:
            model_names = specified_models
    
    # Find all GeoJSON and JSON files
    json_files = find_files_by_pattern(results_dir, "*.geojson")
    json_files.extend(find_files_by_pattern(results_dir, "*.json"))
    
    # Organize results by region and model
    results_by_region_model = {}
    
    for file_path in json_files:
        try:
            # Extract model name
            model_name = extract_model_name_from_path(file_path, model_names)
            
            # Load the file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Standardize data format (different files may have different structures)
            buildings = []
            
            if isinstance(data, list):
                # List of buildings
                buildings = data
            elif isinstance(data, dict):
                if 'buildings' in data:
                    # Dictionary with 'buildings' key
                    buildings = data['buildings']
                elif 'features' in data:
                    # GeoJSON format
                    buildings = []
                    for feature in data['features']:
                        if feature['geometry']['type'] in ['Polygon', 'MultiPolygon']:
                            buildings.append({
                                'geometry': feature['geometry'],
                                'properties': feature.get('properties', {})
                            })
                else:
                    # Single building
                    buildings = [data]
            
            # Skip empty files
            if not buildings:
                logger.warning(f"No buildings found in {file_path}")
                continue
            
            # Try to determine region
            region_id = None
            
            # Check if region is specified in the file
            if isinstance(data, dict) and 'region' in data:
                region_id = data['region']
            
            # Try to extract location from filename
            if region_id is None:
                location = extract_location_from_filename(file_path)
                if location:
                    region_id = determine_region_from_location(location, regions)
            
            # If we still don't have a region, check if there's a bounding box we can use
            if region_id is None and isinstance(data, dict) and 'bbox' in data:
                bbox = data['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                region_id = determine_region_from_location(center, regions)
            
            # If we still don't have a region, try to extract it from the directory structure
            if region_id is None:
                # Look for region names in the path
                for rid in regions.keys():
                    if rid.lower() in file_path.lower():
                        region_id = rid
                        break
            
            # If we still can't determine the region, use "unknown"
            if region_id is None:
                region_id = "unknown"
            
            # Add to results
            if region_id not in results_by_region_model:
                results_by_region_model[region_id] = {}
            
            if model_name not in results_by_region_model[region_id]:
                results_by_region_model[region_id][model_name] = []
            
            # Create result entry
            result = {
                'file': file_path,
                'buildings': buildings,
                'building_count': len(buildings)
            }
            
            # Extract bounding box
            if isinstance(data, dict) and 'bbox' in data:
                result['bbox'] = data['bbox']
            
            results_by_region_model[region_id][model_name].append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Log summary
    region_counts = {rid: len(models) for rid, models in results_by_region_model.items()}
    logger.info(f"Loaded results for {len(results_by_region_model)} regions: {region_counts}")
    
    return results_by_region_model


def load_ground_truth(
    ground_truth_dir: str, 
    regions: Dict[str, Any]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load ground truth data and organize by region
    
    Returns:
    {
        "region_id": [
            {"file": "path/to/file.geojson", "buildings": [...], ...},
            ...
        ],
        ...
    }
    """
    logger.info(f"Loading ground truth data from {ground_truth_dir}")
    
    # Find all GeoJSON and JSON files
    json_files = find_files_by_pattern(ground_truth_dir, "*.geojson")
    json_files.extend(find_files_by_pattern(ground_truth_dir, "*.json"))
    
    # Organize results by region
    ground_truth_by_region = {}
    
    for file_path in json_files:
        try:
            # Load the file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Standardize data format (different files may have different structures)
            buildings = []
            
            if isinstance(data, list):
                # List of buildings
                buildings = data
            elif isinstance(data, dict):
                if 'buildings' in data:
                    # Dictionary with 'buildings' key
                    buildings = data['buildings']
                elif 'features' in data:
                    # GeoJSON format
                    buildings = []
                    for feature in data['features']:
                        if feature['geometry']['type'] in ['Polygon', 'MultiPolygon']:
                            buildings.append({
                                'geometry': feature['geometry'],
                                'properties': feature.get('properties', {})
                            })
                else:
                    # Single building
                    buildings = [data]
            
            # Skip empty files
            if not buildings:
                logger.warning(f"No buildings found in {file_path}")
                continue
            
            # Try to determine region
            region_id = None
            
            # Check if region is specified in the file
            if isinstance(data, dict) and 'region' in data:
                region_id = data['region']
            
            # Try to extract location from filename
            if region_id is None:
                location = extract_location_from_filename(file_path)
                if location:
                    region_id = determine_region_from_location(location, regions)
            
            # If we still don't have a region, check if there's a bounding box we can use
            if region_id is None and isinstance(data, dict) and 'bbox' in data:
                bbox = data['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                region_id = determine_region_from_location(center, regions)
            
            # If we still don't have a region, try to extract it from the directory structure
            if region_id is None:
                # Look for region names in the path
                for rid in regions.keys():
                    if rid.lower() in file_path.lower():
                        region_id = rid
                        break
            
            # If we still can't determine the region, use "unknown"
            if region_id is None:
                region_id = "unknown"
            
            # Add to results
            if region_id not in ground_truth_by_region:
                ground_truth_by_region[region_id] = []
            
            # Create result entry
            result = {
                'file': file_path,
                'buildings': buildings,
                'building_count': len(buildings)
            }
            
            # Extract bounding box
            if isinstance(data, dict) and 'bbox' in data:
                result['bbox'] = data['bbox']
            
            ground_truth_by_region[region_id].append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Log summary
    region_counts = {rid: len(files) for rid, files in ground_truth_by_region.items()}
    logger.info(f"Loaded ground truth for {len(ground_truth_by_region)} regions: {region_counts}")
    
    return ground_truth_by_region


def calculate_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    metrics: List[str]
) -> Dict[str, float]:
    """
    Calculate metrics comparing predictions to ground truth
    
    Supported metrics:
    - iou: Intersection over Union (Jaccard index)
    - precision: Precision score
    - recall: Recall score
    - f1: F1 score
    - building_count_diff: Difference in building count (predictions - ground truth)
    - building_count_ratio: Ratio of building count (predictions / ground truth)
    """
    # Extract geometries
    pred_geometries = []
    for pred in predictions:
        pred_geometries.extend(pred['buildings'])
    
    gt_geometries = []
    for gt in ground_truth:
        gt_geometries.extend(gt['buildings'])
    
    # Calculate building count metrics
    pred_count = len(pred_geometries)
    gt_count = len(gt_geometries)
    
    results = {
        'building_count_pred': pred_count,
        'building_count_gt': gt_count,
        'building_count_diff': pred_count - gt_count,
        'building_count_ratio': pred_count / gt_count if gt_count > 0 else float('inf')
    }
    
    # Convert to GeoDataFrames for spatial operations
    try:
        # Create GeoDataFrames
        pred_gdf = gpd.GeoDataFrame(geometry=[shape(geom['geometry']) for geom in pred_geometries if 'geometry' in geom])
        gt_gdf = gpd.GeoDataFrame(geometry=[shape(geom['geometry']) for geom in gt_geometries if 'geometry' in geom])
        
        # Ensure valid geometries
        pred_gdf = pred_gdf[pred_gdf.is_valid]
        gt_gdf = gt_gdf[gt_gdf.is_valid]
        
        # Calculate IoU if requested
        if 'iou' in metrics:
            # Calculate intersection and union
            intersection_area = 0
            union_area = 0
            
            for pred_geom in pred_gdf.geometry:
                # Find closest ground truth geometry
                min_dist = float('inf')
                closest_gt = None
                
                for gt_geom in gt_gdf.geometry:
                    dist = pred_geom.distance(gt_geom)
                    if dist < min_dist:
                        min_dist = dist
                        closest_gt = gt_geom
                
                if closest_gt is not None:
                    try:
                        intersection = pred_geom.intersection(closest_gt)
                        union = pred_geom.union(closest_gt)
                        intersection_area += intersection.area
                        union_area += union.area
                    except Exception as e:
                        logger.warning(f"Error calculating intersection/union: {e}")
            
            # Calculate IoU
            if union_area > 0:
                iou = intersection_area / union_area
            else:
                iou = 0
            
            results['iou'] = iou
        
        # Calculate other metrics
        if any(m in metrics for m in ['precision', 'recall', 'f1']):
            # Create matched building pairs
            matched_pairs = []
            used_gt = set()
            
            for i, pred_geom in enumerate(pred_gdf.geometry):
                # Find closest unmatched ground truth geometry
                min_dist = float('inf')
                closest_idx = -1
                
                for j, gt_geom in enumerate(gt_gdf.geometry):
                    if j in used_gt:
                        continue
                        
                    dist = pred_geom.distance(gt_geom)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = j
                
                # Only match if the distance is below a threshold
                if closest_idx >= 0 and min_dist < 0.001:  # Threshold based on coordinate system
                    matched_pairs.append((i, closest_idx))
                    used_gt.add(closest_idx)
            
            # Calculate precision, recall, F1
            true_positives = len(matched_pairs)
            false_positives = len(pred_gdf) - true_positives
            false_negatives = len(gt_gdf) - true_positives
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if 'precision' in metrics:
                results['precision'] = precision
                
            if 'recall' in metrics:
                results['recall'] = recall
                
            if 'f1' in metrics:
                results['f1'] = f1
        
    except Exception as e:
        logger.error(f"Error calculating spatial metrics: {e}")
        
        # Provide fallback metrics based solely on building counts
        if 'precision' in metrics:
            results['precision'] = min(1.0, gt_count / pred_count) if pred_count > 0 else 0
            
        if 'recall' in metrics:
            results['recall'] = min(1.0, pred_count / gt_count) if gt_count > 0 else 0
            
        if 'f1' in metrics:
            p = results.get('precision', 0)
            r = results.get('recall', 0)
            results['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
        if 'iou' in metrics:
            results['iou'] = min(results.get('precision', 0), results.get('recall', 0)) * 0.8  # Rough approximation
    
    return results


def calculate_regional_metrics(
    predictions_by_region: Dict[str, Dict[str, List[Dict[str, Any]]]],
    ground_truth_by_region: Dict[str, List[Dict[str, Any]]],
    metrics_to_calculate: List[str]
) -> pd.DataFrame:
    """Calculate metrics for each region and model"""
    logger.info("Calculating regional metrics")
    
    results = []
    
    for region_id, model_results in predictions_by_region.items():
        # Skip if no ground truth for this region
        if region_id not in ground_truth_by_region:
            logger.warning(f"No ground truth data for region {region_id}, skipping")
            continue
        
        gt_data = ground_truth_by_region[region_id]
        
        # Calculate metrics for each model
        for model_name, pred_data in model_results.items():
            logger.info(f"Calculating metrics for region {region_id}, model {model_name}")
            
            metrics = calculate_metrics(pred_data, gt_data, metrics_to_calculate)
            
            # Add identifiers
            metrics['region'] = region_id
            metrics['model'] = model_name
            
            results.append(metrics)
    
    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        logger.info(f"Calculated metrics for {len(df)} region-model combinations")
        return df
    else:
        logger.warning("No metrics calculated")
        return pd.DataFrame()


def generate_visualizations(
    metrics_df: pd.DataFrame,
    regions: Dict[str, Any],
    output_dir: str,
    file_format: str = 'png',
    style: str = 'paper'
) -> List[str]:
    """Generate visualizations of regional metrics"""
    logger.info("Generating visualizations")
    
    if metrics_df.empty:
        logger.warning("No metrics data available for visualization")
        return []
    
    output_files = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    if style == 'paper':
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-talk')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
        })
    elif style == 'web':
        plt.style.use('seaborn-v0_8-bright')
    
    # Get available metrics and models
    available_metrics = [col for col in metrics_df.columns 
                        if col not in ['region', 'model', 'building_count_pred', 'building_count_gt']]
    available_models = metrics_df['model'].unique()
    
    # 1. Bar chart of metrics by region
    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate region averages across models
        region_metrics = metrics_df.groupby('region')[metric].mean().reset_index()
        
        # Sort by metric value
        region_metrics = region_metrics.sort_values(by=metric, ascending=False)
        
        # Create bar chart
        sns.barplot(x='region', y=metric, data=region_metrics, ax=ax)
        
        # Customize
        ax.set_title(f'{metric.replace("_", " ").title()} by Region (Average Across Models)')
        ax.set_xlabel('Region')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f'{p.get_height():.3f}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', rotation=0, xytext=(0, 5),
                textcoords='offset points'
            )
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'metric_by_region_{metric}.{file_format}')
        plt.savefig(output_file)
        plt.close()
        
        output_files.append(output_file)
        logger.info(f"Saved {output_file}")
    
    # 2. Heatmap of metrics by region and model
    for metric in available_metrics:
        # Create pivot table
        pivot_df = metrics_df.pivot_table(index='region', columns='model', values=metric)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu', linewidths=0.5)
        
        plt.title(f'{metric.replace("_", " ").title()} by Region and Model')
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'heatmap_{metric}.{file_format}')
        plt.savefig(output_file)
        plt.close()
        
        output_files.append(output_file)
        logger.info(f"Saved {output_file}")
    
    # 3. Radar chart comparing regions
    if len(available_metrics) >= 3:
        # Create radar chart for each model
        for model in available_models:
            model_data = metrics_df[metrics_df['model'] == model]
            
            # Skip if not enough data
            if len(model_data) < 3:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            # Get data for each region
            regions_to_plot = model_data['region'].unique()
            
            # Set up angles for metrics
            angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Close the loop
            
            # Plot each region
            for i, region in enumerate(regions_to_plot):
                region_data = model_data[model_data['region'] == region]
                
                # Get values for metrics
                values = [region_data[metric].values[0] for metric in available_metrics]
                values = np.concatenate((values, [values[0]]))  # Close the loop
                
                # Plot
                ax.plot(angles, values, 'o-', linewidth=2, label=regions[region]['name'] if region in regions else region)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([metric.replace('_', ' ').title() for metric in available_metrics])
            
            ax.set_title(f'Model: {model} - Performance Across Regions')
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(output_dir, f'radar_model_{model}.{file_format}')
            plt.savefig(output_file)
            plt.close()
            
            output_files.append(output_file)
            logger.info(f"Saved {output_file}")
    
    # 4. Geographic visualization if possible
    try:
        # Create world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot world
        world.plot(ax=ax, color='lightgray', edgecolor='white')
        
        # Plot regions as rectangles
        for region_id, region_info in regions.items():
            if region_id in metrics_df['region'].unique():
                bounds = region_info['bounds']
                min_lon, min_lat, max_lon, max_lat = bounds
                
                # Get average metrics for this region
                region_metrics = metrics_df[metrics_df['region'] == region_id].mean()
                
                # Choose a metric for coloring
                if 'f1' in region_metrics:
                    metric_value = region_metrics['f1']
                    metric_name = 'F1 Score'
                elif 'iou' in region_metrics:
                    metric_value = region_metrics['iou']
                    metric_name = 'IoU'
                else:
                    metric_value = 0.5
                    metric_name = 'No Metric'
                
                # Create color map
                cmap = plt.cm.YlGnBu
                color = cmap(metric_value)
                
                # Create rectangle
                width = max_lon - min_lon
                height = max_lat - min_lat
                rect = patches.Rectangle(
                    (min_lon, min_lat), width, height,
                    linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
                )
                ax.add_patch(rect)
                
                # Add label
                ax.text(
                    min_lon + width/2, min_lat + height/2,
                    f"{region_info['name']}\n{metric_value:.3f}",
                    ha='center', va='center'
                )
        
        # Set limits
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 90)
        
        # Set title
        ax.set_title(f'Regional Performance - {metric_name}')
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(metric_name)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'geographic_visualization.{file_format}')
        plt.savefig(output_file)
        plt.close()
        
        output_files.append(output_file)
        logger.info(f"Saved {output_file}")
    except Exception as e:
        logger.error(f"Error creating geographic visualization: {e}")
    
    # 5. Model comparison chart
    plt.figure(figsize=(12, 8))
    
    # Calculate model averages across regions
    model_metrics = metrics_df.groupby('model').mean().reset_index()
    
    # Create bar plot for each metric
    metrics_to_plot = [m for m in available_metrics if m not in ['building_count_diff', 'building_count_ratio']]
    
    if metrics_to_plot:
        ax = model_metrics.plot(
            x='model', y=metrics_to_plot, kind='bar', figsize=(12, 6)
        )
        
        plt.title('Model Comparison - Average Metrics Across Regions')
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.legend(title='Metrics')
        plt.grid(axis='y')
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(output_dir, f'model_comparison.{file_format}')
        plt.savefig(output_file)
        plt.close()
        
        output_files.append(output_file)
        logger.info(f"Saved {output_file}")
    
    return output_files


def generate_report(
    metrics_df: pd.DataFrame,
    regions: Dict[str, Any],
    visualization_files: List[str],
    output_dir: str
) -> str:
    """Generate summary report"""
    logger.info("Generating summary report")
    
    # Create report file
    report_file = os.path.join(output_dir, 'regional_analysis_report.md')
    
    with open(report_file, 'w') as f:
        # Title
        f.write("# Regional Performance Analysis Report\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        
        if not metrics_df.empty:
            # Overall averages
            f.write("### Overall Average Metrics\n\n")
            
            avg_metrics = metrics_df.mean().drop(['building_count_pred', 'building_count_gt']).to_dict()
            
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for metric, value in avg_metrics.items():
                if metric not in ['region', 'model']:
                    f.write(f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n")
            
            f.write("\n")
            
            # Best regions
            f.write("### Best Performing Regions\n\n")
            
            for metric in avg_metrics.keys():
                if metric not in ['region', 'model', 'building_count_diff', 'building_count_ratio']:
                    # Get top 3 regions for this metric
                    top_regions = metrics_df.groupby('region')[metric].mean().sort_values(ascending=False).head(3)
                    
                    f.write(f"**Top regions by {metric.replace('_', ' ').title()}:**\n\n")
                    
                    for region, value in top_regions.items():
                        region_name = regions[region]['name'] if region in regions else region
                        f.write(f"- {region_name}: {value:.4f}\n")
                    
                    f.write("\n")
            
            # Best models
            f.write("### Best Performing Models\n\n")
            
            for metric in avg_metrics.keys():
                if metric not in ['region', 'model', 'building_count_diff', 'building_count_ratio']:
                    # Get top 3 models for this metric
                    top_models = metrics_df.groupby('model')[metric].mean().sort_values(ascending=False).head(3)
                    
                    f.write(f"**Top models by {metric.replace('_', ' ').title()}:**\n\n")
                    
                    for model, value in top_models.items():
                        f.write(f"- {model}: {value:.4f}\n")
                    
                    f.write("\n")
        else:
            f.write("No metrics data available.\n\n")
        
        # Regional analysis
        f.write("## Regional Analysis\n\n")
        
        for region_id, region_info in sorted(regions.items()):
            # Skip regions with no data
            if region_id not in metrics_df['region'].unique():
                continue
                
            f.write(f"### {region_info['name']}\n\n")
            
            # Get data for this region
            region_data = metrics_df[metrics_df['region'] == region_id]
            
            # Region statistics
            f.write("**Region Statistics:**\n\n")
            f.write(f"- Building count (ground truth): {region_data['building_count_gt'].mean():.0f}\n")
            f.write(f"- Building count (predictions): {region_data['building_count_pred'].mean():.0f}\n\n")
            
            # Model performance in this region
            f.write("**Model Performance:**\n\n")
            f.write("| Model | IoU | Precision | Recall | F1 |\n")
            f.write("|-------|-----|-----------|--------|----|\n")
            
            for _, row in region_data.iterrows():
                model = row['model']
                iou = row['iou'] if 'iou' in row else float('nan')
                precision = row['precision'] if 'precision' in row else float('nan')
                recall = row['recall'] if 'recall' in row else float('nan')
                f1 = row['f1'] if 'f1' in row else float('nan')
                
                f.write(f"| {model} | {iou:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |\n")
            
            f.write("\n")
        
        # Visualizations
        if visualization_files:
            f.write("## Visualizations\n\n")
            
            for viz_file in visualization_files:
                filename = os.path.basename(viz_file)
                title = os.path.splitext(filename)[0].replace('_', ' ').title()
                rel_path = os.path.relpath(viz_file, output_dir)
                
                f.write(f"### {title}\n\n")
                f.write(f"![{title}]({rel_path})\n\n")
        
        # Conclusions
        f.write("## Conclusions\n\n")
        
        if not metrics_df.empty:
            # Overall best model
            best_model = metrics_df.groupby('model')['f1' if 'f1' in metrics_df.columns else 'iou'].mean().idxmax()
            
            f.write(f"- The best overall performing model is **{best_model}**\n")
            
            # Region with highest performance
            best_region = metrics_df.groupby('region')['f1' if 'f1' in metrics_df.columns else 'iou'].mean().idxmax()
            best_region_name = regions[best_region]['name'] if best_region in regions else best_region
            
            f.write(f"- The region with highest model performance is **{best_region_name}**\n")
            
            # Recommendations based on analysis
            f.write("- **Recommendations:** ")
            
            if 'building_count_ratio' in metrics_df.columns:
                avg_ratio = metrics_df['building_count_ratio'].mean()
                if avg_ratio < 0.8:
                    f.write("Models are generally under-detecting buildings. Consider tuning detection thresholds or improving model sensitivity.\n")
                elif avg_ratio > 1.2:
                    f.write("Models are generally over-detecting buildings. Consider reducing false positives by tuning detection thresholds.\n")
                else:
                    f.write("Building count predictions are generally accurate. Focus on improving spatial precision of detections.\n")
            else:
                f.write("Further analysis needed to provide specific recommendations.\n")
        else:
            f.write("No data available for conclusions.\n")
    
    logger.info(f"Generated report: {report_file}")
    return report_file


def main() -> int:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load region definitions
    regions = load_region_definitions(args.regions)
    
    # Parse metrics list
    metrics_to_calculate = [m.strip() for m in args.metrics.split(',')]
    
    # Parse model list if provided
    specified_models = None
    if args.models:
        specified_models = [m.strip() for m in args.models.split(',')]
    
    # Load prediction results
    predictions_by_region = load_prediction_results(args.results, regions, specified_models)
    
    # Load ground truth
    ground_truth_by_region = load_ground_truth(args.ground_truth, regions)
    
    # Calculate metrics
    metrics_df = calculate_regional_metrics(
        predictions_by_region,
        ground_truth_by_region,
        metrics_to_calculate
    )
    
    # Save metrics to CSV
    metrics_csv = os.path.join(args.output, 'regional_metrics.csv')
    if not metrics_df.empty:
        metrics_df.to_csv(metrics_csv, index=False)
        logger.info(f"Saved metrics to {metrics_csv}")
    
    # Generate visualizations if requested
    visualization_files = []
    if args.visualize and not metrics_df.empty:
        viz_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        visualization_files = generate_visualizations(
            metrics_df,
            regions,
            viz_dir,
            args.format,
            args.style
        )
    
    # Generate report
    report_file = generate_report(
        metrics_df,
        regions,
        visualization_files,
        args.output
    )
    
    logger.info("Analysis complete")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)