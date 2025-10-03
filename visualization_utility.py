#!/usr/bin/env python3
"""
Visualization utility for GeoAI Research project
Creates various visualizations of building footprints and model performance
"""

import argparse
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from pathlib import Path


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line arguments"""
    parser = argparse.ArgumentParser(
        description="Visualization utility for GeoAI Research project"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input directory with results data"
    )
    parser.add_argument(
        "--output", "-o", default="visualization_output", help="Output directory for visualizations"
    )
    parser.add_argument(
        "--type", "-t", choices=["buildings", "performance", "comparison", "all"], 
        default="all", help="Type of visualization to generate"
    )
    parser.add_argument(
        "--region", "-r", help="Region to visualize (e.g., 'usa-midwest')"
    )
    parser.add_argument(
        "--format", "-f", choices=["png", "pdf", "svg"], default="png",
        help="Output file format"
    )
    parser.add_argument(
        "--dpi", "-d", type=int, default=300, help="DPI for output images"
    )
    parser.add_argument(
        "--style", "-s", choices=["paper", "presentation", "web"], default="paper",
        help="Visualization style"
    )
    return parser


def load_building_data(input_dir: str, region: Optional[str] = None) -> List[Dict]:
    """Load building footprint data from JSON files"""
    print(f"Loading building data from {input_dir}...")
    
    building_data = []
    
    # Find all JSON files in the input directory
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json') and 'building' in file.lower():
                if region is None or region.lower() in file.lower():
                    json_files.append(os.path.join(root, file))
    
    # Load data from each file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Extract relevant building information
                if isinstance(data, list):
                    # List of buildings
                    building_data.extend(data)
                elif isinstance(data, dict):
                    if 'buildings' in data:
                        # Dictionary with 'buildings' key
                        building_data.extend(data['buildings'])
                    elif 'features' in data:
                        # GeoJSON format
                        for feature in data['features']:
                            if feature['geometry']['type'] in ['Polygon', 'MultiPolygon']:
                                building_data.append({
                                    'geometry': feature['geometry'],
                                    'properties': feature.get('properties', {})
                                })
                    else:
                        # Single building
                        building_data.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(building_data)} buildings")
    return building_data


def load_performance_data(input_dir: str, region: Optional[str] = None) -> pd.DataFrame:
    """Load model performance data from CSV files"""
    print(f"Loading performance data from {input_dir}...")
    
    # Find all CSV files in the input directory
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv') and ('performance' in file.lower() or 'metrics' in file.lower()):
                if region is None or region.lower() in file.lower():
                    csv_files.append(os.path.join(root, file))
    
    # Load and combine data from each file
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Extract region from filename if not in the data
            if 'region' not in df.columns:
                filename = os.path.basename(csv_file)
                region_name = filename.split('_')[0] if '_' in filename else 'unknown'
                df['region'] = region_name
            
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    # Combine all data
    if df_list:
        performance_df = pd.concat(df_list, ignore_index=True)
        print(f"Loaded performance data with {len(performance_df)} records")
        return performance_df
    else:
        print("No performance data found")
        return pd.DataFrame()


def visualize_buildings(
    building_data: List[Dict], 
    output_dir: str, 
    region: Optional[str] = None,
    file_format: str = 'png',
    dpi: int = 300,
    style: str = 'paper'
) -> List[str]:
    """Visualize building footprints"""
    print("Generating building footprint visualizations...")
    
    output_files = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style based on parameter
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
        
    # Group buildings by area or region if available
    building_groups = {}
    
    for building in building_data:
        # Try to get area/region from properties
        area = None
        if 'properties' in building:
            area = building['properties'].get('region') or building['properties'].get('area')
        
        if not area and region:
            area = region
        
        if not area:
            area = 'unknown'
            
        if area not in building_groups:
            building_groups[area] = []
        
        building_groups[area].append(building)
    
    # Generate visualizations for each group
    for area_name, buildings in building_groups.items():
        print(f"Generating visualization for {area_name} with {len(buildings)} buildings")
        
        # Skip if there are too few buildings
        if len(buildings) < 5:
            print(f"Skipping {area_name}: too few buildings")
            continue
            
        # Extract coordinates for visualization
        coordinates = []
        for building in buildings:
            if 'geometry' in building:
                geom = building['geometry']
                if geom['type'] == 'Polygon':
                    coords = geom['coordinates'][0]  # First ring (exterior)
                    coordinates.extend(coords)
                elif geom['type'] == 'MultiPolygon':
                    for poly in geom['coordinates']:
                        coords = poly[0]  # First ring of each polygon
                        coordinates.extend(coords)
        
        if not coordinates:
            print(f"Skipping {area_name}: no valid coordinates")
            continue
            
        # Convert to numpy array for easier processing
        coords_array = np.array(coordinates)
        
        # Handle empty or invalid data
        if coords_array.size == 0 or np.isnan(coords_array).any():
            print(f"Skipping {area_name}: invalid coordinates")
            continue
            
        # Calculate bounds
        min_x, min_y = coords_array.min(axis=0)
        max_x, max_y = coords_array.max(axis=0)
        
        # Add some padding
        padding = 0.05
        x_range = max_x - min_x
        y_range = max_y - min_y
        min_x -= x_range * padding
        max_x += x_range * padding
        min_y -= y_range * padding
        max_y += y_range * padding
        
        # Create figure for density plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a 2D histogram
        x, y = coords_array[:, 0], coords_array[:, 1]
        heatmap, xedges, yedges = np.histogram2d(
            x, y, bins=(50, 50), 
            range=[[min_x, max_x], [min_y, max_y]]
        )
        
        # Create custom colormap (blue to red)
        cmap = LinearSegmentedColormap.from_list(
            'custom_cmap', ['#FFFFFF', '#EBF5FB', '#D6EAF8', '#AED6F1', '#85C1E9', '#5DADE2', '#3498DB', '#2E86C1', '#2874A6', '#1B4F72']
        )
        
        # Display the heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            heatmap.T, 
            origin='lower', 
            extent=extent,
            aspect='auto', 
            cmap=cmap,
            alpha=0.7
        )
        
        # Plot individual buildings (up to a reasonable number)
        max_buildings_to_plot = 100
        sample_buildings = buildings
        if len(buildings) > max_buildings_to_plot:
            sample_buildings = np.random.choice(buildings, max_buildings_to_plot, replace=False)
            
        for building in sample_buildings:
            if 'geometry' in building:
                geom = building['geometry']
                if geom['type'] == 'Polygon':
                    # Plot polygon outline
                    polygon = np.array(geom['coordinates'][0])
                    ax.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5, alpha=0.6)
                elif geom['type'] == 'MultiPolygon':
                    # Plot each polygon in the multipolygon
                    for poly in geom['coordinates']:
                        polygon = np.array(poly[0])
                        ax.plot(polygon[:, 0], polygon[:, 1], 'k-', linewidth=0.5, alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Building Density')
        
        # Set title and labels
        ax.set_title(f'Building Footprints - {area_name.capitalize()}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Add scale bar if style is paper
        if style == 'paper':
            # Approximate scale - this should be improved for actual geographic data
            scale_length = (max_x - min_x) * 0.1
            ax.plot(
                [min_x + x_range * 0.1, min_x + x_range * 0.1 + scale_length],
                [min_y + y_range * 0.1, min_y + y_range * 0.1],
                'k-', linewidth=2
            )
            ax.text(
                min_x + x_range * 0.1 + scale_length/2, 
                min_y + y_range * 0.08,
                f'{scale_length:.2f} deg',
                ha='center'
            )
        
        # Set aspect ratio to be equal
        ax.set_aspect('equal')
        
        # Save figure
        density_file = os.path.join(
            output_dir, 
            f'building_density_{area_name.lower().replace(" ", "_")}.{file_format}'
        )
        plt.tight_layout()
        plt.savefig(density_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        output_files.append(density_file)
        print(f"Saved {density_file}")
        
        # Create figure for building shape analysis
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
        # Calculate area and perimeter for each building
        areas = []
        perimeters = []
        aspect_ratios = []
        
        for building in buildings:
            if 'geometry' in building:
                geom = building['geometry']
                # Simple approximation - should use proper geodesic calculations for real work
                if geom['type'] == 'Polygon':
                    poly = np.array(geom['coordinates'][0])
                    # Calculate area using shoelace formula
                    x = poly[:, 0]
                    y = poly[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    
                    # Calculate perimeter
                    dx = np.diff(x, append=x[0])
                    dy = np.diff(y, append=y[0])
                    perimeter = np.sum(np.sqrt(dx*dx + dy*dy))
                    
                    # Calculate aspect ratio (using bounding box)
                    min_x, min_y = poly.min(axis=0)
                    max_x, max_y = poly.max(axis=0)
                    width = max_x - min_x
                    height = max_y - min_y
                    aspect = max(width/height if height > 0 else 1, height/width if width > 0 else 1)
                    
                    areas.append(area)
                    perimeters.append(perimeter)
                    aspect_ratios.append(aspect)
                    
                elif geom['type'] == 'MultiPolygon':
                    for poly_coords in geom['coordinates']:
                        poly = np.array(poly_coords[0])
                        # Calculate area using shoelace formula
                        x = poly[:, 0]
                        y = poly[:, 1]
                        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                        
                        # Calculate perimeter
                        dx = np.diff(x, append=x[0])
                        dy = np.diff(y, append=y[0])
                        perimeter = np.sum(np.sqrt(dx*dx + dy*dy))
                        
                        # Calculate aspect ratio (using bounding box)
                        min_x, min_y = poly.min(axis=0)
                        max_x, max_y = poly.max(axis=0)
                        width = max_x - min_x
                        height = max_y - min_y
                        aspect = max(width/height if height > 0 else 1, height/width if width > 0 else 1)
                        
                        areas.append(area)
                        perimeters.append(perimeter)
                        aspect_ratios.append(aspect)
        
        # Filter out extreme outliers
        if areas:
            # Remove top and bottom 1% to avoid outliers skewing the visualization
            areas = np.array(areas)
            perimeters = np.array(perimeters)
            aspect_ratios = np.array(aspect_ratios)
            
            min_area, max_area = np.percentile(areas, [1, 99])
            min_perim, max_perim = np.percentile(perimeters, [1, 99])
            min_aspect, max_aspect = np.percentile(aspect_ratios, [1, 99])
            
            mask = (
                (areas >= min_area) & (areas <= max_area) &
                (perimeters >= min_perim) & (perimeters <= max_perim) &
                (aspect_ratios >= min_aspect) & (aspect_ratios <= max_aspect)
            )
            
            areas = areas[mask]
            perimeters = perimeters[mask]
            aspect_ratios = aspect_ratios[mask]
            
            # Create hexbin plot for area vs perimeter
            hb = axs[0].hexbin(
                areas, perimeters, 
                gridsize=30, 
                cmap='viridis',
                mincnt=1
            )
            axs[0].set_title('Building Area vs Perimeter')
            axs[0].set_xlabel('Area (sq units)')
            axs[0].set_ylabel('Perimeter (units)')
            fig.colorbar(hb, ax=axs[0], label='Count')
            
            # Create histogram of aspect ratios
            axs[1].hist(aspect_ratios, bins=30, alpha=0.7, color='seagreen')
            axs[1].set_title('Building Aspect Ratio Distribution')
            axs[1].set_xlabel('Aspect Ratio')
            axs[1].set_ylabel('Frequency')
            
            # Add some statistics
            mean_aspect = np.mean(aspect_ratios)
            median_aspect = np.median(aspect_ratios)
            axs[1].axvline(mean_aspect, color='r', linestyle='--', label=f'Mean: {mean_aspect:.2f}')
            axs[1].axvline(median_aspect, color='b', linestyle=':', label=f'Median: {median_aspect:.2f}')
            axs[1].legend()
            
            plt.suptitle(f'Building Shape Analysis - {area_name.capitalize()}')
            plt.tight_layout()
            
            # Save figure
            shape_file = os.path.join(
                output_dir, 
                f'building_shape_analysis_{area_name.lower().replace(" ", "_")}.{file_format}'
            )
            plt.savefig(shape_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            output_files.append(shape_file)
            print(f"Saved {shape_file}")
    
    return output_files


def visualize_performance(
    performance_df: pd.DataFrame,
    output_dir: str,
    file_format: str = 'png',
    dpi: int = 300,
    style: str = 'paper'
) -> List[str]:
    """Visualize model performance metrics"""
    print("Generating performance visualizations...")
    
    if performance_df.empty:
        print("No performance data available")
        return []
    
    output_files = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style based on parameter
    if style == 'paper':
        sns.set_context("paper")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
        })
    elif style == 'presentation':
        sns.set_context("talk")
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
        })
    elif style == 'web':
        sns.set_context("notebook")
        sns.set_style("whitegrid")
    
    # Check available metrics
    metrics = [col for col in performance_df.columns if col not in ['region', 'model', 'date', 'timestamp']]
    
    if not metrics:
        print("No metrics found in performance data")
        return []
    
    # Check if we have model comparison data
    has_models = 'model' in performance_df.columns and len(performance_df['model'].unique()) > 1
    
    # Performance by region
    if 'region' in performance_df.columns:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            if metric in performance_df.columns:
                sns.barplot(x='region', y=metric, data=performance_df, ax=axes[i], 
                           palette='viridis', errwidth=1, capsize=0.1)
                axes[i].set_title(f'{metric.replace("_", " ").title()} by Region')
                axes[i].set_xlabel('Region')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        region_file = os.path.join(output_dir, f'performance_by_region.{file_format}')
        plt.savefig(region_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        output_files.append(region_file)
        print(f"Saved {region_file}")
    
    # Model comparison if available
    if has_models:
        # Create a figure for each metric
        for metric in metrics:
            if metric in performance_df.columns:
                plt.figure(figsize=(10, 6))
                
                if 'region' in performance_df.columns:
                    # Plot with regions
                    ax = sns.barplot(
                        x='model', y=metric, hue='region', data=performance_df,
                        palette='viridis', errwidth=1, capsize=0.1
                    )
                    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Plot without regions
                    ax = sns.barplot(
                        x='model', y=metric, data=performance_df,
                        palette='viridis', errwidth=1, capsize=0.1
                    )
                
                plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
                plt.xlabel('Model')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.tick_params(axis='x', rotation=45)
                
                # Add value labels on top of bars
                for p in ax.patches:
                    ax.annotate(
                        f'{p.get_height():.3f}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', rotation=0, xytext=(0, 5),
                        textcoords='offset points'
                    )
                
                plt.tight_layout()
                model_file = os.path.join(output_dir, f'model_comparison_{metric}.{file_format}')
                plt.savefig(model_file, dpi=dpi, bbox_inches='tight')
                plt.close()
                
                output_files.append(model_file)
                print(f"Saved {model_file}")
    
    # Create correlation matrix of metrics
    numeric_df = performance_df[metrics].select_dtypes(include=[np.number])
    
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        plt.figure(figsize=(8, 6))
        corr_matrix = numeric_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f", 
            cmap='coolwarm', square=True, linewidths=.5
        )
        
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        
        corr_file = os.path.join(output_dir, f'metrics_correlation.{file_format}')
        plt.savefig(corr_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        output_files.append(corr_file)
        print(f"Saved {corr_file}")
    
    # Create radar plot for multiple metrics comparison
    numeric_df = performance_df[metrics].select_dtypes(include=[np.number])
    
    if not numeric_df.empty and numeric_df.shape[1] >= 3:
        # Get models or regions for comparison
        categories = None
        if has_models:
            categories = performance_df['model'].unique()
            cat_col = 'model'
        elif 'region' in performance_df.columns:
            categories = performance_df['region'].unique()
            cat_col = 'region'
        
        if categories is not None and len(categories) <= 6:  # Limit to 6 for readability
            # Calculate mean values for each category and metric
            radar_data = {}
            for cat in categories:
                cat_data = performance_df[performance_df[cat_col] == cat]
                if not cat_data.empty:
                    radar_data[cat] = [cat_data[metric].mean() for metric in numeric_df.columns]
            
            # Create radar plot
            angles = np.linspace(0, 2*np.pi, len(numeric_df.columns), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Close the loop
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            
            for i, (cat, values) in enumerate(radar_data.items()):
                values = np.concatenate((values, [values[0]]))  # Close the loop
                ax.plot(angles, values, 'o-', linewidth=2, label=cat)
                ax.fill(angles, values, alpha=0.1)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([col.replace('_', ' ').title() for col in numeric_df.columns])
            
            ax.set_title(f'Performance Metrics Comparison')
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            radar_file = os.path.join(output_dir, f'radar_comparison.{file_format}')
            plt.savefig(radar_file, dpi=dpi, bbox_inches='tight')
            plt.close()
            
            output_files.append(radar_file)
            print(f"Saved {radar_file}")
    
    return output_files


def visualize_comparison(
    building_data: List[Dict], 
    performance_df: pd.DataFrame,
    output_dir: str,
    file_format: str = 'png',
    dpi: int = 300,
    style: str = 'paper'
) -> List[str]:
    """Visualize comparison between building characteristics and model performance"""
    print("Generating comparison visualizations...")
    
    output_files = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have the necessary data
    if not building_data or performance_df.empty:
        print("Not enough data for comparison visualizations")
        return []
    
    # Get regions from both datasets
    building_regions = set()
    for building in building_data:
        if 'properties' in building and 'region' in building['properties']:
            building_regions.add(building['properties']['region'])
    
    performance_regions = set()
    if 'region' in performance_df.columns:
        performance_regions = set(performance_df['region'].unique())
    
    # Find common regions
    common_regions = building_regions.intersection(performance_regions)
    
    if not common_regions:
        print("No common regions found for comparison")
        return []
    
    # Set style based on parameter
    if style == 'paper':
        sns.set_context("paper")
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
        })
    elif style == 'presentation':
        sns.set_context("talk")
    elif style == 'web':
        sns.set_context("notebook")
    
    # Prepare data for comparison
    comparison_data = []
    
    for region in common_regions:
        # Get building metrics for this region
        region_buildings = [b for b in building_data if 'properties' in b and 
                           b['properties'].get('region') == region]
        
        if not region_buildings:
            continue
            
        # Calculate building statistics
        building_count = len(region_buildings)
        
        # Calculate average area and perimeter
        areas = []
        perimeters = []
        
        for building in region_buildings:
            if 'geometry' in building:
                geom = building['geometry']
                if geom['type'] == 'Polygon':
                    poly = np.array(geom['coordinates'][0])
                    # Calculate area using shoelace formula
                    x = poly[:, 0]
                    y = poly[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    
                    # Calculate perimeter
                    dx = np.diff(x, append=x[0])
                    dy = np.diff(y, append=y[0])
                    perimeter = np.sum(np.sqrt(dx*dx + dy*dy))
                    
                    areas.append(area)
                    perimeters.append(perimeter)
        
        if not areas:
            continue
            
        avg_area = np.mean(areas)
        avg_perimeter = np.mean(perimeters)
        
        # Get performance metrics for this region
        region_perf = performance_df[performance_df['region'] == region]
        
        if region_perf.empty:
            continue
            
        # Get available performance metrics
        metrics = [col for col in region_perf.columns if col not in ['region', 'model', 'date', 'timestamp']]
        
        if not metrics:
            continue
            
        # Create entry for this region
        entry = {
            'region': region,
            'building_count': building_count,
            'avg_area': avg_area,
            'avg_perimeter': avg_perimeter
        }
        
        # Add performance metrics
        for metric in metrics:
            entry[metric] = region_perf[metric].mean()
        
        comparison_data.append(entry)
    
    if not comparison_data:
        print("No comparison data available")
        return []
        
    # Convert to DataFrame
    comp_df = pd.DataFrame(comparison_data)
    
    # Create scatter plots
    building_features = ['building_count', 'avg_area', 'avg_perimeter']
    perf_metrics = [col for col in comp_df.columns if col not in ['region'] + building_features]
    
    if not perf_metrics:
        print("No performance metrics available for comparison")
        return []
    
    # Create plots for each performance metric against building features
    for metric in perf_metrics:
        fig, axes = plt.subplots(1, len(building_features), figsize=(15, 5))
        
        if len(building_features) == 1:
            axes = [axes]
        
        for i, feature in enumerate(building_features):
            if feature in comp_df.columns and metric in comp_df.columns:
                # Create scatter plot
                sns.scatterplot(
                    x=feature, y=metric, data=comp_df, ax=axes[i], 
                    s=80, color='blue', alpha=0.7
                )
                
                # Add labels for each point
                for j, row in comp_df.iterrows():
                    axes[i].annotate(
                        row['region'], 
                        (row[feature], row[metric]),
                        xytext=(5, 5), textcoords='offset points'
                    )
                    
                # Add trend line
                sns.regplot(
                    x=feature, y=metric, data=comp_df, ax=axes[i],
                    scatter=False, ci=None, line_kws={'color': 'red', 'linestyle': '--'}
                )
                
                # Calculate correlation
                corr = comp_df[[feature, metric]].corr().iloc[0, 1]
                axes[i].set_title(f'{feature.replace("_", " ").title()} vs {metric.replace("_", " ").title()}\nCorrelation: {corr:.2f}')
                axes[i].set_xlabel(feature.replace('_', ' ').title())
                axes[i].set_ylabel(metric.replace('_', ' ').title())
        
        plt.suptitle(f'Building Characteristics vs {metric.replace("_", " ").title()} Performance')
        plt.tight_layout()
        
        comp_file = os.path.join(output_dir, f'comparison_{metric}.{file_format}')
        plt.savefig(comp_file, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        output_files.append(comp_file)
        print(f"Saved {comp_file}")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Get numeric columns
    numeric_cols = comp_df.select_dtypes(include=[np.number]).columns
    
    # Create correlation matrix
    corr_matrix = comp_df[numeric_cols].corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
               square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Between Building Characteristics and Performance Metrics')
    plt.tight_layout()
    
    corr_file = os.path.join(output_dir, f'correlation_heatmap.{file_format}')
    plt.savefig(corr_file, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    output_files.append(corr_file)
    print(f"Saved {corr_file}")
    
    return output_files


def main() -> int:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    building_data = []
    performance_df = pd.DataFrame()
    
    if args.type in ['buildings', 'all', 'comparison']:
        building_data = load_building_data(args.input, args.region)
        
    if args.type in ['performance', 'all', 'comparison']:
        performance_df = load_performance_data(args.input, args.region)
    
    # Generate visualizations
    all_outputs = []
    
    if args.type in ['buildings', 'all'] and building_data:
        building_outputs = visualize_buildings(
            building_data, 
            os.path.join(args.output, 'buildings'),
            args.region,
            args.format,
            args.dpi,
            args.style
        )
        all_outputs.extend(building_outputs)
    
    if args.type in ['performance', 'all'] and not performance_df.empty:
        performance_outputs = visualize_performance(
            performance_df,
            os.path.join(args.output, 'performance'),
            args.format,
            args.dpi,
            args.style
        )
        all_outputs.extend(performance_outputs)
    
    if args.type in ['comparison', 'all'] and building_data and not performance_df.empty:
        comparison_outputs = visualize_comparison(
            building_data,
            performance_df,
            os.path.join(args.output, 'comparison'),
            args.format,
            args.dpi,
            args.style
        )
        all_outputs.extend(comparison_outputs)
    
    # Generate HTML report
    if all_outputs:
        generate_report(args.output, all_outputs, args.format)
    
    print(f"Generated {len(all_outputs)} visualizations")
    return 0


def generate_report(output_dir: str, output_files: List[str], file_format: str) -> None:
    """Generate HTML report with all visualizations"""
    report_path = os.path.join(output_dir, 'visualization_report.html')
    
    with open(report_path, 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GeoAI Research - Visualization Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }
        .section {
            margin-top: 30px;
            border-top: 1px solid #eee;
            padding-top: 20px;
        }
        .visualization {
            margin-bottom: 40px;
        }
        .visualization img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .caption {
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <h1>GeoAI Research - Visualization Report</h1>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
''')
        
        # Organize files by section
        sections = {}
        for file in output_files:
            rel_path = os.path.relpath(file, output_dir)
            section_name = os.path.dirname(rel_path)
            if section_name == '.':
                section_name = 'general'
                
            if section_name not in sections:
                sections[section_name] = []
                
            sections[section_name].append(file)
        
        # Generate table of contents
        for section_name in sections:
            display_name = section_name.capitalize() if section_name != 'general' else 'General'
            f.write(f'            <li><a href="#{section_name}">{display_name} Visualizations</a></li>\n')
            
        f.write('''        </ul>
    </div>
''')

        # Generate sections with visualizations
        for section_name, files in sections.items():
            display_name = section_name.capitalize() if section_name != 'general' else 'General'
            f.write(f'''
    <div id="{section_name}" class="section">
        <h2>{display_name} Visualizations</h2>
''')
            
            for file in files:
                rel_path = os.path.relpath(file, output_dir)
                filename = os.path.basename(file)
                name_without_ext = os.path.splitext(filename)[0]
                title = name_without_ext.replace('_', ' ').title()
                
                f.write(f'''
        <div class="visualization">
            <h3>{title}</h3>
            <img src="{rel_path}" alt="{title}">
            <p class="caption">Figure: {title}</p>
        </div>
''')
                
            f.write('    </div>\n')
            
        f.write('''
    <footer>
        <p>Generated on ''' + f'{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}' + '''</p>
    </footer>
</body>
</html>''')
    
    print(f"Generated HTML report: {report_path}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)