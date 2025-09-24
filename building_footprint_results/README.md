# Building Footprint Data Directory

This directory is designed to contain rasterized building footprint data for USA states.

## Expected Structure

```
data/
├── Alabama/
│   ├── Alabama_avg.tif
│   ├── Alabama_cnt.tif
│   └── ...
├── Arizona/
├── ...
└── README.md
```

## Data Source

### Microsoft Building Footprints Dataset

This directory is designed to contain rasterized data from the **Microsoft Building Footprints** dataset:

- **Official Repository**: https://github.com/Microsoft/USBuildingFootprints  
- **Direct Downloads**: https://usbuildingdata.blob.core.windows.net/usbuildings-v2/
- **License**: Open Data Commons Open Database License (ODbL)
- **Coverage**: 130+ million building footprints across all US states

### Download Instructions

1. **Download individual state files** from the Microsoft repository
2. **Convert GeoJSON to raster format** using the preprocessing scripts (see `src/data_handler.py`)
3. **Extract statistical layers** (avg, cnt, sum, max) for efficient processing

### File Format

Each state should contain the following rasterized layers:
- `{State}_avg.tif`: Average building density per pixel
- `{State}_cnt.tif`: Count of buildings per pixel  
- `{State}_sum.tif`: Sum of building areas per pixel
- `{State}_max.tif`: Maximum building density per pixel

**Note**: Due to size constraints (~3GB), actual data files are not included in this repository. Download from Microsoft's official repository and run preprocessing locally.