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

The data is expected to come from Microsoft Building Footprints or similar rasterized building datasets.

**Note**: Due to size constraints, actual data files are not included in this repository. Download and extract the dataset locally to use the pipeline with real data.