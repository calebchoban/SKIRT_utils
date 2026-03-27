# SKIRT_utils
Utility functions used for SKIRT. Including GIZMO snapshots reduction, SKIRT parameter file creation, and SKIRT output analysis.

## Repository Structure

```
SKIRT_utils/
├── data_reduction/   # Tools for reducing raw simulation snapshots (e.g., GIZMO)
│                     # and creating SKIRT parameter files
└── analysis/         # Tools for analysing SKIRT output (SEDs, synthetic images, etc.)
```

## Packages

### `data_reduction`
Contains utilities for:
- Reading and processing GIZMO simulation snapshots
- Creating SKIRT parameter (`.ski`) files from simulation data

### `analysis`
Contains utilities for:
- Processing SKIRT output spectral energy distributions (SEDs)
- Analyzing synthetic images produced by SKIRT
