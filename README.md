# SKIRT_utils
Utility functions used for SKIRT. Including GIZMO snapshots reduction, SKIRT parameter file creation, and SKIRT output analysis.

## Repository Structure

```
SKIRT_utils/
├── src/
│   └── SKIRT_utils/
│       ├── data_reduction/   # Tools for reducing raw simulation snapshots (e.g., GIZMO)
│       │                     # and creating SKIRT parameter files
│       └── analysis/         # Tools for analyzing SKIRT output (SEDs, synthetic images, etc.)
└── pyproject.toml
```

## Installation

Install directly from the repository with pip:

```bash
pip install .
```

Or in editable/development mode:

```bash
pip install -e .
```

### Dependencies

JWST PSF functionality requires installation of the [stpsf](https://github.com/calebchoban/crc_scripts/) package.

Instrument plotting (**/analysis/inst_plots**) functionality requires installation of the [crc_scripts](https://stpsf.readthedocs.io/en/latest/index.html) repository.

## Packages

### `data_reduction`
Contains utilities for:
- Reading and processing GIZMO simulation snapshots
- Creating SKIRT parameter (`.ski`) files from simulation data

### `analysis`
Contains utilities for:
- Processing SKIRT output spectral energy distributions (SEDs)
- Analyzing synthetic images produced by SKIRT
