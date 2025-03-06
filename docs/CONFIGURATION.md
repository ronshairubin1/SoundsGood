# SoundClassifier Configuration Guide

## Configuration Structure

The SoundClassifier application uses a centralized configuration system to manage all settings, paths, and parameters. This document explains the organization and usage of the configuration system.

### Main Configuration File

The main configuration is defined in `config.py` at the root of the project. This file contains the `Config` class which centralizes all settings:

```python
class Config:
    # App directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DICTIONARIES_DIR = os.path.join(DATA_DIR, 'dictionaries')
    ANALYSIS_DIR = os.path.join(DATA_DIR, 'analysis')
    # ... other directory settings
    
    # Methods for working with dictionaries, model paths, etc.
    @classmethod
    def get_dictionary(cls):
        # ...
```

### Directory Structure

All data is organized in a clean, hierarchical structure with standardized path references:

```
SoundClassifier_v09/
├── config.py                 # Main configuration file
├── data/                     # All application data stored here
│   ├── analysis/             # Model analysis results (Config.ANALYSIS_DIR)
│   ├── dictionaries/         # Dictionary metadata (Config.DICTIONARIES_DIR)
│   │   └── dictionaries.json # Consolidated dictionary data
│   ├── models/               # Trained ML models (Config.MODELS_DIR)
│   └── sounds/               # Sound files
│       ├── raw_sounds/       # (Config.RAW_SOUNDS_DIR)
│       ├── training_sounds/  # (Config.TRAINING_SOUNDS_DIR)
│       ├── test_sounds/      # (Config.TEST_SOUNDS_DIR)
│       └── ...
├── legacy/                   # Older configuration files preserved here
│   └── config/
└── src/                      # Application source code
```

### Dictionary Management

Dictionaries are managed through a single consolidated file at `data/dictionaries/dictionaries.json`. This file stores:

1. All dictionaries with their metadata
2. The currently active dictionary

The format is:

```json
{
  "dictionaries": {
    "dictionary_name1": {
      "name": "dictionary_name1",
      "sounds": ["ah", "ee", "oh"],
      "classes": ["ah", "ee", "oh"],
      "created_at": "2023-01-01T12:00:00",
      "updated_at": "2023-01-01T12:00:00",
      "created_by": "admin",
      "description": "A dictionary description",
      "sample_count": 150
    },
    "dictionary_name2": {
      // ...
    }
  },
  "active_dictionary": "dictionary_name1"
}
```

### Accessing Configuration

To use the configuration in your code, import the `Config` class and access paths directly:

```python
from config import Config

# Get the active dictionary
active_dict = Config.get_dictionary()

# Use directory paths - ALWAYS access via Config constants
analysis_path = os.path.join(Config.ANALYSIS_DIR, 'my_analysis.json')
dictionary_path = os.path.join(Config.DICTIONARIES_DIR, 'my_file.json')
model_path = os.path.join(Config.MODELS_DIR, 'my_model.h5')

# Set the active dictionary
Config.set_active_dictionary('my_dictionary')
```

## Standardized Path References

The application uses consistent path references throughout the codebase:

| Data Type | Path Reference | Physical Location |
|-----------|---------------|-------------------|
| Dictionary data | `Config.DICTIONARIES_DIR` | `data/dictionaries/` |
| Active dictionary | `Config.get_dictionary()` | From `data/dictionaries/dictionaries.json` |
| Analysis files | `Config.ANALYSIS_DIR` | `data/analysis/` |
| Model files | `Config.MODELS_DIR` | `data/models/` |
| Training sounds | `Config.TRAINING_SOUNDS_DIR` | `data/sounds/training_sounds/` |
| Raw sounds | `Config.RAW_SOUNDS_DIR` | `data/sounds/raw_sounds/` |
| Temporary files | `Config.TEMP_DIR` | `temp/` |

## Configuration Migration

If you have an older version of the application with configs in multiple locations, you can run the migration script:

```bash
python migrate_config.py
```

This will:
1. Move dictionary data to `data/dictionaries/dictionaries.json`
2. Move analysis files to `data/analysis/`
3. Preserve legacy copies in `legacy/config/`

## Best Practices

1. **Always use the Config class constants** to access configuration data - never hardcode paths
2. **Use a single reference for each path** - for example, always use `Config.ANALYSIS_DIR` instead of constructing paths like `os.path.join(Config.DATA_DIR, 'analysis')`
3. When adding new settings, add them to the main `Config` class
4. Keep all configuration data in the consolidated locations 