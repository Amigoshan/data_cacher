# Data Cacher Configuration Schema

This document describes the YAML configuration format for Data Cacher. Configurations define how to load multi-modal sequential datasets, with global defaults and per-dataset overrides.

## Top-Level Structure

```yaml
task: <string>  # Required. Task name, e.g., "flowvo", "stereo"
global:         # Optional. Global defaults for all datasets
  modality: {...}
  cacher: {...}
  dataset: {...}
  parameter: {...}
data:           # Required. Per-dataset configurations
  "1": {...}    # Dataset ID (string or int)
  "2": {...}
```

## Sections

### Global
Applies defaults to all `data` entries. Can be overridden per-dataset.

### Modality
Defines how to load data modalities (e.g., images, flows).

- **cacher_size**: `[height, width]` (list of 2 ints). Size to cache/resample to.
- **length**: Sequence length (int >=1). Number of frames to load.

Example:
```yaml
modality:
  image_lcam_front:
    img0:
      cacher_size: [640, 640]
      length: 2
```

### Cacher
Controls RAM caching behavior.

- **data_root_key**: Key for data root path (string). Matches keys in `data_roots.py`.
- **data_root_path_override**: Override path (string, optional).
- **subset_framenum**: Frames to cache per subset (int >=1).
- **worker_num**: Threads for loading (int >=0).
- **load_traj**: Load entire trajectories? (bool).

Example:
```yaml
cacher:
  data_root_key: tartan2
  subset_framenum: 200
  worker_num: 2
  load_traj: false
```

### Dataset
Sampling parameters.

- **frame_skip**: Frames to skip between sequence items (int >=0).
- **seq_stride**: Frames to skip between sequences (int >=1).
- **frame_dir**: Use frame directories? (bool).

Example:
```yaml
dataset:
  frame_skip: 0
  seq_stride: 1
  frame_dir: true
```

### Parameter
Dataset-specific params (e.g., camera intrinsics).

- **intrinsics**: `[w, h, fx, fy, ox, oy]` (list of 6 numbers).
- **intrinsics_scale**: `[scale_x, scale_y]` (list of 2 numbers).
- **fxbl**: Focal length * baseline (number).
- **input_size**: `[h, w]` (list of 2 ints).
- **cam_model_for_flow**: Camera model for flow (list of 6 numbers).
- **cam_model_for_intrinsics_layer**: Camera model for intrinsics (list of 6 numbers).

Example:
```yaml
parameter:
  intrinsics: [640, 640, 320, 320, 320, 320]
  intrinsics_scale: [0.25, 0.25]
```

## Data Section
Each dataset has:
- **file**: Path to datafile (string, required).
- **modality**: Modality configs (dict, required).
- **cacher**: Cacher config (dict, required).
- **dataset**: Dataset config (dict, optional, uses global).
- **parameter**: Parameter config (dict, optional, uses global).

## Validation
Configs are validated on load. Errors include field paths for easy fixing.

## Example
See `dataspec/sample_tartanair_random.yaml` for a full example.

## CLI Validation
Run `python -c "from ConfigParser import ConfigParser; ConfigParser().validate_config_file('path/to/config.yaml')"` to check validity.</content>
<parameter name="filePath">/home/wenshan/workspace/pytorch/data_cacher/config_schema.md