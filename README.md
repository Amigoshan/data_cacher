# Loading Multi-Modal Sequencial Data
A general and efficient dataloader for TartanAir-style datasets

## Basic Ideas
add a gif


## Load Data From TartanAir


## Features and Assumptions 
- Data is organized in trajectories
- Each trajectory contrains multiple modalities, which are organized in different folders
- Different modalities are temporally aligned. It is only allowed that 
Different modalities 
Easy to extend to 
The data_cacher allows you to load from multiple datasets and mix them during training


## Data spec file
```
    1:
      file: "/home/amigo/workspace/pytorch/geometry_vision/data/tartan_train_local.txt"
      modality: 
        "depth_left":
          "depth":
            cacher_size: [160, 120]
            length: 1
        "flow":
          "flow":
            cacher_size: [320, 240]
            length: 1
          "mask":
            cacher_size: [160, 120]
            length: 1
      cacher:
        data_root_key: tartan 
        subset_framenum: 200
        worker_num: 0
      dataset:
      parameter:
        intrinsics: [640, 480, 320, 320, 320, 240]
        intrinsics_scale: [0.25, 0.25]
```

## Extend to a New Dataset
Using KITTI dataset as an example. 