# DITTO

Official Pytorch implementation of **DITTO: Dual and Integrated Latent Topologies for Implicit 3D Reconstruction (CVPR 2024)**

<h2>
<a href="https://arxiv.org/abs/2403.05005">Paper (arxiv)</a> |
<a href="https://vision3d-lab.github.io/ditto">Project Page</a>
</h2>

<!-- demo image -->

<!-- TODO ditto icon -->


<!-- video demo -->
<!-- [![results_video_in_3d](https://github.com/Kitsunetic/DITTO_CVPR24/releases/download/assets/video_thumbnail.png)](https://github.com/Kitsunetic/DITTO_CVPR24/releases/download/assets/qualitative_video_original.mp4) -->
![teaser](assets/qualitative_video_for_github.gif)


<!-- demo script (google colab?) -->


## Installation

### Requirements - Hardware-Dependent Packages

Please install the following packages as your own.

- [pytorch](https://pytorch.org/get-started/previous-versions/) (we have tested with version of 2.0.1 with CUDA toolkit 11.7)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- [xformers](https://github.com/facebookresearch/xformers?tab=readme-ov-file#installing-xformers)

### Requirements - Python Packages

- h5py
- einops
- scipy
- pykdtree
- kitsu==0.1.2

Please install following packages by running following command:
```sh
pip install h5py einops scipy pykdtree kitsu==0.1.2
```

`kitsu` is individually developed and used library, which is pack of boilerplate codes like PyTorchLightning and HuggingFace transformers developed by `Kitsunetic`.

### Requirements - Cython Packages

Please run the following command:

```sh
python setup.py build_ext --inplace
```



## Datasets

### ShapeNet and SyntheticRooms datasets

We use `ShappeNet` and `SyntheticRooms` datasets for training.

You can download the datasets from [ConvONet's github page](https://github.com/autonomousvision/convolutional_occupancy_networks).

Please download datasets and unzip them so that have directory structures like following:

```
+ DITTO_CVPR24
    + assets
    + src
    + main.py
+ data
    + occupancynetwork
        + ShapeNet
            + 02691156
            ...
            + metadata.yaml
    + synthetic_room_dataset
        + rooms_04
        ...
        + rooms_08
```

Or you can modify `*.yaml` config files to designate your own directory path.


### Scannet

We use Scannet only for evaluation only with trained models in SyntheticRooms dataset following the convention.

Please follow preprocessing preprocess of [ALTO's script](https://github.com/wzhen1/ALTO#scannet), then move the preprocessed dataset folders following:

```
+ DITTO_CVPR24
    + assets
    + src
    + main.py
+ data
    + occupancynetwork
    + synthetic_room_dataset
    + scannet
        + scannet_v2 (actually it is not required, just for example)
            + scans
                ...
        + scannet_v2_alto (this is the target folder generated by preprocessing script)
            + scans
                + scenes
                    ...
            + scenes
                - SensorData.py
```

Or you can modify `[src/generate_scannet.py](src/generate_scannet.py)` file to designate your own directory path.


<!-- ## Pretrained Models

You can download every pretrained checkpoint in [here](https://github.com/vision3d-lab/ditto/releases/download/checkpoints/ditto_checkpoints.zip).

Then unzip in the `results` folder so that the directory structure is following:

```
+ results
    + ditto
        + shapenet
            + 240301_0000_00
                - args.yaml
                - best.pth
            + 240301_0000_01.pts1000
                - args.yaml
                - best.pth
            + 240301_0000_02.pts300
                ...
        + synthetic_rooms_triplane
            + 240301_0000_00
                ...
            + 240301_0000_01.pts3000
                ...
            + 240301_0000_02.noisy
                ...
        + synthetic_rooms_voxel
            + 240301_0000_00
                ...
            + 240301_0000_01.sparse
                ...
``` -->



## Training & Evaluation

The training command has same structure regardless on the model and dataset:

```sh
python main.py {yaml config file path} --gpus {gpu ids separated by ','}
```

The yaml config file possess information required to instantiate model, optimizer, dataset, trainer, e.t.c.

Here are some examples of training commands:

```sh
python main.py src/ditto/config/shapenet/00.yaml --gpus 0
python main.py src/ditto/config/shapenet/01.pts1000.yaml --gpus 0,1,2,3,4,5,6,7
python main.py src/ditto/config/synthetic_rooms_triplane/01.pts3000.yaml --gpus 0,1,2,3
```

These commands will create directory named `./results/ditto/shapenet/{yaml_file_name}`, and store logs and checkpoints inside of the directory.



### Training on ShapeNet

```sh
# 3K points & 0.005 noise
python main.py src/ditto/config/shapenet/00.yaml --gpus {}

# 1K points & 0.005 noise
python main.py src/ditto/config/shapenet/01.pts1000.yaml --gpus {}

# 300 points & 0.005 noise
python main.py src/ditto/config/shapenet/02.pts300yaml --gpus {}
```

### Training on SyntheticRooms

```sh
# Triplane, 10K points & 0.005 noise
python main.py src/ditto/config/synthetic_rooms_triplane/00.yaml --gpus {}

# Triplane, 3K points & 0.005 noise
python main.py src/ditto/config/synthetic_rooms_triplane/01.pts3000.yaml --gpus {}

# Triplane, 10K points & 0.025 noise
python main.py src/ditto/config/synthetic_rooms_triplane/02.noisy.yaml --gpus {}

# Voxel, 10K points & 0.005 noise
python main.py src/ditto/config/synthetic_rooms_voxel/00.yaml --gpus {}

# Voxel, 3K points & 0.005 noise
python main.py src/ditto/config/synthetic_rooms_voxel/01.pts3000.yaml --gpus {}
```

### Evaluation

Generate results using `src/ditto/generate.py` and evaluate the generated meshes using `src/ditto/eval.py`:

```sh
PYTHONPATH=`pwd` python src/ditto/generate.py --resolution {128 for object, 256 for scene} {the_generated_experiments_directory_path}/args.yaml
PYTHONPATH=`pwd` python src/ditto/eval.py {the_generated_experiments_directory_path}/args.yaml
```

Here are a few examples:

```sh
# ShapeNet
PYTHONPATH=`pwd` python src/ditto/generate.py --resolution 128 results/ditto/shapenet/240301_0000_00/args.yaml
PYTHONPATH=`pwd` python src/ditto/eval.py results/ditto/shapenet/240301_0000_00/args.yaml

# SyntheticRooms
PYTHONPATH=`pwd` python src/ditto/generate.py --resolution 256 results/ditto/synthetic_rooms_triplane/240301_0000_00/args.yaml
PYTHONPATH=`pwd` python src/ditto/eval.py results/ditto/synthetic_rooms_triplane/240301_0000_00/args.yaml
```

<!-- ### ScanNet Evaluation

Generate results using `src/ditto/generate_scannet.py`:

```sh

``` -->


## Acknowledgements

<!-- Please reference this when you utilizing our work in academic research!!
```bib

``` -->

Please consider in mind to refer following works too.

- [ONet (CVPR 2019)](https://github.com/autonomousvision/occupancy_networks)
- [ConvONet (ECCV 2020)](https://github.com/autonomousvision/convolutional_occupancy_networks)
- [DPConvONet (WACV 2021)](https://github.com/dsvilarkovic/dynamic_plane_convolutional_onet)
- [POCO (CVPR 2022)](https://github.com/valeoai/POCO)
- [ARO-Net (CVPR 2023)](https://github.com/yizhiwang96/ARO-Net)
- [ALTO (CVPR 2023)](https://github.com/wzhen1/ALTO)
- [GeoUDF (ICCV 2023)](https://github.com/rsy6318/GeoUDF)


## Please Give us a Star

<!-- ![click_on_a_star](assets/click_on_a_star.webp) -->
<!-- <div align="center">
    <img src="assets/click_on_a_star2.webp" width="384px">
</div>

We are hungry... -->

<!-- Please click on a star. -->

<!-- Your finger movements will help humans discover light in a dark sky and imagine stars hidden by dust. -->

<b style="color: red;">Please give us a ⭐ star if you think our project helpful!!</b>

## BibTeX

```bib
@misc{shim2024ditto,
      title={DITTO: Dual and Integrated Latent Topologies for Implicit 3D Reconstruction}, 
      author={Jaehyeok Shim and Kyungdon Joo},
      year={2024},
      eprint={2403.05005},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
