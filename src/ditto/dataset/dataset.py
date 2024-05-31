import os

import numpy as np
import torch as th
import torch.distributed as dist
from kitsu.utils.data import build_dataloaders
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from src.ditto.dataset.core import Shapes3dDataset
from src.ditto.dataset.fields import (
    IndexField,
    PartialPointCloudField,
    PatchPointCloudField,
    PatchPointsField,
    PointCloudField,
    PointsField,
    VoxelsField,
)
from src.ditto.dataset.transformer import PointcloudNoise, SubsamplePointcloud, SubsamplePoints


def get_data_fields(mode, cfg):
    """Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    """
    points_transform = SubsamplePoints(cfg["data"]["points_subsample"])

    input_type = cfg["data"]["input_type"]
    fields = {}
    if cfg["data"]["points_file"] is not None:
        if input_type != "pointcloud_crop":
            fields["points"] = PointsField(
                cfg["data"]["points_file"],
                points_transform,
                unpackbits=cfg["data"]["points_unpackbits"],
                multi_files=cfg["data"]["multi_files"],
            )
        else:
            fields["points"] = PatchPointsField(
                cfg["data"]["points_file"],
                transform=points_transform,
                unpackbits=cfg["data"]["points_unpackbits"],
                multi_files=cfg["data"]["multi_files"],
            )

    if mode in ("val", "test"):
        points_iou_file = cfg["data"]["points_iou_file"]
        voxels_file = cfg["data"]["voxels_file"]
        if points_iou_file is not None:
            if input_type == "pointcloud_crop":
                fields["points_iou"] = PatchPointsField(
                    points_iou_file, unpackbits=cfg["data"]["points_unpackbits"], multi_files=cfg["data"]["multi_files"]
                )
            else:
                fields["points_iou"] = PointsField(
                    points_iou_file, unpackbits=cfg["data"]["points_unpackbits"], multi_files=cfg["data"]["multi_files"]
                )
        if voxels_file is not None:
            fields["voxels"] = VoxelsField(voxels_file)

    return fields


def get_inputs_field(mode, cfg):
    """Returns the inputs fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): config dictionary
    """
    input_type = cfg["data"]["input_type"]

    if input_type is None:
        inputs_field = None
    elif input_type == "pointcloud":
        transform = transforms.Compose(
            [SubsamplePointcloud(cfg["data"]["pointcloud_n"]), PointcloudNoise(cfg["data"]["pointcloud_noise"])]
        )
        inputs_field = PointCloudField(cfg["data"]["pointcloud_file"], transform, multi_files=cfg["data"]["multi_files"])
    elif input_type == "partial_pointcloud":
        transform = transforms.Compose(
            [SubsamplePointcloud(cfg["data"]["pointcloud_n"]), PointcloudNoise(cfg["data"]["pointcloud_noise"])]
        )
        inputs_field = PartialPointCloudField(cfg["data"]["pointcloud_file"], transform, multi_files=cfg["data"]["multi_files"])
    elif input_type == "pointcloud_crop":
        transform = transforms.Compose(
            [SubsamplePointcloud(cfg["data"]["pointcloud_n"]), PointcloudNoise(cfg["data"]["pointcloud_noise"])]
        )

        inputs_field = PatchPointCloudField(
            cfg["data"]["pointcloud_file"],
            transform,
            multi_files=cfg["data"]["multi_files"],
        )

    elif input_type == "voxels":
        inputs_field = VoxelsField(cfg["data"]["voxels_file"])
    elif input_type == "idx":
        inputs_field = IndexField()
    else:
        raise ValueError("Invalid input type (%s)" % input_type)
    return inputs_field


def get_dataset(mode, cfg, return_idx=False):
    """Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    """
    dataset_type = cfg["data"]["dataset"]
    dataset_folder = cfg["data"]["path"]
    categories = cfg["data"]["classes"]

    # Get split
    splits = {
        "train": cfg["data"]["train_split"],
        "val": cfg["data"]["val_split"],
        "test": cfg["data"]["test_split"],
    }

    split = splits[mode]

    # Create dataset
    if dataset_type == "Shapes3D":
        # Dataset fields
        # Method specific fields (usually correspond to output)
        fields = get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(mode, cfg)
        if inputs_field is not None:
            fields["inputs"] = inputs_field

        if return_idx:
            fields["idx"] = IndexField()

        # field for evaluation
        if mode == "test":
            fields["pointcloud_chamfer"] = PointCloudField(
                cfg["data"]["pointcloud_chamfer_file"], multi_files=cfg["data"]["multi_files"]
            )

        dataset = Shapes3dDataset(dataset_folder, fields, split=split, categories=categories, cfg=cfg)
    else:
        raise ValueError('Invalid dataset "%s"' % cfg["data"]["dataset"])

    return dataset


def collate_remove_none(batch):
    """Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    """

    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def worker_init_fn(worker_id):
    """Worker init function to ensure true randomness."""

    def set_num_threads(nt):
        try:
            import mkl

            mkl.set_num_threads(nt)
        except:
            pass
            th.set_num_threads(1)
            os.environ["IPC_ENABLE"] = "1"
            for o in ["OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)


def load_dataloaders(batch_size, num_workers, cfg):
    if dist.is_initialized():
        world_size = dist.get_world_size()
        batch_size = max(1, batch_size // world_size)
        num_workers = min(num_workers, max(1, num_workers // world_size))

    ds_train = get_dataset("train", cfg)
    ds_valid = get_dataset("val", cfg)
    ds_test = get_dataset("test", cfg, return_idx=True)

    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_remove_none,
        worker_init_fn=worker_init_fn,
    )
    dl_train = build_dataloaders(ds=ds_train, shuffle=True, **dl_kwargs)
    dl_valid = build_dataloaders(ds=ds_valid, shuffle=False, **dl_kwargs)
    dl_test = build_dataloaders(batch_size=1, num_workers=1, ds=ds_test, shuffle=False)
    return dl_train, dl_valid, dl_test
