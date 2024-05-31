import argparse
import math
import os
from collections import defaultdict
from pathlib import Path
from pdb import set_trace

import numpy as np
import pandas as pd
import torch
import torch as th
import trimesh
import yaml
from easydict import EasyDict
from kitsu.utils import instantiate_from_config
from plyfile import PlyData, PlyElement
from pykdtree.kdtree import KDTree
from tqdm import tqdm, trange

from lib.libmesh import check_mesh_contains

EMPTY_PCL_DICT = {
    "completeness": np.sqrt(3),
    "accuracy": np.sqrt(3),
    "completeness2": 3,
    "accuracy2": 3,
    "chamfer": 6,
}

EMPTY_PCL_DICT_NORMALS = {
    "normals completeness": -1.0,
    "normals accuracy": -1.0,
    "normals": -1.0,
}


def compute_iou(occ1, occ2):
    """Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    """
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.5

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = area_intersect / area_union

    return iou


def get_threshold_percentage(dist, thresholds):
    """Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [(dist <= t).mean() for t in thresholds]
    return in_threshold


def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([plydata["vertex"]["x"], plydata["vertex"]["y"], plydata["vertex"]["z"]], axis=1)
    return vertices


class MeshEvaluator(object):
    """Mesh evaluation class.

    It handles the mesh evaluation process.

    Args:
        n_points (int): number of points to be used for evaluation
    """

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def eval_mesh(self, mesh, pointcloud_tgt, normals_tgt, points_iou, occ_tgt, remove_wall=False):
        """Evaluates a mesh.

        Args:
            mesh (trimesh): mesh which should be evaluated
            pointcloud_tgt (numpy array): target point cloud
            normals_tgt (numpy array): target normals
            points_iou (numpy_array): points tensor for IoU evaluation
            occ_tgt (numpy_array): GT occupancy values for IoU points
        """
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            if remove_wall:  #! Remove walls and floors
                pointcloud, idx = mesh.sample(2 * self.n_points, return_index=True)
                eps = 0.007
                x_max, x_min = pointcloud_tgt[:, 0].max(), pointcloud_tgt[:, 0].min()
                y_max, y_min = pointcloud_tgt[:, 1].max(), pointcloud_tgt[:, 1].min()
                z_max, z_min = pointcloud_tgt[:, 2].max(), pointcloud_tgt[:, 2].min()

                # add small offsets
                x_max, x_min = x_max + eps, x_min - eps
                y_max, y_min = y_max + eps, y_min - eps
                z_max, z_min = z_max + eps, z_min - eps

                mask_x = (pointcloud[:, 0] <= x_max) & (pointcloud[:, 0] >= x_min)
                mask_y = pointcloud[:, 1] >= y_min  # floor
                mask_z = (pointcloud[:, 2] <= z_max) & (pointcloud[:, 2] >= z_min)

                mask = mask_x & mask_y & mask_z
                pointcloud_new = pointcloud[mask]
                # Subsample
                idx_new = np.random.randint(pointcloud_new.shape[0], size=self.n_points)
                pointcloud = pointcloud_new[idx_new]
                idx = idx[mask][idx_new]
            else:
                pointcloud, idx = mesh.sample(self.n_points, return_index=True)

            pointcloud = pointcloud.astype(np.float32)
            normals = mesh.face_normals[idx]
        else:
            pointcloud = np.empty((0, 3))
            normals = np.empty((0, 3))

        out_dict = self.eval_pointcloud(pointcloud, pointcloud_tgt, normals, normals_tgt)

        # DEV: is it differ to `iou(model(points_iou).sigmoid(), occ_tgt)`?
        if len(mesh.vertices) != 0 and len(mesh.faces) != 0:
            occ = check_mesh_contains(mesh, points_iou)
            out_dict["iou"] = compute_iou(occ, occ_tgt)
        else:
            out_dict["iou"] = 0.0

        return out_dict

    def eval_pointcloud(
        self, pointcloud, pointcloud_tgt, normals=None, normals_tgt=None, thresholds=np.linspace(1.0 / 1000, 1, 1000)
    ):
        """Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        """
        # Return maximum losses if pointcloud is empty
        if pointcloud.shape[0] == 0:
            print("Empty pointcloud / mesh detected!")
            out_dict = EMPTY_PCL_DICT.copy()
            if normals is not None and normals_tgt is not None:
                out_dict.update(EMPTY_PCL_DICT_NORMALS)
            return out_dict

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(pointcloud_tgt, normals_tgt, pointcloud, normals)
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(pointcloud, normals, pointcloud_tgt, normals_tgt)
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]

        out_dict = {
            "completeness": completeness,
            "accuracy": accuracy,
            "normals completeness": completeness_normals,
            "normals accuracy": accuracy_normals,
            "normals": normals_correctness,
            "completeness2": completeness2,
            "accuracy2": accuracy2,
            "chamfer-L2": chamferL2,
            "chamfer-L1": chamferL1,
            "f-score": F[9],  # threshold = 1.0%
            "f-score-15": F[14],  # threshold = 1.5%
            "f-score-20": F[19],  # threshold = 2.0%
        }

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array([np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def main():
    parser = argparse.ArgumentParser(description="MNIST toy experiment")
    parser.add_argument("config", type=str, help="Path to config file.")
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--fix_total", type=int, default=-1, help="debug option")
    opt = parser.parse_args()

    args_path = Path(opt.config)
    out_dir = args_path.parent / "output"
    if opt.out_dir is not None:
        out_dir = Path(opt.out_dir)
    eval_result_path = out_dir / "eval_result.csv"

    with open(args_path) as f:
        args = EasyDict(yaml.safe_load(f))
        cfg = args.cfg

    # load outputs that splited in multiple directories because of runit
    mesh_files = sorted(list(out_dir.rglob("**/meshes/**/*.ply")))
    mesh_files_map = {f"{f.parent.name}/{f.stem}": f for f in mesh_files}

    *_, dl_test = instantiate_from_config(args.dataset, cfg=args.cfg)
    ds = dl_test.dataset
    # assert len(mesh_files) == len(ds), f"mesh_files={len(mesh_files)} != testset={len(ds)}"
    if len(mesh_files) != len(ds):
        print(f"[WARN] the number of meshes are different with test dataset: mesh_files={len(mesh_files)}, testset={len(ds)}")

    evaluator = MeshEvaluator(n_points=100000)

    total = len(ds)
    if opt.fix_total > 0:
        total = opt.fix_total
    eval_dicts = []

    for i in trange(total, ncols=100):
        data = dl_test.collate_fn([ds[i]])
        idx = data["idx"].item()

        try:
            model_dict = ds.get_model_dict(idx)
        except AttributeError:
            model_dict = {"model": str(idx), "category": "n/a"}

        modelname = model_dict["model"]
        category_id = model_dict["category"]
        file_mapping_name = f"{category_id}/{modelname}"
        if file_mapping_name not in mesh_files_map:
            print(f"WARN: modelname({modelname}) not in mesh files")
            continue

        try:
            category_name = ds.metadata[category_id].get("name", "n/a")
            # for room dataset
            if category_name == "n/a":
                category_name = category_id
        except AttributeError:
            category_name = "n/a"

        # if category_id != "n/a":
        #     mesh_dir = os.path.join(mesh_dir, category_id)
        #     pointcloud_dir = os.path.join(pointcloud_dir, category_id)

        pointcloud_tgt = data["pointcloud_chamfer"].squeeze(0).numpy()
        normals_tgt = data["pointcloud_chamfer.normals"].squeeze(0).numpy()
        points_tgt = data["points_iou"].squeeze(0).numpy()
        occ_tgt = data["points_iou.occ"].squeeze(0).numpy()

        eval_dict = {
            "idx": idx,
            "class id": category_id,
            "class name": category_name,
            "modelname": modelname,
        }
        eval_dicts.append(eval_dict)

        # Evaluate mesh
        if cfg["test"]["eval_mesh"]:
            # mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)  # mc
            # mesh_file = os.path.join(mesh_dir, "%s.ply" % modelname)  # poco mc
            # mesh_file = str(mesh_files[i])
            mesh_file = mesh_files_map[file_mapping_name]

            if os.path.exists(mesh_file):
                try:
                    mesh = trimesh.load(mesh_file, process=False)
                    eval_dict_mesh = evaluator.eval_mesh(
                        mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt, remove_wall=cfg["test"]["remove_wall"]
                    )
                    for k, v in eval_dict_mesh.items():
                        eval_dict[k + " (mesh)"] = v
                except Exception as e:
                    print("Error: Could not evaluate mesh: %s" % mesh_file)
            else:
                print("Warning: mesh does not exist: %s" % mesh_file)

        # Evaluate point cloud (DEPRECATED)
        if cfg["test"]["eval_pointcloud"]:
            pointcloud_file = os.path.join(pointcloud_dir, "%s.ply" % modelname)

            if os.path.exists(pointcloud_file):
                pointcloud = load_pointcloud(pointcloud_file)
                eval_dict_pcl = evaluator.eval_pointcloud(pointcloud, pointcloud_tgt)
                for k, v in eval_dict_pcl.items():
                    eval_dict[k + " (pcl)"] = v
            else:
                print("Warning: pointcloud does not exist: %s" % pointcloud_file)

        # break  # DEBUG

    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(["idx"], inplace=True)
    eval_df.to_csv(str(eval_result_path))


if __name__ == "__main__":
    main()
