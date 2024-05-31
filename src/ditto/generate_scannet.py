import argparse
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import torch as th
import trimesh
import yaml
from easydict import EasyDict
from kitsu.utils import instantiate_from_config
from kitsu.utils.vis3d import save_point_cloud
from pytorch3d.ops import knn_points, sample_farthest_points
from tqdm import tqdm

from src.ditto.eval import distance_p2p, get_threshold_percentage
from src.ditto.generate import get_generator

th.set_grad_enabled(False)
th.cuda.set_device(0)


def point_interpolation(xyz, n, k=3):
    m = xyz.size(1)
    t = math.ceil(math.log(n / m, 2))

    for _ in range(t):
        knn = knn_points(xyz, xyz, K=k, return_nn=True)
        centroid = knn.knn.mean(-2)  # b n 3
        xyz = th.cat([xyz, centroid], 1)  # b 2n 3

    if xyz.size(1) > n:
        xyz, _ = sample_farthest_points(xyz, K=n, random_start_point=True)

    return xyz


def eval_mesh(mesh: trimesh.Trimesh, xyz_gt: np.ndarray, n_pts_eval: int, remove_wall=True):
    if remove_wall:
        xyz, idx = mesh.sample(2 * n_pts_eval, return_index=True)  # m 3, numpy
        xyz = xyz.astype(np.float32)
        eps = 0.007
        x_max, x_min = xyz_gt[:, 0].max() + eps, xyz_gt[:, 0].min() - eps
        y_max, y_min = xyz_gt[:, 1].max() + eps, xyz_gt[:, 1].min() - eps
        z_max, z_min = xyz_gt[:, 2].max() + eps, xyz_gt[:, 2].min() - eps

        mask_x = (xyz[:, 0] <= x_max) & (xyz[:, 0] >= x_min)
        mask_y = xyz[:, 1] >= y_min
        mask_z = (xyz[:, 2] <= z_max) & (xyz[:, 2] >= z_min)

        mask = mask_x & mask_y & mask_z
        xyz_new = xyz[mask]
        idx = np.random.permutation(xyz_new.shape[0])[:n_pts_eval]
        xyz = xyz_new[idx]  # m 3

    else:
        xyz, idx = mesh.sample(n_pts_eval, return_index=True)  # m 3, numpy
        xyz = xyz.astype(np.float32)

    dist1, _ = distance_p2p(xyz_gt, None, xyz, None)
    dist2, _ = distance_p2p(xyz, None, xyz_gt, None)
    chamfer_l1 = 100 * 0.5 * (dist1.mean() + dist2.mean())

    thresholds = np.linspace(1.0 / 1000, 1, 1000)
    recall = get_threshold_percentage(dist1, thresholds)
    precision = get_threshold_percentage(dist2, thresholds)
    f_scores = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(len(precision))]
    f_score = f_scores[9]  # threshold = 1.0%

    return chamfer_l1, f_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--n_pts", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--n_pts_eval", type=int, default=100000)
    parser.add_argument("--outdir", type=str, default="output_scannet")
    parser.add_argument("--idx", type=int, nargs=2, default=(0, -1))
    parser.add_argument("--point_interpolation", type=int, default=-1)
    opt = parser.parse_args()

    args_path = Path(opt.config_file)
    ckpt_path = sorted(list(args_path.parent.glob("best*.pth")))[-1]
    out_dir = args_path.parent / opt.outdir
    (out_dir / "meshes").mkdir(parents=True, exist_ok=True)

    print("args_path:", str(args_path))
    print("ckpt_path:", str(ckpt_path))
    print("out_dir:", str(out_dir))

    with open(args_path) as f:
        args = EasyDict(yaml.safe_load(f))

    model = instantiate_from_config(args.model).cuda().eval()
    ckpt = th.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    generator = get_generator(model, global_resolution=opt.resolution, cfg=args.cfg, device="cuda")

    data_dir = Path("../data/scannet/scannet_v2_alto/scans/scenes")
    data_files = sorted(list(data_dir.glob("scene*")))
    data_files_ = []
    for data_file in data_files:
        if (data_file / "pointcloud.npz").exists():
            data_files_.append(data_file)
        else:
            print("[WARN] File", data_file, "not exists!!")
    data_files = data_files_

    idx1 = max(0, opt.idx[0])
    idx2 = min(len(data_files), opt.idx[1] if opt.idx[1] > 0 else len(data_files))
    data_files = data_files[idx1:idx2]
    print(len(data_files))

    result_file = open(out_dir / "results.txt", "a")

    with tqdm(total=len(data_files), ncols=100, desc="Scannet Generation") as pbar:
        for data_file in data_files:
            pbar.set_postfix_str(data_file.name)
            out_path = out_dir / "meshes" / f"{data_file.name}.ply"
            if out_path.exists():
                pbar.update()
                continue

            xyz_gt = np.load(data_file / "pointcloud.npz")["points"].astype(np.float32)  # m 3, numpy
            xyz = th.from_numpy(xyz_gt).cuda()  # m 3
            xyz = xyz[None, th.randperm(xyz.size(0))[: opt.n_pts]]  # 1 n 3
            # xyz = xyz + th.randn_like(xyz) * 0.005

            # point interpolation
            if opt.point_interpolation > 0:
                xyz = point_interpolation(xyz, n=opt.point_interpolation, k=16)

            # generate mesh
            mesh, _ = generator.generate_mesh({"inputs": xyz})
            o3d.io.write_triangle_mesh(str(out_path), mesh)

            # eval results
            mesh = trimesh.load(str(out_path))

            msg = f"{data_file.name}"
            chamfer_l1, f_score = eval_mesh(mesh, xyz_gt, opt.n_pts_eval, remove_wall=False)
            msg += f"remove_wall=False: chamferL1[{chamfer_l1:.4f}] F1[{f_score:.4f}];"
            chamfer_l1, f_score = eval_mesh(mesh, xyz_gt, opt.n_pts_eval, remove_wall=True)
            msg += f"remove_wall=True: chamferL1[{chamfer_l1:.4f}] F1[{f_score:.4f}]"
            print(msg)
            result_file.write(msg + "\r\n")
            result_file.flush()
            save_point_cloud(str(out_path.parent / f"{data_file.name}_input.ply"), xyz_gt)

            pbar.update()

    result_file.close()


if __name__ == "__main__":
    main()
