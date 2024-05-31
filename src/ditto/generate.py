from pdb import set_trace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

np.set_printoptions(precision=4)
import argparse
import math
import os
import os.path as osp
import shutil
import time
from collections import defaultdict
from pathlib import Path
from pdb import set_trace as st

import open3d as o3d
import pandas as pd
import torch as th
import trimesh
import yaml
from easydict import EasyDict
from kitsu.utils import instantiate_from_config
from plyfile import PlyData, PlyElement
from scipy import ndimage
from skimage import measure
from skimage.measure import block_reduce
from tqdm import tqdm, trange

from lib.libmesh import check_mesh_contains

########## HARD CODED OPTIONS ##########
# GRID_RESOLUTION = 128  # 128 for object
# GRID_RESOLUTION = 256  # 256 for scene
########################################


class VoxelGrid:
    def __init__(self, data, loc=(0.0, 0.0, 0.0), scale=1):
        assert data.shape[0] == data.shape[1] == data.shape[2]
        data = np.asarray(data, dtype=np.bool_)
        loc = np.asarray(loc)
        self.data = data
        self.loc = loc
        self.scale = scale

    @classmethod
    def from_mesh(cls, mesh, resolution, loc=None, scale=None, method="ray"):
        bounds = mesh.bounds
        # Default location is center
        if loc is None:
            loc = (bounds[0] + bounds[1]) / 2

        # Default scale, scales the mesh to [-0.45, 0.45]^3
        if scale is None:
            scale = (bounds[1] - bounds[0]).max() / 0.9

        loc = np.asarray(loc)
        scale = float(scale)

        # Transform mesh
        mesh = mesh.copy()
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

        # Apply method
        if method == "ray":
            voxel_data = voxelize_ray(mesh, resolution)
        elif method == "fill":
            voxel_data = voxelize_fill(mesh, resolution)

        voxels = cls(voxel_data, loc, scale)
        return voxels

    def down_sample(self, factor=2):
        if not (self.resolution % factor) == 0:
            raise ValueError("Resolution must be divisible by factor.")
        new_data = block_reduce(self.data, (factor,) * 3, np.max)
        return VoxelGrid(new_data, self.loc, self.scale)

    def to_mesh(self):
        # Shorthand
        occ = self.data

        # Shape of voxel grid
        nx, ny, nz = occ.shape
        # Shape of corresponding occupancy grid
        grid_shape = (nx + 1, ny + 1, nz + 1)

        # Convert values to occupancies
        occ = np.pad(occ, 1, "constant")

        # Determine if face present
        f1_r = occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1]
        f2_r = occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1]
        f3_r = occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:]

        f1_l = ~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1]
        f2_l = ~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1]
        f3_l = ~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:]

        f1 = f1_r | f1_l
        f2 = f2_r | f2_l
        f3 = f3_r | f3_l

        assert f1.shape == (nx + 1, ny, nz)
        assert f2.shape == (nx, ny + 1, nz)
        assert f3.shape == (nx, ny, nz + 1)

        # Determine if vertex present
        v = np.full(grid_shape, False)

        v[:, :-1, :-1] |= f1
        v[:, :-1, 1:] |= f1
        v[:, 1:, :-1] |= f1
        v[:, 1:, 1:] |= f1

        v[:-1, :, :-1] |= f2
        v[:-1, :, 1:] |= f2
        v[1:, :, :-1] |= f2
        v[1:, :, 1:] |= f2

        v[:-1, :-1, :] |= f3
        v[:-1, 1:, :] |= f3
        v[1:, :-1, :] |= f3
        v[1:, 1:, :] |= f3

        # Calculate indices for vertices
        n_vertices = v.sum()
        v_idx = np.full(grid_shape, -1)
        v_idx[v] = np.arange(n_vertices)

        # Vertices
        v_x, v_y, v_z = np.where(v)
        v_x = v_x / nx - 0.5
        v_y = v_y / ny - 0.5
        v_z = v_z / nz - 0.5
        vertices = np.stack([v_x, v_y, v_z], axis=1)

        # Face indices
        f1_l_x, f1_l_y, f1_l_z = np.where(f1_l)
        f2_l_x, f2_l_y, f2_l_z = np.where(f2_l)
        f3_l_x, f3_l_y, f3_l_z = np.where(f3_l)

        f1_r_x, f1_r_y, f1_r_z = np.where(f1_r)
        f2_r_x, f2_r_y, f2_r_z = np.where(f2_r)
        f3_r_x, f3_r_y, f3_r_z = np.where(f3_r)

        faces_1_l = np.stack(
            [
                v_idx[f1_l_x, f1_l_y, f1_l_z],
                v_idx[f1_l_x, f1_l_y, f1_l_z + 1],
                v_idx[f1_l_x, f1_l_y + 1, f1_l_z + 1],
                v_idx[f1_l_x, f1_l_y + 1, f1_l_z],
            ],
            axis=1,
        )

        faces_1_r = np.stack(
            [
                v_idx[f1_r_x, f1_r_y, f1_r_z],
                v_idx[f1_r_x, f1_r_y + 1, f1_r_z],
                v_idx[f1_r_x, f1_r_y + 1, f1_r_z + 1],
                v_idx[f1_r_x, f1_r_y, f1_r_z + 1],
            ],
            axis=1,
        )

        faces_2_l = np.stack(
            [
                v_idx[f2_l_x, f2_l_y, f2_l_z],
                v_idx[f2_l_x + 1, f2_l_y, f2_l_z],
                v_idx[f2_l_x + 1, f2_l_y, f2_l_z + 1],
                v_idx[f2_l_x, f2_l_y, f2_l_z + 1],
            ],
            axis=1,
        )

        faces_2_r = np.stack(
            [
                v_idx[f2_r_x, f2_r_y, f2_r_z],
                v_idx[f2_r_x, f2_r_y, f2_r_z + 1],
                v_idx[f2_r_x + 1, f2_r_y, f2_r_z + 1],
                v_idx[f2_r_x + 1, f2_r_y, f2_r_z],
            ],
            axis=1,
        )

        faces_3_l = np.stack(
            [
                v_idx[f3_l_x, f3_l_y, f3_l_z],
                v_idx[f3_l_x, f3_l_y + 1, f3_l_z],
                v_idx[f3_l_x + 1, f3_l_y + 1, f3_l_z],
                v_idx[f3_l_x + 1, f3_l_y, f3_l_z],
            ],
            axis=1,
        )

        faces_3_r = np.stack(
            [
                v_idx[f3_r_x, f3_r_y, f3_r_z],
                v_idx[f3_r_x + 1, f3_r_y, f3_r_z],
                v_idx[f3_r_x + 1, f3_r_y + 1, f3_r_z],
                v_idx[f3_r_x, f3_r_y + 1, f3_r_z],
            ],
            axis=1,
        )

        faces = np.concatenate(
            [
                faces_1_l,
                faces_1_r,
                faces_2_l,
                faces_2_r,
                faces_3_l,
                faces_3_r,
            ],
            axis=0,
        )

        vertices = self.loc + self.scale * vertices
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        return mesh

    @property
    def resolution(self):
        assert self.data.shape[0] == self.data.shape[1] == self.data.shape[2]
        return self.data.shape[0]

    def contains(self, points):
        nx = self.resolution

        # Rescale bounding box to [-0.5, 0.5]^3
        points = (points - self.loc) / self.scale
        # Discretize points to [0, nx-1]^3
        points_i = ((points + 0.5) * nx).astype(np.int32)
        # i1, i2, i3 have sizes (batch_size, T)
        i1, i2, i3 = points_i[..., 0], points_i[..., 1], points_i[..., 2]
        # Only use indices inside bounding box
        mask = (i1 >= 0) & (i2 >= 0) & (i3 >= 0) & (nx > i1) & (nx > i2) & (nx > i3)
        # Prevent out of bounds error
        i1 = i1[mask]
        i2 = i2[mask]
        i3 = i3[mask]

        # Compute values, default value outside box is 0
        occ = np.zeros(points.shape[:-1], dtype=np.bool_)
        occ[mask] = self.data[i1, i2, i3]

        return occ


def voxelize_ray(mesh, resolution):
    occ_surface = voxelize_surface(mesh, resolution)
    # TODO: use surface voxels here?
    occ_interior = voxelize_interior(mesh, resolution)
    occ = occ_interior | occ_surface
    return occ


def voxelize_fill(mesh, resolution):
    bounds = mesh.bounds
    if (np.abs(bounds) >= 0.5).any():
        raise ValueError("voxelize fill is only supported if mesh is inside [-0.5, 0.5]^3/")

    occ = voxelize_surface(mesh, resolution)
    occ = ndimage.morphology.binary_fill_holes(occ)
    return occ


def voxelize_surface(mesh, resolution):
    vertices = mesh.vertices
    faces = mesh.faces

    vertices = (vertices + 0.5) * resolution

    face_loc = vertices[faces]
    occ = np.full((resolution,) * 3, 0, dtype=np.int32)
    face_loc = face_loc.astype(np.float32)

    voxelize_mesh_(occ, face_loc)
    occ = occ != 0

    return occ


def voxelize_interior(mesh, resolution):
    shape = (resolution,) * 3
    bb_min = (0.5,) * 3
    bb_max = (resolution - 0.5,) * 3
    # Create points. Add noise to break symmetry
    points = make_3d_grid(bb_min, bb_max, shape=shape).numpy()
    points = points + 0.1 * (np.random.rand(*points.shape) - 0.5)
    points = points / resolution - 0.5
    occ = check_mesh_contains(mesh, points)
    occ = occ.reshape(shape)
    return occ


def check_voxel_occupied(occupancy_grid):
    occ = occupancy_grid

    occupied = (
        occ[..., :-1, :-1, :-1]
        & occ[..., :-1, :-1, 1:]
        & occ[..., :-1, 1:, :-1]
        & occ[..., :-1, 1:, 1:]
        & occ[..., 1:, :-1, :-1]
        & occ[..., 1:, :-1, 1:]
        & occ[..., 1:, 1:, :-1]
        & occ[..., 1:, 1:, 1:]
    )
    return occupied


def check_voxel_unoccupied(occupancy_grid):
    occ = occupancy_grid

    unoccupied = ~(
        occ[..., :-1, :-1, :-1]
        | occ[..., :-1, :-1, 1:]
        | occ[..., :-1, 1:, :-1]
        | occ[..., :-1, 1:, 1:]
        | occ[..., 1:, :-1, :-1]
        | occ[..., 1:, :-1, 1:]
        | occ[..., 1:, 1:, :-1]
        | occ[..., 1:, 1:, 1:]
    )
    return unoccupied


def check_voxel_boundary(occupancy_grid):
    occupied = check_voxel_occupied(occupancy_grid)
    unoccupied = check_voxel_unoccupied(occupancy_grid)
    return ~occupied & ~unoccupied


# from src.utils import export_mesh, export_pointcloud, is_url, load_config, load_model_manual, load_url, mc_from_psr, scale2onet
def export_pointcloud(vertices, out_file, as_text=True):
    assert vertices.shape[1] == 3
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, "vertex")
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)


def make_3d_grid(bb_min, bb_max, shape):
    """Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    """
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def add_key(base, new, base_name, new_name, device=None):
    """Add new keys to the given input

    Args:
        base (tensor): inputs
        new (tensor): new info for the inputs
        base_name (str): name for the input
        new_name (str): name for the new info
        device (device): pytorch device
    """
    if (new is not None) and (isinstance(new, dict)):
        if device is not None:
            for key in new.keys():
                new[key] = new[key].to(device)
        base = {base_name: base, new_name: new}
    return base


class Generator3D(object):
    """Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    """

    def __init__(
        self,
        model,
        global_resolution,
        points_batch_size=100000,
        threshold=0.5,
        refinement_step=0,
        device=None,
        resolution0=16,
        upsampling_steps=3,
        with_normals=False,
        padding=0.1,
        sample=False,
        input_type=None,
        vol_info=None,
        vol_bound=None,
        simplify_nfaces=None,
    ):
        self.model = model.to(device)
        self.global_resolution = global_resolution
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces

        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_mesh(self, data, return_stats=True):
        """Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get("inputs", torch.empty(1, 0)).to(device)
        kwargs = {}

        t0 = time.time()

        # obtain features for all crops
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            c_plane = self.encode_crop(inputs, device)
            c_final = c_plane

        # input the entire volume
        else:
            inputs = add_key(inputs, data.get("inputs.ind"), "points", "index", device=device)
            t0 = time.time()
            with torch.no_grad():
                c_plane = self.model.encode(inputs)
                c_final = c_plane

        stats_dict["time (encode inputs)"] = time.time() - t0

        mesh = self.generate_from_latent(c_final, inputs, stats_dict=stats_dict, **kwargs)  ### add inputs for poco mc try

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, c_final, inputs, stats_dict={}, **kwargs):
        """Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
        inputs = inputs[0].cpu().numpy()  # .detach().cpu().numpy() ###
        # print('inputs shape======',inputs.shape) #(10000,3)
        threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            # input_points (10000,3)
            input_points = inputs
            bmin = input_points.min()
            bmax = input_points.max()

            ########################## hard-code paramters for now #########################
            step = None
            resolution = self.global_resolution
            padding = 1
            dilation_size = 2
            device = self.device
            num_pts = 50000
            out_value = 1
            mc_value = 0
            return_volume = False
            refine_iter = 10
            simplification_target = None
            refine_threshold = None
            ###############################################################################

            if step is None:
                step = (bmax - bmin) / (resolution - 1)  # 0.0039886895348044005
                resolutionX = resolution  # 256
                resolutionY = resolution  # 256
                resolutionZ = resolution  # 256
            else:
                bmin = input_points.min(axis=0)
                bmax = input_points.max(axis=0)
                resolutionX = math.ceil((bmax[0] - bmin[0]) / step)
                resolutionY = math.ceil((bmax[1] - bmin[1]) / step)
                resolutionZ = math.ceil((bmax[2] - bmin[2]) / step)

            bmin_pad = bmin - padding * step
            bmax_pad = bmax + padding * step

            pts_ids = (input_points - bmin) / step + padding
            pts_ids = pts_ids.astype(np.int32)  # (10000,3)

            # create the volume
            volume = np.full(
                (resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), np.nan, dtype=np.float64
            )
            mask_to_see = np.full(
                (resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), True, dtype=bool
            )
            while pts_ids.shape[0] > 0:
                # print("Pts", pts_ids.shape)

                # creat the mask
                mask = np.full(
                    (resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False, dtype=bool
                )
                mask[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = True

                # dilation
                for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
                    xc = int(pts_ids[i, 0])
                    yc = int(pts_ids[i, 1])
                    zc = int(pts_ids[i, 2])
                    mask[
                        max(0, xc - dilation_size) : xc + dilation_size,
                        max(0, yc - dilation_size) : yc + dilation_size,
                        max(0, zc - dilation_size) : zc + dilation_size,
                    ] = True

                # get the valid points
                valid_points_coord = np.argwhere(mask).astype(np.float32)
                valid_points = valid_points_coord * step + bmin_pad
                # print('valid_points===',valid_points.shape)

                # get the prediction for each valid points
                z = []
                near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
                for pnts in tqdm(torch.split(near_surface_samples_torch, num_pts, dim=0), ncols=100, disable=True):
                    ### our decoder
                    occ_hat = self.eval_points(pnts, c_final, **kwargs).cpu().numpy()
                    occ_hat_pos = torch.tensor(occ_hat)  # [0,1]
                    occ_hat_neg = occ_hat - 1  # [-1,0]
                    outputs = -(occ_hat_pos + occ_hat_neg)  # [-1,1]
                    z.append(outputs)

                z = np.concatenate(z, axis=0)
                z = z.astype(np.float64)

                # update the volume
                volume[mask] = z

                # create the masks
                mask_pos = np.full(
                    (resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False, dtype=bool
                )
                mask_neg = np.full(
                    (resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False, dtype=bool
                )

                # dilation
                for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
                    xc = int(pts_ids[i, 0])
                    yc = int(pts_ids[i, 1])
                    zc = int(pts_ids[i, 2])
                    mask_to_see[xc, yc, zc] = False
                    if volume[xc, yc, zc] <= 0:
                        mask_neg[
                            max(0, xc - dilation_size) : xc + dilation_size,
                            max(0, yc - dilation_size) : yc + dilation_size,
                            max(0, zc - dilation_size) : zc + dilation_size,
                        ] = True
                    if volume[xc, yc, zc] >= 0:
                        mask_pos[
                            max(0, xc - dilation_size) : xc + dilation_size,
                            max(0, yc - dilation_size) : yc + dilation_size,
                            max(0, zc - dilation_size) : zc + dilation_size,
                        ] = True

                # get the new points

                new_mask = (mask_neg & (volume >= 0) & mask_to_see) | (mask_pos & (volume <= 0) & mask_to_see)
                pts_ids = np.argwhere(new_mask).astype(np.int32)

            volume[0:padding, :, :] = out_value
            volume[-padding:, :, :] = out_value
            volume[:, 0:padding, :] = out_value
            volume[:, -padding:, :] = out_value
            volume[:, :, 0:padding] = out_value
            volume[:, :, -padding:] = out_value

            # volume[np.isnan(volume)] = out_value
            maxi = volume[~np.isnan(volume)].max()
            mini = volume[~np.isnan(volume)].min()

            if not (maxi > mc_value and mini < mc_value):
                return None

            if return_volume:
                return volume

            # compute the marching cubes
            verts, faces, _, _ = measure.marching_cubes(
                volume=volume.copy(),
                level=mc_value,
            )

            # removing the nan values in the vertices
            values = verts.sum(axis=1)
            o3d_verts = o3d.utility.Vector3dVector(verts)
            o3d_faces = o3d.utility.Vector3iVector(faces)
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
            mesh.remove_vertices_by_mask(np.isnan(values))
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            if refine_iter > 0:
                dirs = verts - np.floor(verts)
                dirs = (dirs > 0).astype(dirs.dtype)

                mask = np.logical_and(dirs.sum(axis=1) > 0, dirs.sum(axis=1) < 2)
                v = verts[mask]
                dirs = dirs[mask]

                # initialize the two values (the two vertices for mc grid)
                v1 = np.floor(v)
                v2 = v1 + dirs

                # get the predicted values for both set of points
                v1 = v1.astype(int)
                v2 = v2.astype(int)
                preds1 = volume[v1[:, 0], v1[:, 1], v1[:, 2]]
                preds2 = volume[v2[:, 0], v2[:, 1], v2[:, 2]]

                # get the coordinates in the real coordinate system
                v1 = v1.astype(np.float32) * step + bmin_pad
                v2 = v2.astype(np.float32) * step + bmin_pad

                # tmp mask
                mask_tmp = np.logical_and(np.logical_not(np.isnan(preds1)), np.logical_not(np.isnan(preds2)))
                v = v[mask_tmp]
                dirs = dirs[mask_tmp]
                v1 = v1[mask_tmp]
                v2 = v2[mask_tmp]
                mask[mask] = mask_tmp

                # initialize the vertices
                verts = verts * step + bmin_pad
                v = v * step + bmin_pad

                # iterate for the refinement step
                for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):
                    preds = []
                    pnts_all = torch.tensor(v, dtype=torch.float, device=device)
                    for pnts in tqdm(torch.split(pnts_all, num_pts, dim=0), ncols=100, disable=True):
                        occ_hat = self.eval_points(pnts, c_final, **kwargs).cpu().numpy()
                        occ_hat_pos = torch.tensor(occ_hat)  # [0,1]
                        occ_hat_neg = occ_hat - 1  # [-1,0]
                        outputs = -(occ_hat_pos + occ_hat_neg)  # [-1,1]
                        preds.append(outputs)

                    preds = np.concatenate(preds, axis=0)

                    mask1 = (preds * preds1) > 0
                    v1[mask1] = v[mask1]
                    preds1[mask1] = preds[mask1]

                    mask2 = (preds * preds2) > 0
                    v2[mask2] = v[mask2]
                    preds2[mask2] = preds[mask2]

                    v = (v2 + v1) / 2

                    verts[mask] = v

                    # keep only the points that needs to be refined
                    if refine_threshold is not None:
                        mask_vertices = np.linalg.norm(v2 - v1, axis=1) > refine_threshold
                        # print("V", mask_vertices.sum() , "/", v.shape[0])
                        v = v[mask_vertices]
                        preds1 = preds1[mask_vertices]
                        preds2 = preds2[mask_vertices]
                        v1 = v1[mask_vertices]
                        v2 = v2[mask_vertices]
                        mask[mask] = mask_vertices

                        if v.shape[0] == 0:
                            break
                        # print("V", v.shape[0])

            else:
                verts = verts * step + bmin_pad

            o3d_verts = o3d.utility.Vector3dVector(verts)
            o3d_faces = o3d.utility.Vector3iVector(faces)
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

            if simplification_target is not None and simplification_target > 0:
                mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)

            return mesh

    def eval_points(self, p, c_info=None, point_feature=None, n=None, N=None, vol_bound=None, **kwargs):  ### add n, N
        """Evaluates the occupancy values for the points.

        Args:
            p (tensor): points
            c (tensor): encoded feature volumes
        """
        # p_split = torch.split(p, self.points_batch_size)
        p_split = torch.split(p.cuda(non_blocking=True), self.points_batch_size)  # pre-uploading to gpu
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0)  # 1 m 3
            # chunk_size = 5000
            # pi_chunks = torch.split(pi, chunk_size, 1)
            with torch.no_grad():
                tri_feat, xyz, c = c_info
                p_r = self.model.decode(tri_feat, pi, xyz, c)
                occ_hat = p_r.sigmoid()
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        occ_hat = torch.cat(occ_hats, dim=0)
        occ_hat.squeeze_(-1)
        return occ_hat


def get_generator(model, global_resolution, cfg, device, **kwargs) -> Generator3D:
    """Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    """
    vol_bound = None
    vol_info = None

    generator = Generator3D(
        model,
        global_resolution=global_resolution,
        device=device,
        threshold=cfg["test"]["threshold"],
        resolution0=cfg["generation"]["resolution_0"],
        upsampling_steps=cfg["generation"]["upsampling_steps"],
        sample=cfg["generation"]["use_sampling"],
        refinement_step=cfg["generation"]["refinement_step"],
        simplify_nfaces=cfg["generation"]["simplify_nfaces"],
        input_type=cfg["data"]["input_type"],
        padding=cfg["data"]["padding"],
        vol_info=vol_info,
        vol_bound=vol_bound,
    )
    return generator


def main():
    parser = argparse.ArgumentParser(description="MNIST toy experiment")
    parser.add_argument("config", type=str, help="Path to config file.")
    # parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
    # parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    # parser.add_argument("--iter", type=int, metavar="S", help="the training iteration to be evaluated.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--resolution", type=int, default=256)

    opt = parser.parse_args()
    with open(opt.config) as f:
        args = EasyDict(yaml.safe_load(f))
        cfg = args.cfg

    vis_n_outputs = cfg["generation"]["vis_n_outputs"]
    if vis_n_outputs is None:
        vis_n_outputs = -1

    if opt.prefix:
        out_dir = Path(opt.config).parent / f"output/output{opt.prefix}"
    else:
        out_dir = Path(opt.config).parent / f"output/output0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = str(out_dir)

    generation_dir = osp.join(out_dir, cfg["generation"]["generation_dir"])
    out_time_file = osp.join(generation_dir, "time_generation_full.pkl")
    out_time_file_class = osp.join(generation_dir, "time_generation.pkl")

    # PYTORCH VERSION > 1.0.0
    assert float(torch.__version__.split(".")[-3]) > 0

    *_, test_loader = instantiate_from_config(args.dataset, cfg=args.cfg)
    dataset = test_loader.dataset
    print("Total testset length:", len(dataset))
    if opt.end == -1:
        opt.end = len(dataset)

    model = instantiate_from_config(args.model).cuda().eval()

    ckpt_path = sorted(list(Path(opt.config).parent.glob("best*.pth")))[-1]
    ckpt = th.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    generator = get_generator(model, opt.resolution, cfg, device="cuda")

    # Statistics
    time_dicts = []

    # Count how many models already created
    model_counter = defaultdict(int)
    generate_mesh = True

    print("Start generating ...")
    # for it, data in enumerate(tqdm(test_loader)):
    with tqdm(total=opt.end - opt.start, ncols=169, desc=f"Generation[{opt.start:04d}/{opt.end:04d}]") as pbar:
        for i in range(opt.start, opt.end):
            data = test_loader.collate_fn([dataset[i]])

            # Output folders
            mesh_dir = osp.join(generation_dir, "meshes")
            in_dir = osp.join(generation_dir, "input")
            pointcloud_dir = osp.join(generation_dir, "pointcloud")
            generation_vis_dir = osp.join(generation_dir, "vis")

            # Get index etc.
            idx = data["idx"].item()

            try:
                model_dict = dataset.get_model_dict(idx)
            except AttributeError:
                model_dict = {"model": str(idx), "category": "n/a"}

            modelname = model_dict["model"]
            category_id = model_dict["category"]

            try:
                category_name = dataset.metadata[category_id].get("name", "n/a")
            except AttributeError:
                category_name = "n/a"

            if category_id != "n/a":
                mesh_dir = osp.join(mesh_dir, str(category_id))
                pointcloud_dir = osp.join(pointcloud_dir, str(category_id))
                in_dir = osp.join(in_dir, str(category_id))

                folder_name = str(category_id)
                if category_name != "n/a":
                    folder_name = str(folder_name) + "_" + category_name.split(",")[0]

                generation_vis_dir = osp.join(generation_vis_dir, folder_name)

            # Create directories if necessary
            if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
                os.makedirs(generation_vis_dir)

            if generate_mesh and not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)

            # if generate_pointcloud and not os.path.exists(pointcloud_dir):
            #     os.makedirs(pointcloud_dir)

            if not os.path.exists(in_dir):
                os.makedirs(in_dir)

            # Timing dict
            time_dict = {
                "idx": idx,
                "class id": category_id,
                "class name": category_name,
                "modelname": modelname,
            }
            time_dicts.append(time_dict)

            # Generate outputs
            out_file_dict = {}

            if generate_mesh:
                t0 = time.time()
                out = generator.generate_mesh(data)
                time_dict["mesh"] = time.time() - t0

                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}
                time_dict.update(stats_dict)

            if mesh is not None:
                vertices = np.asarray(mesh.vertices)
                vertices = vertices / 1  # scale
                vertices = o3d.utility.Vector3dVector(vertices)
                mesh.vertices = vertices

                mesh_out_path = os.path.join(mesh_dir, modelname)
                # print(mesh_out_path)
                pbar.set_postfix_str("/".join(Path(mesh_out_path).parts[-3:]))

                if os.path.splitext(modelname)[1] == ".ply":
                    o3d.io.write_triangle_mesh(mesh_out_path, mesh)
                else:
                    o3d.io.write_triangle_mesh(mesh_out_path + ".ply", mesh)

            # if generate_pointcloud:
            #     pointcloud_out_file = osp.join(pointcloud_dir, "%s.ply" % modelname)
            #     export_pointcloud(pointcloud_out_file, scale2onet(points), normals)
            #     out_file_dict["pointcloud"] = pointcloud_out_file

            input_type = cfg["data"]["input_type"]
            if cfg["generation"]["copy_input"]:
                # Save inputs
                if input_type == "voxels":
                    inputs_path = os.path.join(in_dir, "%s.off" % modelname)
                    inputs = data["inputs"].squeeze(0).cpu()
                    voxel_mesh = VoxelGrid(inputs).to_mesh()
                    voxel_mesh.export(inputs_path)
                    out_file_dict["in"] = inputs_path
                elif input_type == "pointcloud_crop":
                    inputs_path = os.path.join(in_dir, "%s.ply" % modelname)
                    inputs = data["inputs"].squeeze(0).cpu().numpy()
                    export_pointcloud(inputs, inputs_path, False)
                    out_file_dict["in"] = inputs_path
                elif input_type == "pointcloud" or "partial_pointcloud":
                    inputs_path = os.path.join(in_dir, "%s.ply" % modelname)
                    inputs = data["inputs"].squeeze(0).cpu().numpy()
                    export_pointcloud(inputs, inputs_path, False)
                    out_file_dict["in"] = inputs_path

            # Copy to visualization directory for first vis_n_output samples
            c_it = model_counter[category_id]
            if c_it < vis_n_outputs:
                # Save output files
                img_name = "%02d.off" % c_it
                for k, filepath in out_file_dict.items():
                    ext = os.path.splitext(filepath)[1]
                    out_file = os.path.join(generation_vis_dir, "%02d_%s%s" % (c_it, k, ext))
                    shutil.copyfile(filepath, out_file)

            model_counter[category_id] += 1
            pbar.update()

    # Create pandas dataframe and save
    """
    {'idx': 0, 'class id': '04379243', 'class name': 'table', 'modelname': 'cd4e8748514e028642d23b95defe1ce5', 'mesh': 10.209066152572632, 'time (encode inputs)': 2.1451504230499268}, 
    """
    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(["idx"], inplace=True)
    time_df.to_pickle(out_time_file)

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=["class name"]).mesh.mean()
    time_df_class.to_pickle(out_time_file_class)

    # Print results
    time_df_class.loc["mean"] = time_df_class.mean()
    print("Timings [s]:")
    print(time_df_class)

    # save time_df
    time_df.to_csv((Path(out_dir) / "time_df.csv"), index=False)


if __name__ == "__main__":
    main()
