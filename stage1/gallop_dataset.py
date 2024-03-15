import os
from pathlib import Path
import sys
from itertools import permutations  # , combinations
import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d
from utils import farthest_point_sample, square_distance, normalize_area_scale

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # add the path to the DiffusionNet src
import diffusion_net  # noqa


class GallopDataset(Dataset):
    def __init__(self, root_dir, name="horse", train=True, k_eig=128, n_fmap=30, use_cache=True, op_cache_dir=None):

        self.train = train  # bool
        self.k_eig = k_eig
        self.n_fmap = n_fmap
        self.root_dir = root_dir
        self.cache_dir = os.path.join(root_dir, name, "cache")
        self.op_cache_dir = op_cache_dir

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.vts_list = []
        self.names_list = []
        self.sample_list = []

        # set combinations
        n_total = 49
        if self.train:
            self.combinations = list(permutations(range(n_total), 2))
        else:
            self.combinations = list(permutations(range(n_total), 2))

        # check the cache
        if use_cache:
            train_cache = os.path.join(self.cache_dir, "train.pt")
            load_cache = train_cache
            print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache):
                print("  --> loading dataset from cache")
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.vts_list,
                    self.names_list,
                    self.sample_list
                ) = torch.load(load_cache)
                return
            print("  --> dataset not in cache, repopulating")

        # Load the meshes & labels

        # Get all the files
        mesh_files = []

        # load faust data
        mesh_dirpath = (Path(self.root_dir)).resolve()
        for fname in mesh_dirpath.iterdir():
            if fname.suffix != ".off":
                continue
            mesh_fullpath = os.path.join(mesh_dirpath, fname)
            mesh_files.append(mesh_fullpath)

        print("loading {} meshes".format(len(mesh_files)))

        mesh_files = sorted(mesh_files)

        # Load the actual files
        for iFile in range(len(mesh_files)):

            print("loading mesh " + str(mesh_files[iFile]))

            verts, faces = pp3d.read_mesh(mesh_files[iFile])

            # to torch
            verts = torch.tensor(np.ascontiguousarray(verts)).float()
            faces = torch.tensor(np.ascontiguousarray(faces))

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            # normalize area
            verts = normalize_area_scale(verts, faces)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.names_list.append(os.path.basename(mesh_files[iFile]).split(".")[0])
            idx0 = farthest_point_sample(verts.t(), ratio=0.9)
            dists, idx1 = square_distance(verts.unsqueeze(0), verts[idx0].unsqueeze(0)).sort(dim=-1)
            dists, idx1 = dists[:, :, :130].clone(), idx1[:, :, :130].clone()
            self.sample_list.append((idx0, idx1, dists))

        for ind, labels in enumerate(self.vts_list):
            self.vts_list[ind] = labels

        # Precompute operators
        (
            self.frames_list,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = diffusion_net.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=self.op_cache_dir,
        )

        self.hks_list = [diffusion_net.geometry.compute_hks_autoscale(self.evals_list[i], self.evecs_list[i], 16)
                         for i in range(len(self.L_list))]

        # save to cache
        if use_cache:
            diffusion_net.utils.ensure_dir_exists(self.cache_dir)
            torch.save(
                (
                    self.verts_list,
                    self.faces_list,
                    self.frames_list,
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    self.hks_list,
                    self.vts_list,
                    self.names_list,
                    self.sample_list
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        idx1, idx2 = self.combinations[idx]

        shape1 = {
            "xyz": self.verts_list[idx1],
            "faces": self.faces_list[idx1],
            "frames": self.frames_list[idx1],
            "mass": self.massvec_list[idx1],
            "L": self.L_list[idx1],
            "evals": self.evals_list[idx1],
            "evecs": self.evecs_list[idx1],
            "gradX": self.gradX_list[idx1],
            "gradY": self.gradY_list[idx1],
            "name": self.names_list[idx1],
            "sample_idx": self.sample_list[idx1],
        }

        shape2 = {
            "xyz": self.verts_list[idx2],
            "faces": self.faces_list[idx2],
            "frames": self.frames_list[idx2],
            "mass": self.massvec_list[idx2],
            "L": self.L_list[idx2],
            "evals": self.evals_list[idx2],
            "evecs": self.evecs_list[idx2],
            "gradX": self.gradX_list[idx2],
            "gradY": self.gradY_list[idx2],
            "name": self.names_list[idx2],
            "sample_idx": self.sample_list[idx2],
        }

        return {"shape1": shape1, "shape2": shape2}


def shape_to_device(dict_shape, device):
    names_to_device = ["xyz", "faces", "mass", "evals", "evecs", "gradX", "gradY"]
    for k, v in dict_shape.items():
        if "shape" in k:
            for name in names_to_device:
                v[name] = v[name].to(device)
            dict_shape[k] = v
        else:
            dict_shape[k] = v.to(device)

    return dict_shape
