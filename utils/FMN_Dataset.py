import numpy as np
import pickle
import os
import os.path as osp

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from .dataloading import (
    dataloader_GALLOP,
    dataloader_FAUST,
    dataloader_TRUCK,
    dataloader_syn,
    def_FMN_from_p2p,
    extend_FMN_by_p2p,
)

import diffusion_net as dfn


class FMN_Dataset(Dataset):
    """
    Parameters:
    select_parts: select parts, list of strings
    select_tt: select timesteps, array of integers
    k_eig: number of Laplace-Beltrami eigenvectors loaded
    FM_meshes: precomputed Functional Maps Network
    p2p_maps: filename where P2P maps are saved for reference shapes for the different parts of the dataset
    name: name of the dataset
    ndim: set the dimension of the encoded functional maps.
    nCCLB: set size for canonicalization of consistent latent basis
    trainsplit: in range (0,1)
    use_cache: boolean
    cache_dir: where to save/load the cached dataset if use_cache is True, default: 'data/cache'
    train: boolean
    """

    def __init__(
        self,
        select_parts,
        select_tt,
        k_eig=128,
        FM_meshes=None,
        p2p_maps="data/TRUCK_p2p.pickle",
        name="TRUCK-1part",
        ndim=60,
        nCCLB=50,
        maxsize_circle=None,
        trainsplit=0.8,
        seed=71,
        templates_ids=[0],
        template_features=None,
        template_feature_size=0,
        select_template=False,
        fmn_only_train=True,  # consider only train meshes to calculate CLB
        use_cache=True,
        cache_dir="data/cache",
        cache_FMN=None,
        order_st=False,
        datadir="data",
        train=True,
        verbose=True,
    ):
        self.k_eig = k_eig
        self.ndim = ndim
        self.nCCLB = nCCLB
        self.cache_dir = cache_dir
        # for test dataset the functional map network is given
        self.FM_meshes = FM_meshes
        self.maxsize_circle = maxsize_circle

        self.templates_ids = templates_ids

        self.template_features = template_features
        self.template_feature_size = template_feature_size

        split = "train" if train else "test"

        if use_cache:
            if isinstance(trainsplit, list):
                train_ref = len(trainsplit)
                dataname_seed = name
            elif isinstance(trainsplit, str):
                train_ref = trainsplit
                dataname_seed = name
            else:
                # if trainsplit is random
                train_ref = trainsplit
                dataname_seed = "{}-{}".format(name, seed)

            # if 'TRUCK' in name:
            # # referece p2p maps version
            load_cache = os.path.join(
                self.cache_dir,
                f"cache_{dataname_seed}_{self.ndim}_{self.nCCLB}_c{self.maxsize_circle}_onlytrain{fmn_only_train}_{self.k_eig}_template_{self.template_feature_size}{self.template_features}_{split}_{train_ref}_{p2p_maps.split('/')[-1].split('.')[0]}.pt",
            )

            if verbose:
                print("using dataset cache path: " + str(load_cache))
            if os.path.exists(load_cache) and os.path.exists(cache_FMN):
                with open(cache_FMN, "rb") as handle:
                    if verbose:
                        print("using FMN cache path: {}".format(cache_FMN))
                    self.FM_meshes = pickle.load(handle)

                if verbose:
                    print("  --> loading dataset from cache")
                (
                    # mesh
                    self.verts_list,
                    self.numverts_list,
                    self.faces_list,
                    self.center_mass_list,
                    self.sqrtarea_list,
                    # projection matrices
                    self.phi_pinv_list,
                    self.phi_list,
                    self.Y_pinv_list,
                    # diffNet
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    # sample info
                    self.mesh_id,
                    self.template_id,
                    self.part_id,
                    self.sim_id,
                    self.timestep,
                    self.p2p_2_temp,
                    self.select_samp,
                    # templates
                    self.templates_ids,
                    self.template_verts_list,
                    self.template_numverts_list,
                    self.template_faces_list,
                    self.template_phi_list,
                    self.template_Y_list,
                    # template meshfeatures
                    self.template_meshfeature_list,
                    # diffNet
                    self.template_massvec_list,
                    self.template_L_list,
                    self.template_evals_list,
                    self.template_evecs_list,
                    self.template_gradX_list,
                    self.template_gradY_list,
                ) = torch.load(load_cache)

                return
            print("  --> dataset not in cache, repopulating")

        # # load meshes
        if "gallop" in name or "GALLOP" in name:
            meshes = dataloader_GALLOP(
                select_parts, select_tt, ndim=self.ndim, datadir=datadir
            )
            # set ids with respect to original dataset ordering
            self.templates_ids_org = []
            for pn, pp in enumerate(select_parts):
                self.templates_ids_org += [pn * len(select_tt)]

        if "faust-scape" in name or "FAUST-SCAPE" in name:
            meshes = dataloader_FAUST(
                select_parts, select_tt, ndim=self.ndim, datadir=datadir
            )
            # set ids with respect to original dataset ordering
            self.templates_ids_org = []
            for pn, pp in enumerate(select_parts):
                self.templates_ids_org += [pn * len(select_tt[0])]   
                print(self.templates_ids_org)
                
        elif "faust" in name or "FAUST" in name:
            meshes = dataloader_FAUST(
                select_parts, select_tt, ndim=self.ndim, datadir=datadir
            )
            # set ids with respect to original dataset ordering
            self.templates_ids_org = []
            for pn, pp in enumerate(select_parts):
                self.templates_ids_org += [pn * len(select_tt)]   
                print(self.templates_ids_org)
                
        elif "scape" in name or "SCAPE" in name:
            meshes = dataloader_FAUST(
                select_parts, select_tt, ndim=self.ndim, datadir=datadir
            )
            # set ids with respect to original dataset ordering
            self.templates_ids_org = []
            for pn, pp in enumerate(select_parts):
                self.templates_ids_org += [pn * len(select_tt)]   
                print(self.templates_ids_org)
                
        if "TRUCK" in name:
            meshes = dataloader_TRUCK(
                select_parts,
                select_tt,
                ndim=self.ndim,
                datadir=datadir,
                verbose=verbose,
                order_p_s_t=order_st
            )
            # set ids with respect to original dataset ordering
            self.templates_ids_org = []
            for pn, pp in enumerate(select_parts):
                self.templates_ids_org += [pn * len(select_tt) * 32]
            self.select_sim = None
            if isinstance(trainsplit, str):
                if 'select' in trainsplit:
                    self.select_sim = ['027', '049', '053', '111', '112', '221', '222', '273', '274', '285']
                    # for 12 train simulations:
                    # ['027', '049', '053', '110', '111', '112', '221', '222', '257', '273', '274', '285']
                    # # select only the selected simulations, not the template simulation
                    self.select_samp = []
                    for mi, mm in enumerate(meshes):
                        if mm.simid in self.select_sim:
                            self.select_samp += [mi]
                    self.select_samp = np.asarray(self.select_samp)

        # get train-test split
        nshapes_all = len(meshes)
        np.random.seed(seed)
        if isinstance(trainsplit, list):
            self.select_samp = np.asarray(trainsplit)
            for ii in self.templates_ids_org:
                tmp = np.where(self.select_samp == ii)[0]
                self.select_samp = np.delete(self.select_samp, tmp)
        elif 'TRUCK' in name and self.select_sim is not None:
            print('Fixed simulation set for training:', self.select_sim)
        else:
            # training samples (skip the template!)
            tmp = np.delete(np.arange(nshapes_all, dtype=int), self.templates_ids_org)
            self.select_samp = np.random.choice(
                tmp,
                int(trainsplit * (nshapes_all - len(self.templates_ids_org))),
                replace=False,
            )
        if split == "test":
            # testing samples (skip the template!)
            self.select_samp = np.delete(
                np.arange(nshapes_all, dtype=int), self.select_samp
            )
            for ii in self.templates_ids_org:
                tmp = np.where(self.select_samp == ii)[0]
                self.select_samp = np.delete(self.select_samp, tmp)
        self.select_samp.sort()

        # # define functional map network
        if self.FM_meshes is None:
            if split == "train":
                print("Get Functional Maps Network")
                # # load point-to-point maps
                with open(p2p_maps, "rb") as handle:
                    samples_p2p = pickle.load(handle)

                if fmn_only_train:
                    # select only train samples and templates to calculate CLB and CCLB
                    select_meshes = np.concatenate(
                        (self.select_samp, np.array(self.templates_ids_org))
                    )
                    select_meshes.sort()
                    print('select_meshes',select_meshes)
                    print('meshes', len(meshes))
                    meshlist = [meshes[ii] for ii in select_meshes]
                else:
                    meshlist = meshes

                # select_tt[0], because first timestep is the template shape
                template_tt = select_tt[0]

                if isinstance(select_tt, list):
                    template_tt = 0
                self.FM_meshes, self.templates_ids = def_FMN_from_p2p(
                    meshlist,
                    select_parts,
                    template_tt,
                    samples_p2p,
                    maxsize_circle=self.maxsize_circle,
                    verbose=verbose,
                )

                # convert the p2p maps to functional maps.
                print("  Compute Maps (ndim={})".format(ndim))
                self.FM_meshes.compute_maps(ndim)  # for every edge in graph
                self.FM_meshes = self.FM_meshes.set_maps(self.FM_meshes.maps)
                self.FM_meshes = self.FM_meshes.set_weights(weight_type="adjacency")
                # iscm: makes it more robust because somehow bad FM have lower weights
                # adjacency: every wheight one

                # compute the consistent latent basis: FM_meshes.CLB
                print("  ", end="")
                self.FM_meshes.compute_CLB(verbose=True)
                # canonicalize the first nCCLB consistent latent basis: FM_meshes.CCLB
                self.FM_meshes = self.FM_meshes.compute_CCLB(nCCLB)
            else:
                print(
                    "Error: FM_meshes must be given for test split. at first define FMN for train data and compute the CCLB"
                )
                return 0

        else:
            if split == "test" and fmn_only_train:
                # # load point-to-point maps
                with open(p2p_maps, "rb") as handle:
                    samples_p2p = pickle.load(handle)
                self.FM_meshes = extend_FMN_by_p2p(
                    self.FM_meshes,
                    [meshes[ii] for ii in self.select_samp],
                    select_parts,
                    self.templates_ids,
                    samples_p2p,
                    verbose=verbose,
                )

        # # define mesh-vise variables necessary for the network
        self.verts_list = []
        self.numverts_list = []
        self.faces_list = []
        self.center_mass_list = []
        self.sqrtarea_list = []
        self.phi_pinv_list = []
        self.phi_list = []
        self.Y_pinv_list = []
        self.mesh_id = []
        self.template_id = []
        self.p2p_2_temp = []
        self.part_id = []
        self.sim_id = []
        self.timestep = []
        self.template_verts_list = []
        self.template_numverts_list = []
        self.template_faces_list = []
        self.template_phi_list = []
        self.template_Y_list = []
        self.template_meshfeature_list = []

        listi = self.select_samp
        if fmn_only_train:
            if split == "test":
                listi = np.arange(
                    nshapes_all - len(self.select_samp), nshapes_all
                )  # test are in the end of the list
            else:
                listi = np.delete(
                    np.arange(len(self.select_samp) + len(self.templates_ids_org)),
                    self.templates_ids,
                )

        print("Prepare {} Samples".format(split))
        # # train: len(self.select_samp) + len(self.templates_ids_org)
        # if idx in self.templates_ids: continue
        for idx in tqdm(listi):
            mesh_sel = self.FM_meshes.meshlist[idx]
            self.numverts_list += [len(mesh_sel.vertlist)]
            # to torch
            verts = torch.tensor(np.ascontiguousarray(mesh_sel.vertlist)).float()
            faces = torch.tensor(np.ascontiguousarray(mesh_sel.facelist))
            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.center_mass_list.append(mesh_sel.center_mass_org)
            self.sqrtarea_list.append(mesh_sel.sqrtarea_org)

            self.phi_pinv_list += [
                np.linalg.pinv(mesh_sel.eigenvectors[:, : self.ndim])
            ]
            self.phi_list += [mesh_sel.eigenvectors[:, : self.ndim]]
            self.Y_pinv_list += [np.linalg.pinv(self.FM_meshes.CCLB[idx])]
            self.mesh_id += [idx]
            if type(mesh_sel.partid) != int:
                self.template_id += [(select_parts).index(mesh_sel.partid)]
            else:
                self.template_id += [mesh_sel.partid]
            self.part_id += [mesh_sel.partid]
            self.sim_id += [mesh_sel.simid]
            self.timestep += [mesh_sel.timestep]
            # ## point to point map to go from shape to template vertices
            temp_sel = self.FM_meshes.meshlist[self.templates_ids[self.template_id[-1]]]
            # from IPython.core.debugger import Pdb; Pdb().set_trace()
            
            mesh_timestep = mesh_sel.timestep
            if 'triangle' in self.part_id:
                temp_sel.timestep = 0
                mesh_timestep = 0
            
            self.p2p_2_temp += [
                samples_p2p[
                    (
                        "{}_{}_{}".format(
                            temp_sel.simid, temp_sel.partid, temp_sel.timestep
                        ),
                        "{}_{}_{}".format(
                            mesh_sel.simid, mesh_sel.partid, mesh_timestep
                        ),
                    )
                ]
            ]

        # Precompute operators
        (
            _,
            self.massvec_list,
            self.L_list,
            self.evals_list,
            self.evecs_list,
            self.gradX_list,
            self.gradY_list,
        ) = dfn.geometry.get_all_operators(
            self.verts_list,
            self.faces_list,
            k_eig=self.k_eig,
            op_cache_dir=cache_dir + "/cache",
        )

        print("Prepare Templates")
        for idx in tqdm(self.templates_ids):
            mesh_sel = self.FM_meshes.meshlist[idx]
            self.template_numverts_list += [len(mesh_sel.vertlist)]
            # to torch
            verts = torch.tensor(np.ascontiguousarray(mesh_sel.vertlist)).float()
            faces = torch.tensor(np.ascontiguousarray(mesh_sel.facelist))
            self.template_verts_list.append(verts)
            self.template_faces_list.append(faces)

            self.template_phi_list += [mesh_sel.eigenvectors[:, : self.ndim]]
            self.template_Y_list += [self.FM_meshes.CCLB[idx]]

            # # calculate/load the mesh feature
            if self.template_features is not None:
                datapath = "Data_{}".format(select_parts[self.templates_ids.index(idx)])
                if self.template_features.lower() == "shot":
                    shot_features = [
                        f.name
                        for f in os.scandir(osp.join(datadir, datapath))
                        if "shot" in f.name
                        and int(f.name.split("-")[1]) == mesh_sel.timestep
                        and "checkpoints" not in f.name
                    ]
                    if len(shot_features) == 0:
                        print(
                            "either no shot features for reference shape in {} given or more than one file has been found.".format(
                                datapath
                            )
                        )
                        exit()
                    shot_features = shot_features[0]
                    tmp = np.load(osp.join(datadir, datapath, shot_features))
                    tmp = tmp[:, : self.template_feature_size]
                    self.template_meshfeature_list += [tmp]

                elif self.template_features.lower() == "xyz":
                    self.template_meshfeature_list.append(verts)

            else:
                self.template_meshfeature_list += [0]

        # templates: Precompute operators
        (
            _,
            self.template_massvec_list,
            self.template_L_list,
            self.template_evals_list,
            self.template_evecs_list,
            self.template_gradX_list,
            self.template_gradY_list,
        ) = dfn.geometry.get_all_operators(
            self.template_verts_list,
            self.template_faces_list,
            k_eig=self.k_eig,
            op_cache_dir=cache_dir + "/cache",
        )

        # save to cache
        if use_cache:
            if not osp.exists(self.cache_dir):
                os.makedirs(self.cache_dir)

            if not os.path.exists(cache_FMN):
                if verbose:
                    print("using FMN cache path: {}".format(cache_FMN))
                with open(cache_FMN, "wb") as handle:
                    pickle.dump(self.FM_meshes, handle)

            torch.save(
                (
                    self.verts_list,
                    self.numverts_list,
                    self.faces_list,
                    self.center_mass_list,
                    self.sqrtarea_list,
                    #
                    self.phi_pinv_list,
                    self.phi_list,
                    self.Y_pinv_list,
                    # diffNet
                    self.massvec_list,
                    self.L_list,
                    self.evals_list,
                    self.evecs_list,
                    self.gradX_list,
                    self.gradY_list,
                    #
                    self.mesh_id,
                    self.template_id,
                    self.part_id,
                    self.sim_id,
                    self.timestep,
                    self.p2p_2_temp,
                    #
                    self.select_samp,
                    #
                    self.templates_ids,
                    self.template_verts_list,
                    self.template_numverts_list,
                    self.template_faces_list,
                    self.template_phi_list,
                    self.template_Y_list,
                    # template meshfeatures
                    self.template_meshfeature_list,
                    # diffNet
                    self.template_massvec_list,
                    self.template_L_list,
                    self.template_evals_list,
                    self.template_evecs_list,
                    self.template_gradX_list,
                    self.template_gradY_list,
                ),
                load_cache,
            )

    def __len__(self):
        return len(self.select_samp)

    def __getitem__(self, idx):
        shape = {
            "xyz": self.verts_list[idx],
            "nn": self.numverts_list[idx],
            "faces": self.faces_list[idx],
            "center_mass": self.center_mass_list[idx],
            "sqrtarea": self.sqrtarea_list[idx],
            #
            "phi_pinv": self.phi_pinv_list[idx],
            "phi": self.phi_list[idx],
            "Y_pinv": self.Y_pinv_list[idx],
            "mesh_id": self.mesh_id[idx],
            "template_id": self.template_id[idx],
            "p2p_temp": self.p2p_2_temp[idx],
            # diffNet
            "mass": self.massvec_list[idx],
            "L": self.L_list[idx],
            "evals": self.evals_list[idx],
            "evecs": self.evecs_list[idx],
            "gradX": self.gradX_list[idx],
            "gradY": self.gradY_list[idx],
            #
            "part_id": self.part_id[idx],
            "simid": self.sim_id[idx],
            "timestep": self.timestep[idx],
        }

        template = {
            # "xyz": self.template_verts_list[self.template_id[idx]],
            "nn": self.template_numverts_list[self.template_id[idx]],
            "faces": self.template_faces_list[self.template_id[idx]],
            #
            "phi": self.template_phi_list[self.template_id[idx]],
            "Y": self.template_Y_list[self.template_id[idx]],
            # mesh feature
            "meshfeatures": self.template_meshfeature_list[self.template_id[idx]],
            # diffNet
            "mass": self.template_massvec_list[self.template_id[idx]],
            "L": self.template_L_list[self.template_id[idx]],
            "evals": self.template_evals_list[self.template_id[idx]],
            "evecs": self.template_evecs_list[self.template_id[idx]],
            "gradX": self.template_gradX_list[self.template_id[idx]],
            "gradY": self.template_gradY_list[self.template_id[idx]],
        }

        return {"shape": shape, "template": template}


def shape_to_device(dict_shape, device):
    names_to_device = [
        "xyz",
        "faces",
        "phi_pinv",
        "Y_pinv",
        "phi",
        "Y",
        "meshfeatures",
        "mass",
        "evals",
        "evecs",
        "gradX",
        "gradY",
        "p2p_temp",
    ]
    names_long = ["p2p_temp"]
    for k, v in dict_shape.items():
        for k2, v2 in v.items():
            if k2 in names_to_device:
                if v2 is not None:
                    if k2 in names_long:
                        v[k2] = v[k2].to(device).long()
                    else:
                        v[k2] = v[k2].to(device).float()
    return dict_shape


# runtime of collect_fn function -> x2 (compared when having only one part)
def collate_fn(batch) -> tuple:
    names_padded = ["xyz", "faces", "phi_pinv", "phi"]
    names_nottensor = ["mesh_id", "template_id", "simid", "timestep"]

    shape = dict()
    template = dict()

    for ii, sample in enumerate(batch):
        for kk, kv in sample["shape"].items():
            if ii == 0:
                shape[kk] = []
            if kk in names_padded:
                if "phi_pinv" in kk:
                    kv = torch.swapaxes(torch.tensor(kv), 0, 1)
                else:
                    kv = torch.tensor(kv)
            shape[kk].append(kv)
        for kk, kv in sample["template"].items():
            if ii == 0:
                template[kk] = []
            if kk in names_padded:
                if "phi_pinv" in kk:
                    kv = torch.swapaxes(torch.tensor(kv), 0, 1)
                else:
                    kv = torch.tensor(kv)
            template[kk].append(kv)

    for kk in shape.keys():
        if kk in names_padded:
            shape[kk] = pad_sequence(shape[kk], batch_first=True, padding_value=0)
            if "phi_pinv" in kk:
                shape[kk] = torch.swapaxes(shape[kk], 1, 2)
        elif kk not in names_nottensor:
            shape[kk] = torch.tensor(np.array(shape[kk]))

    for kk in template.keys():
        if kk in names_padded:
            template[kk] = pad_sequence(template[kk], batch_first=True, padding_value=0)
            if "phi_pinv" in kk:
                template[kk] = torch.swapaxes(template[kk], 1, 2)

        elif kk not in names_nottensor:
            template[kk] = torch.tensor(np.array(template[kk]))

    return {"shape": shape, "template": template}
