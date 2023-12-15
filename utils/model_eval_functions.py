import numpy as np
import torch
import igl

import os
import os.path as osp
import pickle

import matplotlib as mpl
from matplotlib import cm

from utils.FMN_Dataset import FMN_Dataset, shape_to_device

import importlib
meshplot_spec = importlib.util.find_spec("meshplot")
meshplot_found = meshplot_spec is not None
if meshplot_found:
    import meshplot as mp
import matplotlib.pyplot as plt
import matplotlib
from utils.utils import get_rgba

import itertools

from model import LS_DF_net

colors = [
    "#9467bd",
    "#ff7f0e",
    "#1f77b4",
    "#8c564b",
    "#7f7f7f",
    "#d62728",
    "#e377c2",
    "#2ca02c",
    "#17becf",
    "#fac205",
    "#bcbd22",
    "#8c000f",
]

## colorblind colors
colors_CB = ['#377eb8', '#ff7f00', '#f781bf', '#4daf4a',
    '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00',"#8c000f",]
br_man = np.asarray(
    [np.arange(10 * ii, 10 * (ii + 1)) for ii in [0, 2, 3, 7, 9]]
).flatten()

faust_labels = [
    "normal",
    "macarena",
    "shouldersup",
    "forearmup",
    "lowerlegbent",
    "headright",
    "armsfront",
    "tree",
    "step",
    "armsup",
]

left = [
    0,
    1,
    2,
    4,
    5,
    8,
]  # normal, macarena, shouldersup, lowerlegbent, headright+armsslightlylifted, step
right = [3, 6, 7, 9]  # forearmup, armsfront, yogatree, armsup

markers = ['o','d','s','^','p','*',"<","v",'P',">"]


def eval_yalm(cfg):

    cfg["model"]["template_features"] = eval(cfg["model"]["template_features"])
    if 'SCAPE' in cfg["dataset"]["templates"] and 'FAUST' in cfg["dataset"]["templates"]:
        print('Scape and Faust')
        cfg["dataset"]["select_tt"] = [ eval(cfg["dataset"]["select_tt"][0]+','+cfg["dataset"]["select_tt"][1]+','+cfg["dataset"]["select_tt"][2]) ,
                                        eval(cfg["dataset"]["select_tt"][3]+','+cfg["dataset"]["select_tt"][4]+','+cfg["dataset"]["select_tt"][5])]
    else:
        cfg["dataset"]["select_tt"] = eval(cfg["dataset"]["select_tt"])

    # evaluate lists define in the config file
    if (
        not isinstance(cfg["training"]["trainsplit"], float)
        and "FAUST" not in cfg["dataset"]["name"]
        and "TRUCK" not in cfg["dataset"]["name"] 
    ):
        cfg["training"]["trainsplit"] = eval(cfg["training"]["trainsplit"])

    # FAUST and SCAPE: interpolate or extrapolate?
    if "FAUST-SCAPE" in cfg["dataset"]["name"] and isinstance(
        cfg["training"]["trainsplit"], str
    ):
        if "int" in cfg["training"]["trainsplit"]:
            cfg["training"]["trainsplit"] = list(
                np.concatenate((np.arange(1, 80),
                                    100 + np.delete(np.arange(71), np.array([6,8,18,22,26,35,39,44,57,64]))
                                   )
                              )
            )
        else:
            cfg["training"]["trainsplit"] = list(
                np.concatenate((np.delete(
                                        np.delete(np.arange(100), np.arange(9, 100, 10)),
                                        np.arange(8, 90, 9),
                                    ),
                                    100 + np.delete(np.arange(71), np.array([6,8,18,22,26,35,39,44,57,64]))
                                   )
                              )
            )
        print("Extrapolate to new shapes. Training timesteps: ", end="")
        print(cfg["training"]["trainsplit"])
    # FAUST: interpolate or extrapolate?
    elif "FAUST" in cfg["dataset"]["name"] and isinstance(
        cfg["training"]["trainsplit"], str
    ):
        if "int" in cfg["training"]["trainsplit"]:
            cfg["training"]["trainsplit"] = list(np.arange(1, 80))
            print("Interpolate between persons. Training timesteps: ", end="")
        else:
            cfg["training"]["trainsplit"] = list(
                np.delete(
                    np.delete(np.arange(100), np.arange(9, 100, 10)),
                    np.arange(8, 90, 9),
                )
            )
            print("Extrapolate to new shapes. Training timesteps: ", end="")
        print(cfg["training"]["trainsplit"])

    if isinstance(cfg["fmnet"]["maxsize_circle"], str):
        cfg["fmnet"]["maxsize_circle"] = eval(cfg["fmnet"]["maxsize_circle"])

    return cfg


def model_id_gen(cfg):
    if isinstance(cfg["training"]["trainsplit"], list):
        train_ref = len(cfg["training"]["trainsplit"])
    else:
        train_ref = cfg["training"]["trainsplit"]
    #print("train_ref", train_ref)

    model_id = (
        "{}_{}-{}_fmnet_{}_{}_c{}_onlytrain_{}_diffnet_{}_{}_{}{}_train_{}_{}".format(
            cfg["model"]["name"],
            cfg["dataset"]["name"],
            cfg["training"]["seed"],
            cfg["fmnet"]["ndim"],
            cfg["fmnet"]["nCCLB"],
            cfg["fmnet"]["maxsize_circle"],
            cfg["fmnet"]["only_train"],
            cfg["diffnet"]["nfeature"],
            cfg["diffnet"]["k_eig"],
            cfg["model"]["template_features"],
            cfg["model"]["template_features_dim"],
            train_ref,
            cfg["training"]["lr"],
        )
    )

    if (
        cfg["model"]["name"] == "LS_DF_NN_net"
        or cfg["model"]["name"] == "LS_DF_fNN_net"
    ):
        model_id = "{}_{}_fmnet_{}_{}_c{}_onlytrain_{}_diffnet_{}_{}_LL_{}_{}{}_train_{}_{}".format(
            cfg["model"]["name"],
            cfg["dataset"]["name"],
            cfg["fmnet"]["ndim"],
            cfg["fmnet"]["nCCLB"],
            cfg["fmnet"]["maxsize_circle"],
            cfg["fmnet"]["only_train"],
            cfg["diffnet"]["nfeature"],
            cfg["diffnet"]["k_eig"],
            cfg["model"]["latentdim"],
            cfg["model"]["template_features"],
            cfg["model"]["template_features_dim"],
            train_ref,
            cfg["training"]["lr"],
        )

    model_id = model_id + "_" + cfg["dataset"]["p2p_maps"].split("/")[-1].split(".")[0]
    return model_id


def FMN_id_gen(cfg):
    cache_FMN = "cache_{}_FMN_{}_{}".format(
        cfg["dataset"]["name"], cfg["fmnet"]["ndim"], cfg["fmnet"]["nCCLB"]
    )
    if cfg["fmnet"]["only_train"]:
        if isinstance(cfg["training"]["trainsplit"], list):
            train_ref = len(cfg["training"]["trainsplit"])
            dataname_seed = cfg["dataset"]["name"]
        elif isinstance(cfg["training"]["trainsplit"], str):
            train_ref = cfg["training"]["trainsplit"]
            dataname_seed = cfg["dataset"]["name"]
        else:
            train_ref = cfg["training"]["trainsplit"]
            dataname_seed = "{}-{}".format(
                cfg["dataset"]["name"], cfg["training"]["seed"]
            )

        cache_FMN = "cache_{}_FMN_{}_{}_c{}_onlytrain_{}".format(
            dataname_seed,
            cfg["fmnet"]["ndim"],
            cfg["fmnet"]["nCCLB"],
            cfg["fmnet"]["maxsize_circle"],
            train_ref,
        )

    cache_FMN = cache_FMN + "_" + cfg["dataset"]["p2p_maps"].split("/")[1].split(".")[0]

    return cache_FMN


def def_model(cfg, device):
    ae_network = LS_DF_net(cfg).to(device)

    return ae_network


def get_emb_size(cfg):
    txt2 = "Embedding Size: {:,} * {:,} = {:,}".format(
        cfg["fmnet"]["nCCLB"],
        cfg["diffnet"]["nfeature"],
        cfg["fmnet"]["nCCLB"] * cfg["diffnet"]["nfeature"],
    )
    return txt2


# # model evaluation and save results
def save_pred_emb_and_loss(
    ae_network, cfg, loader, criterion, shape_to_device, device, norm_val=None
):
    select = 0  # # only works for batch size = 1!

    avg_loss = 0
    avg_limshape_loss = 0

    collect_loader = dict()

    for i, data in enumerate(loader):
        data = shape_to_device(data, device)
        pp = cfg["dataset"]["select_parts"][data["shape"]["template_id"][select]]

        encode = torch.matmul(
            data["shape"]["Y_pinv"],
            torch.matmul(data["shape"]["phi_pinv"], data["shape"]["xyz"]),
        )
        reconstruct_ls = torch.matmul(
            data["template"]["phi"], torch.matmul(data["template"]["Y"], encode)
        )
        avg_limshape_loss += (
            criterion(data["shape"]["xyz"], reconstruct_ls) / reconstruct_ls.shape[1]
        )

        reconstruct_t = ae_network(data)
        avg_loss += (
            criterion(data["shape"]["xyz"], reconstruct_t) / reconstruct_t.shape[1]
        )

        encoding = ae_network.only_encoder(data)

        true_xyz = data["shape"]["xyz"][select].detach().cpu().numpy()
        pred_xyz = reconstruct_t[select].detach().cpu().numpy()

        if norm_val is not None:
            # undo centering and area normalization
            pred_xyz = (
                pred_xyz + data["shape"]["center_mass"][select].detach().cpu().numpy()
            )
            true_xyz = (
                true_xyz + data["shape"]["center_mass"][select].detach().cpu().numpy()
            )
            pred_xyz = (
                pred_xyz * data["shape"]["sqrtarea"][select].detach().cpu().numpy()
            )
            true_xyz = (
                true_xyz * data["shape"]["sqrtarea"][select].detach().cpu().numpy()
            )

            # normalize into range -1,1 with values fixed for all parts
            if "FAUST" in cfg["dataset"]["name"] and pp == 'FAUST':
                pos = data["shape"]["timestep"][select].detach().cpu().numpy() % 10
                # it acutally is person not shape
                pos = int(data["shape"]["timestep"][select].detach().cpu().numpy() / 10)
                vval_min = norm_val[pp][pos][0]
                vval_max = norm_val[pp][pos][1]
            else:
                vval_min = norm_val[pp][0]
                vval_max = norm_val[pp][1]
            # # For truck also center per timestep
            if "TRUCK" in cfg["dataset"]["name"]:
                component_mean = np.mean(true_xyz, axis=(0))  # x, y, z - mean
                pred_xyz = pred_xyz - component_mean
                true_xyz = true_xyz - component_mean
            pred_xyz = (((pred_xyz - vval_min) / (vval_max - vval_min)) * 2) - 1
            true_xyz = (((true_xyz - vval_min) / (vval_max - vval_min)) * 2) - 1

        collect_loader[
            "{}_{}_{}_true".format(
                data["shape"]["simid"][select], pp, data["shape"]["timestep"][select]
            )
        ] = true_xyz
        collect_loader[
            "{}_{}_{}_recon".format(
                data["shape"]["simid"][select], pp, data["shape"]["timestep"][select]
            )
        ] = pred_xyz
        if norm_val is not None:
            collect_loader[
                "{}_{}_{}_trfaces".format(
                    data["shape"]["simid"][select],
                    pp,
                    data["shape"]["timestep"][select],
                )
            ] = (
                data["shape"]["faces"][select].detach().cpu().numpy()
            )
            collect_loader[
                "{}_{}_{}_recfaces".format(
                    data["shape"]["simid"][select],
                    pp,
                    data["shape"]["timestep"][select],
                )
            ] = (
                data["template"]["faces"][select].detach().cpu().numpy()
            )
        else:
            collect_loader[
                "{}_{}_{}_emb".format(
                    data["shape"]["simid"][select],
                    pp,
                    data["shape"]["timestep"][select],
                )
            ] = (
                encoding[select].detach().cpu().numpy().flatten()
            )
            
            collect_loader[
                "{}_{}_{}_sqrtarea".format(
                    data["shape"]["simid"][select],
                    pp,
                    data["shape"]["timestep"][select],
                )
            ] = data["shape"]["sqrtarea"][select].detach().cpu().numpy()
            
            collect_loader[
                "{}_{}_{}_center_mass".format(
                    data["shape"]["simid"][select],
                    pp,
                    data["shape"]["timestep"][select],
                )
            ] = data["shape"]["center_mass"][select].detach().cpu().numpy()
                        

        torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return collect_loader, avg_loss, avg_limshape_loss


def print_losses(
    cfg,
    collect_test_loader,
    collect_train_loader,
    log_file=None,
    save_recon_meshes=None,
    norm_val=None,
    Euclidean=False,
    vmin=0,
    vmax=0.0005,
):
    if log_file is not None:
        log_file.write("--------" + "\n")
    print("--------")

    # ## print the losses per part
    all_test = np.empty((0))
    all_train = np.empty((0))
    keys = list(collect_test_loader.keys())
    keys_train = list(collect_train_loader.keys())
    for pp in cfg["dataset"]["select_parts"]:
        xx_true = np.asarray(
            [
                collect_test_loader[keys[ii]]
                for ii, kk in enumerate(keys)
                if pp in kk and "true" in kk
            ]
        )
        xx_recon = np.asarray(
            [
                collect_test_loader[keys[ii]]
                for ii, kk in enumerate(keys)
                if pp in kk and "recon" in kk
            ]
        )

        xx_true_train = np.asarray(
            [
                collect_train_loader[keys_train[ii]]
                for ii, kk in enumerate(keys_train)
                if pp in kk and "true" in kk
            ]
        )
        xx_recon_train = np.asarray(
            [
                collect_train_loader[keys_train[ii]]
                for ii, kk in enumerate(keys_train)
                if pp in kk and "recon" in kk
            ]
        )

        if save_recon_meshes is not None:
            xx_reconfaces = np.asarray(
                [
                    collect_test_loader[keys[ii]]
                    for ii, kk in enumerate(keys)
                    if pp in kk and "recfaces" in kk
                ],
                dtype=int,
            )
            xx_reconfaces_train = np.asarray(
                [
                    collect_train_loader[keys_train[ii]]
                    for ii, kk in enumerate(keys_train)
                    if pp in kk and "recfaces" in kk
                ],
                dtype=int,
            )
            # get timesteps
            timesteps = np.asarray(
                [
                    int(keys[ii].split("_")[-2])
                    for ii, kk in enumerate(keys)
                    if pp in kk and "true" in kk
                ]
            )
            
            
            
            timesteps_train = np.asarray(
                [
                    int(keys_train[ii].split("_")[-2])
                    for ii, kk in enumerate(keys_train)
                    if pp in kk and "true" in kk
                ]
            )
            #### save simulations versions
            versions = np.asarray(
                [
                    int(keys[ii].split("_")[0])
                    for ii, kk in enumerate(keys)
                    if pp in kk and "true" in kk
                ]
            )
            versions_train = np.asarray(
                [
                    int(keys_train[ii].split("_")[0])
                    for ii, kk in enumerate(keys_train)
                    if pp in kk and "true" in kk
                ]
            )
            #

        if len(xx_true):
            
            if Euclidean:
                vval_min = norm_val[pp][0]
                vval_max = norm_val[pp][1]
                total_range = vval_max - vval_min

                # original range and from mm to cm
                xx_true = xx_true * (
                    total_range / (2.0 * 10.0)
                )  # * total_range / 2) / 10
                xx_recon = xx_recon * (total_range / (2.0 * 10.0))
                xx_true_train = xx_true_train * (total_range / (2.0 * 10.0))
                xx_recon_train = xx_recon_train * (total_range / (2.0 * 10.0))
                

            if Euclidean:
                #print('np.sum((xx_true - xx_recon) ** 2, axis=2)', np.sum((xx_true - xx_recon) ** 2, axis=2).shape)
                #
                all_test = np.concatenate(
                    (all_test, np.mean(np.sqrt(np.sum((xx_true - xx_recon) ** 2, axis=2)), axis=1))
                )  
                all_train = np.concatenate(
                    (
                        all_train,
                        np.mean(np.sqrt(
                            np.sum((xx_true_train - xx_recon_train) ** 2, axis=2)), axis=1
                        ),
                    )
                )  # np.mean(np.sum((xx_true_train-xx_recon_train)**2, axis=2), axis = 1)))
                txt2a = "final test  MSE-loss {:8}:       {:08.8f} cm".format(
                    pp, np.mean(np.mean(np.sqrt(np.sum((xx_true - xx_recon) ** 2, axis=2)), axis=1))
                )
                txt2b = "final train MSE-loss {:8}:       {:08.8f} cm".format(
                    pp,
                    np.mean(
                        np.mean(np.sqrt(
                            np.sum((xx_true_train - xx_recon_train) ** 2, axis=2)), axis=1
                        )
                    ),
                )
                
            else:

                all_test = np.concatenate(
                    (all_test, np.sum(np.sum((xx_true - xx_recon) ** 2, axis=2), axis=1))
                )  # np.mean(np.sum((xx_true-xx_recon)**2, axis=2), axis = 1)))
                all_train = np.concatenate(
                    (
                        all_train,
                        np.sum(
                            np.sum((xx_true_train - xx_recon_train) ** 2, axis=2), axis=1
                        ),
                    )
                )  # np.mean(np.sum((xx_true_train-xx_recon_train)**2, axis=2), axis = 1)))
                txt2a = "final test  MSE-loss {:8}:       {:08.8f}".format(
                    pp, np.mean(np.sum(np.sum((xx_true - xx_recon) ** 2, axis=2), axis=1))
                )
                txt2b = "final train MSE-loss {:8}:       {:08.8f}".format(
                    pp,
                    np.mean(
                        np.sum(
                            np.sum((xx_true_train - xx_recon_train) ** 2, axis=2), axis=1
                        )
                    ),
                )

            print(txt2a + "\n" + txt2b + "\n" + "--------")
            if log_file is not None:
                log_file.write(txt2a + "\n" + txt2b + "\n" + "--------" + "\n")

            
                
            for mm in range(len(xx_true)):
                if save_recon_meshes is not None:
                    
                    saveplots = False
                    model_id = model_id_gen(cfg)
                    
                    if 'GALLOP' in cfg['dataset']['name']:
                        save_path = osp.join(save_recon_meshes, model_id)
                        if not osp.exists(save_path):
                            os.makedirs(save_path)
                        if timesteps[mm] in [40,44]:
                            saveplots = True
                          
                    if 'SCAPE' in cfg['dataset']['name']:
                        save_path = osp.join(save_recon_meshes, model_id)
                        if not osp.exists(save_path):
                            os.makedirs(save_path)
                        print(timesteps[mm])
                        if timesteps[mm] in [6,8,18,22,26,35,39,44,57,64]:
                            saveplots = True
                        
                    elif 'FAUST' in cfg['dataset']['name']:
                        save_path = osp.join(save_recon_meshes, model_id)
                        if not osp.exists(save_path):
                            os.makedirs(save_path)
                        if 'extra' in cfg['dataset']['name']:
                            if timesteps[mm] in [18,19,58,59]:
                                saveplots = True
                        else:
                            if timesteps[mm] in [99,86]:
                                saveplots = True
                            
                            
                    if 'TRUCK' in cfg['dataset']['name']:
                        save_path = osp.join(save_recon_meshes, model_id)
                        if not osp.exists(save_path):
                            os.makedirs(save_path)
                        pnames = ['part_000','part_001']
                        if timesteps[mm] in [1*5,24*5]:
                            if versions[mm] in [1,18]:
                                if '000' in pp or '001' in pp in pnames:
                                    saveplots = True
                        
                    
                    if saveplots:
                        dis = np.sum((xx_recon[mm] - xx_true[mm]) ** 2, axis=1)

                        import meshio

                        for ll in [
                            ["pred", xx_recon[mm], xx_reconfaces[mm]] , ['true', xx_true[mm], xx_reconfaces[mm]]
                        ]:  
                            points = ll[1]
                            cells = [("triangle", ll[2])]

                            ColorHelper = MplColorHelper("plasma", vmin, vmax)
                            rgba = ColorHelper.get_rgb(dis)

                            mesh_pred = meshio.Mesh(
                                points,
                                cells,
                            )
                            title = "{}/{}_{}_v{}_{}_{}.off".format(
                                    save_path, pp, timesteps[mm], versions[mm], "test", ll[0]
                            )
                            mesh_pred.write(title)
                            if ll[0] == "pred":
                                np.savetxt(title[:-4]+'_MSEerror.txt', dis)
                                
                                cmapcolor = "hot_r"
                                vmax = 0.005
                                cmap3 = get_rgba(dis, cmapcolor=cmapcolor, vmin=0, vmax=vmax)

                                if meshplot_found and log_file is None:
                                    d = mp.plot(
                                        points, xx_reconfaces[mm], c=cmap3
                                    )  

                                
    txt2a = "final test  MSE-loss ALL     :       {:08.8f}".format(
        np.mean(np.asarray(all_test))
    )
    txt2b = "final train MSE-loss ALL     :       {:08.8f}".format(
        np.mean(np.asarray(all_train))
    )

    print(txt2a + "\n" + txt2b)
    if log_file is not None:
        log_file.write("--------\n" + txt2a + "\n" + txt2b)


def print_p2s_losses(
    cfg,
    collect_test_loader,
    collect_train_loader,
    save_recon_meshes=None,
    norm_val=None,
    Euclidean=False,
    vmin=0,
    vmax=0.0005,
):
    print("--------")

    # ## print the losses per part
    all_test = np.empty((0))
    all_train = np.empty((0))
    keys = list(collect_test_loader.keys())
    keys_train = list(collect_train_loader.keys())
    for pi, pp in enumerate(cfg["dataset"]["select_parts"]):
        all_test_pp = []
        all_train_pp = []
        # test
        xx_true = np.asarray(
            [
                collect_test_loader[keys[ii]]
                for ii, kk in enumerate(keys)
                if pp in kk and "true" in kk
            ]
        )
        xx_recon = np.asarray(
            [
                collect_test_loader[keys[ii]]
                for ii, kk in enumerate(keys)
                if pp in kk and "recon" in kk
            ]
        )
        xx_truefaces = np.asarray(
            [
                collect_test_loader[keys[ii]]
                for ii, kk in enumerate(keys)
                if pp in kk and "trfaces" in kk
            ],
            dtype=int,
        )
        xx_reconfaces = np.asarray(
            [
                collect_test_loader[keys[ii]]
                for ii, kk in enumerate(keys)
                if pp in kk and "recfaces" in kk
            ],
            dtype=int,
        )
        # train
        xx_true_train = np.asarray(
            [
                collect_train_loader[keys_train[ii]]
                for ii, kk in enumerate(keys_train)
                if pp in kk and "true" in kk
            ]
        )
        xx_recon_train = np.asarray(
            [
                collect_train_loader[keys_train[ii]]
                for ii, kk in enumerate(keys_train)
                if pp in kk and "recon" in kk
            ]
        )
        xx_truefaces_train = np.asarray(
            [
                collect_train_loader[keys_train[ii]]
                for ii, kk in enumerate(keys_train)
                if pp in kk and "trfaces" in kk
            ],
            dtype=int,
        )
        # get timesteps
        timesteps = np.asarray(
            [
                int(keys[ii].split("_")[-2])
                for ii, kk in enumerate(keys)
                if pp in kk and "true" in kk
            ]
        )

        if len(xx_true):
            if Euclidean:
                vval_min = norm_val[pp][0]
                vval_max = norm_val[pp][1]
                total_range = vval_max - vval_min

                # original range and from mm to cm
                xx_true = xx_true * (
                    total_range / (2.0 * 10.0)
                )  # * total_range / 2) / 10
                xx_recon = xx_recon * (total_range / (2.0 * 10.0))
                xx_true_train = xx_true_train * (total_range / (2.0 * 10.0))
                xx_recon_train = xx_recon_train * (total_range / (2.0 * 10.0))

            for mm in range(len(xx_true)):
                # ## p2s error
                dis, _, _ = igl.point_mesh_squared_distance(
                    xx_recon[mm], xx_true[mm], xx_truefaces[mm]
                )
                if Euclidean:
                    all_test_pp.append(np.mean(np.sqrt(dis)))  # np.mean(np.sqrt(dis)))
                else:
                    all_test_pp.append(np.sum(dis))

                if save_recon_meshes is not None:
                    import meshio

                    for ll in [
                        ["pred", xx_recon[mm], xx_reconfaces[mm]]
                        , ['true', xx_true[mm], xx_truefaces[mm]]
                    ]:  # , ['true', xx_true[mm], xx_truefaces[mm]]]:
                        points = ll[1]
                        cells = [("triangle", ll[2])]

                        ColorHelper = MplColorHelper("plasma", vmin, vmax)
                        rgba = ColorHelper.get_rgb(dis)

                        mesh_pred = meshio.Mesh(
                            points,
                            cells,
                            # Optionally provide extra data on points, cells, etc.
                            point_data={"P2S": 1000 * dis, "colors": rgba[:, :3]},
                        )
                        mesh_pred.write(
                            "{}/{}_{}_{}_{}.off".format(
                                save_recon_meshes, pp, timesteps[mm], "test", ll[0]
                            )
                        )

            txt2a = "final test  P2S-loss {:8}:       {:08.8f}".format(
                pp, np.mean(np.asarray(all_test_pp))
            )
            for mm in range(len(xx_true_train)):
                # ## p2s error
                dis, _, _ = igl.point_mesh_squared_distance(
                    xx_recon_train[mm], xx_true_train[mm], xx_truefaces_train[mm]
                )
                if Euclidean:
                    all_train_pp.append(np.mean(np.sqrt(dis)))  # np.mean(np.sqrt(dis)))
                else:
                    all_train_pp.append(np.sum(dis))  # np.mean(dis))

            txt2b = "final train P2S-loss {:8}:       {:08.8f}".format(
                pp, np.mean(np.asarray(all_train_pp))
            )
            if Euclidean:
                txt2a = txt2a + " cm"
                txt2b = txt2b + " cm"

            all_test = np.concatenate((all_test, np.asarray(all_test_pp)))
            all_train = np.concatenate((all_train, np.asarray(all_train_pp)))

            print(txt2a + "\n" + txt2b + "\n" + "--------")

    txt2a = "final test  P2S-loss ALL     :       {:08.8f}".format(
        np.mean(np.asarray(all_test))
    )
    txt2b = "final train P2S-loss ALL     :       {:08.8f}".format(
        np.mean(np.asarray(all_train))
    )

    print(txt2a + "\n" + txt2b)


def eval_network(
    trained_network,
    cfg,
    calculate_errors=True,
    calculate_errors_p2s=False,
    plot_reconstructions=True,
    plot_embeddings=True,
    use_cache=False,
    logfile = None,
    saveplots = True
):
    print("------------ EVAL --------------")

    # model reference
    model_id = model_id_gen(cfg)
    print(model_id)

    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device("cuda:{}".format(cfg["misc"]["device"]))
    else:
        device = torch.device("cpu")

    # important paths
    base_path = osp.os.getcwd()  # os.path.save_recname(__file__)
    cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    cache_FMN = osp.join(cache_dir, FMN_id_gen(cfg))
    p2p_maps_path = os.path.join(base_path, cfg["dataset"]["p2p_maps"])

    save_dir_name = "trained_{}".format(model_id)
    # model_save_path = os.path.join(base_path, "data", save_dir_name, "ep" + "_{}.pth")

    order_st = False
    if 'loading_order' in cfg['dataset']:
        if 'st' in cfg['dataset']['loading_order'] or 's_t' in cfg['dataset']['loading_order']:
            order_st = True

    # create train dataset
    train_dataset = FMN_Dataset(
        cfg["dataset"]["select_parts"],
        cfg["dataset"]["select_tt"],
        k_eig=cfg["diffnet"]["k_eig"],
        p2p_maps=p2p_maps_path,
        name=cfg["dataset"]["name"],
        ndim=cfg["fmnet"]["ndim"],
        nCCLB=cfg["fmnet"]["nCCLB"],
        maxsize_circle=cfg["fmnet"]["maxsize_circle"],
        trainsplit=cfg["training"]["trainsplit"],
        seed=cfg["training"]["seed"],
        template_features=cfg["model"]["template_features"],
        template_feature_size=cfg["model"]["template_features_dim"],
        fmn_only_train=cfg["fmnet"]["only_train"],
        use_cache=True,
        cache_dir=cache_dir,
        cache_FMN=cache_FMN + ".pickle",
        order_st=order_st,
        train=True,
        verbose=True,
    )
    
    print("Loaded training dataset. {} shapes".format(train_dataset.FM_meshes.n_meshes))

    # create test dataset
    test_dataset = FMN_Dataset(
        cfg["dataset"]["select_parts"],
        cfg["dataset"]["select_tt"],
        k_eig=cfg["diffnet"]["k_eig"],
        FM_meshes=train_dataset.FM_meshes,
        p2p_maps=p2p_maps_path,
        name=cfg["dataset"]["name"],
        ndim=cfg["fmnet"]["ndim"],
        nCCLB=cfg["fmnet"]["nCCLB"],
        maxsize_circle=cfg["fmnet"]["maxsize_circle"],
        trainsplit=cfg["training"]["trainsplit"],
        seed=cfg["training"]["seed"],
        template_features=cfg["model"]["template_features"],
        template_feature_size=cfg["model"]["template_features_dim"],
        templates_ids=train_dataset.templates_ids,
        fmn_only_train=cfg["fmnet"]["only_train"],
        use_cache=True,
        cache_dir=cache_dir,
        cache_FMN=cache_FMN + "_test.pickle",
        order_st=order_st,
        train=False,
        verbose=True,
    )
    
    print(
        "Extended to {} shapes with test shapes".format(test_dataset.FM_meshes.n_meshes)
    )

    # train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False
    )

    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


    # define model
    # ae_network = def_model(cfg, device)

    # # print("Load training results saved at:", model_save_path.format(cfg['training']['epochs']))
    # ae_network.load_state_dict(
    #     torch.load(
    #         model_save_path.format(cfg["training"]["epochs"]), map_location=device
    #     )
    # )
    ae_network = trained_network
    ae_network.eval()

    criterion = torch.nn.MSELoss(reduction="sum").to(device)
    pytorch_total_params = sum(
        p.numel() for p in ae_network.parameters() if p.requires_grad
    )

    txt1 = "Number of Parameters: {:,}".format(pytorch_total_params)
    # get embedding size
    txt2 = get_emb_size(cfg)
    print(txt1 + "\n" + txt2)

    with torch.no_grad():
        if calculate_errors:
            # this in parts also happens in the train script

            cache_dir_results = os.path.join(base_path, "data", save_dir_name, "cache")
            os.makedirs(cache_dir_results, exist_ok=True)
            load_cache_test = os.path.join(cache_dir_results, "test.pickle")
            load_cache_train = os.path.join(cache_dir_results, "train.pickle")

            if use_cache and os.path.exists(load_cache_test) and os.path.exists(load_cache_train):
                with open(load_cache_test, "rb") as handle:
                    collect_test_loader = pickle.load(handle)
                with open(load_cache_train, "rb") as handle:
                    collect_train_loader = pickle.load(handle)

            else:
                # test errors
                (
                    collect_test_loader,
                    avg_loss,
                    avg_limshape_loss,
                ) = save_pred_emb_and_loss(
                    ae_network, cfg, test_loader, criterion, shape_to_device, device
                )

                # train errors
                (
                    collect_train_loader,
                    avg_loss,
                    avg_limshape_loss,
                ) = save_pred_emb_and_loss(
                    ae_network, cfg, train_loader, criterion, shape_to_device, device
                )


                # save calculated results
                with open(load_cache_test, "wb") as handle:
                    pickle.dump(collect_test_loader, handle)
                with open(load_cache_train, "wb") as handle:
                    pickle.dump(collect_train_loader, handle)

            # ## print the losses
            print(
                "nCCLB = {}, DiffNet-nfeature = {} \n".format(
                    cfg["fmnet"]["nCCLB"], cfg["diffnet"]["nfeature"]
                )
            )
            print("\n--------\nErrors using area normalization")
            print_losses(cfg, collect_test_loader, collect_train_loader)

        if calculate_errors_p2s:
            print(
                "--------\n\nErrors using normalization into range [-1,1] as for the paper."
            )

            cache_dir_results = os.path.join(base_path, "data", save_dir_name, "cache")
            os.makedirs(cache_dir_results, exist_ok=True)
            load_cache_test = os.path.join(cache_dir_results, "test_norm.pickle")
            load_cache_train = os.path.join(cache_dir_results, "train_norm.pickle")

            # get the normalization values that are used for the comparison to baselines
            norm_val = dict()
            for pp in cfg["dataset"]["select_parts"]:
                datadir = "data"
                datadir = os.path.join(datadir, "Data_{}".format(pp))

                # # TODO for FAUST auch noch index für person dazu
                if "FAUST" in cfg["dataset"]["name"] and pp=='FAUST':
                    norm_val[pp] = dict()
                    for pos in range(10):
                        # different values for every person
                        normalization_values_file = osp.join(
                            datadir,
                            "normalization_min_max_values-faust{}.txt".format(pos),
                        )
                        norm_val[pp][pos] = np.loadtxt(normalization_values_file)
                elif "syn-tria" in cfg["dataset"]["name"]:                    
                    datadir_tria = os.path.join("data", "Data_{}".format(cfg["dataset"]["templates"][0]))
                    print('load normalization values for synthetic dataset from {}'.format(datadir_tria))
                    normalization_values_file = osp.join(
                        datadir_tria, "normalization_min_max_values.txt"
                    )
                    norm_val[pp] = np.loadtxt(normalization_values_file)
                else:
                    normalization_values_file = osp.join(
                        datadir, "normalization_min_max_values.txt"
                    )
                    norm_val[pp] = np.loadtxt(normalization_values_file)

            if use_cache and os.path.exists(load_cache_test) and os.path.exists(load_cache_train):
                with open(load_cache_test, "rb") as handle:
                    collect_test_loader_norm = pickle.load(handle)
                with open(load_cache_train, "rb") as handle:
                    collect_train_loader_norm = pickle.load(handle)

            else:
                # test errors
                (
                    collect_test_loader_norm,
                    avg_loss,
                    avg_limshape_loss,
                ) = save_pred_emb_and_loss(
                    ae_network,
                    cfg,
                    test_loader,
                    criterion,
                    shape_to_device,
                    device,
                    norm_val=norm_val,
                )

                # train errors
                (
                    collect_train_loader_norm,
                    avg_loss,
                    avg_limshape_loss,
                ) = save_pred_emb_and_loss(
                    ae_network,
                    cfg,
                    train_loader,
                    criterion,
                    shape_to_device,
                    device,
                    norm_val=norm_val,
                )

                # save calculated results
                with open(load_cache_test, "wb") as handle:
                    pickle.dump(collect_test_loader_norm, handle)
                with open(load_cache_train, "wb") as handle:
                    pickle.dump(collect_train_loader_norm, handle)

            save_meshes = os.path.join(
                base_path, "data", save_dir_name, "meshes"
            )
            os.makedirs(save_meshes, exist_ok=True)
            #print_p2s_losses(
            #    cfg,
            #    collect_test_loader_norm,
            #    collect_train_loader_norm,
            #    #save_recon_meshes = save_meshes,
            #    norm_val=norm_val,
            #)
            
            if not saveplots:
                print('save_meshes', save_meshes)
                save_meshes = None
            
            if logfile is not None:
                logfile.write("\n\nErrors using normalization into range [-1,1] as for the paper. \n\n")
            
            print_losses(
                cfg,
                collect_test_loader_norm,
                collect_train_loader_norm,
                save_recon_meshes=save_meshes,
                log_file = logfile
            )

            if False: #"TRUCK" in cfg["dataset"]["name"]:
                print_p2s_losses(
                    cfg,
                    collect_test_loader_norm,
                    collect_train_loader_norm,
                    #save_recon_meshes=save_meshes,
                    norm_val=norm_val,
                    Euclidean=True,
                )
                print_losses(
                    cfg,
                    collect_test_loader_norm,
                    collect_train_loader_norm,
                    #save_recon_meshes=save_meshes,
                    norm_val=norm_val,
                    Euclidean=True,
                )

        if plot_reconstructions:
            # plot some test samples
            

            nshape = len(cfg["dataset"]["select_parts"])
            plotshapes = [
                k * int(len(test_loader) / nshape)
                + int(0.5 * len(test_loader) / nshape)
                for k in range(nshape)
            ]
            
            plotshapes = [ll for ll in range(len(test_loader))]

            plotshapes = [0, 2, len(test_loader) - 2, len(test_loader) - 1]
            if len(cfg["dataset"]["select_parts"]) == 2:
                plotshapes = [0, int(len(test_loader)) - 3]
            if len(cfg["dataset"]["select_parts"]) == 3:
                plotshapes = [
                    0,
                    int(len(test_loader) / 2) + 3,
                    int(len(test_loader)) - 2,
                ]
            if len(cfg["dataset"]["select_parts"]) == 4:
                plotshapes = [
                    0,
                    int(len(test_loader) / 3) + 3,
                    int(2 * len(test_loader) / 3) + 3,
                    int(len(test_loader)) - 1,
                ]
            if len(cfg["dataset"]["select_parts"]) == 6:
                plotshapes = [
                    50,
                    int(len(test_loader) / 6) + 80,
                    int(2 * len(test_loader) / 6) + 80,
                    int(3 * len(test_loader) / 6) + 80,
                ]

            vmax = 0.01

            # plot the first train sample

            # nshape = len(cfg['dataset']['select_parts'])
            # plotshapes = [0]

            for k in plotshapes:
                data = next(itertools.islice(test_loader, k, None))
                # data = next(itertools.islice(train_loader, k, None))

                data = shape_to_device(data, device)
                select = 0  # batch size 1

                # print('Go via limit shape of size {}'.format(cfg['fmnet']['nCCLB']))
                print(
                    "\nSimulation {}, Test-Part {}, Timestep {}".format(
                        data["shape"]["simid"][select],
                        cfg["dataset"]["select_parts"][
                            data["shape"]["template_id"][select]
                        ],
                        data["shape"]["timestep"][select],
                    )
                )

                encode = torch.matmul(
                    data["shape"]["Y_pinv"],
                    torch.matmul(data["shape"]["phi_pinv"], data["shape"]["xyz"]),
                )
                reconstruct_ls = torch.matmul(
                    data["template"]["phi"], torch.matmul(data["template"]["Y"], encode)
                )
                loss_ls = criterion(data["shape"]["xyz"], reconstruct_ls)

                reconstruct_t = ae_network(data)
                loss = criterion(data["shape"]["xyz"], reconstruct_t)
                print(
                    "LimitShape-Loss: {:06.6f},   Network-Loss: {:06.6f}".format(
                        loss_ls / data["shape"]["xyz"].shape[1],
                        loss / data["shape"]["xyz"].shape[1],
                    )
                )

                xx_true = data["shape"]["xyz"][select].detach().cpu().numpy()
                # print('  min: {}, max: {}'.format(xx_true.min(axis=0), xx_true.max(axis=0)))
                xx_pred = reconstruct_t[select].detach().cpu().numpy()
                ff_temp = np.asarray(
                    data["template"]["faces"][select].detach().cpu().numpy(), dtype=int
                )

                xx_loss = np.sum((xx_true - xx_pred) ** 2, axis=1)
                ff_loss = np.asarray([np.mean(xx_loss[ff]) for ff in ff_temp])

                d = mp.subplot(
                    xx_true,
                    data["shape"]["faces"][select].detach().cpu().numpy(),
                    s=[2, 2, 0],
                )  # s=[3, 3, 0]
                # mp.subplot(reconstruct_ls[select].detach().cpu().numpy(), data['template']['faces'][select].detach().cpu().numpy(), s=[3, 3, 1], data=d)
                cmapcolor = "Blues"
                cmap3 = get_rgba(ff_loss, cmapcolor=cmapcolor, vmin=0, vmax=vmax)

                mp.subplot(
                    xx_pred, ff_temp, s=[2, 2, 1], data=d, c=cmap3
                )  # s=[3, 3, 2]


                
        if plot_embeddings:
            
            import matplotlib.font_manager as font_manager
            font = font_manager.FontProperties(size=16)
            
            cmaps = [mpl.cm.Blues,  mpl.cm.Greens,
                     mpl.cm.Purples,mpl.cm.Oranges]
            
            if calculate_errors:
                from sklearn.decomposition import PCA

                method = PCA(n_components=2)

                collect_results = collect_test_loader
                collect_results.update(collect_train_loader)
                keys = list(collect_results.keys())

                rows = int((len(cfg["dataset"]["templates"]) - 1) / 3) + 1
                if "FAUST" in cfg["dataset"]["name"]:
                    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
                    axs = [axs]
                elif 'syn' in cfg["dataset"]["name"]:
                    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
                    axs = [axs]
                else:
                    fig, axs = plt.subplots(rows, 3, figsize=(9, rows * 3))
                # # separate embeddings
                for ip, pp in enumerate(cfg["dataset"]["select_parts"]):
                    if rows > 1:
                        ax = axs[int(ip / 3), ip % 3]
                    else:
                        ax = axs[ip % 3]
                    ids = np.asarray(
                        [ii for ii, kk in enumerate(keys) if pp in kk and "emb" in kk]
                    )

                    if len(ids):
                        timesteps = np.asarray(
                            [int(keys[ii].split("_")[-2]) for ii in ids]
                        )

                        emb_h = np.asarray([collect_results[keys[ii]] for ii in ids])

                        emb2 = method.fit_transform(emb_h)

                        tt_sort = np.argsort(timesteps)
                                                
                        if "GALLOP" in cfg["dataset"]["name"]:
                            ax.plot(emb2[tt_sort][:, 0], emb2[tt_sort][:, 1], label=pp)
                            for tt in range(0, 12, 2):
                                ax.text(
                                    emb2[tt_sort][tt, 0], emb2[tt_sort][tt, 1], str(tt)
                                )
                                ax.legend()
                        elif "TRUCK" in cfg["dataset"]["name"]:
                            first_tt = 7
                            last_tt = timesteps.max()
                            select_i = np.where(
                                timesteps >= cfg["dataset"]["select_tt"][first_tt]
                            )[0]
                            emb_h = np.asarray(
                                [collect_results[keys[ii]] for ii in ids[select_i]]
                            )

                            emb2 = method.fit_transform(emb_h)

                            # # starting at t=7
                            cc = []
                            cmapbr1 = truncate_colormap(cmaps[(ip*2)%len(cmaps)], minval=0.3, maxval=1.0)
                            cmapbr2 = truncate_colormap(cmaps[(ip*2+1)%len(cmaps)], minval=0.3, maxval=1.0)
                            label1set = False
                            label2set = False
                            for inn,ii in enumerate(ids[select_i]):
                                if keys[ii].split("_")[0] in br1:
                                    #cc.append(colors[ip * 2])
                                    cc.append(cmapbr1(float(timesteps[select_i[inn]]-first_tt) / (last_tt-first_tt)))
                                    if timesteps[select_i[inn]] == last_tt and not label1set:
                                        ax.scatter(emb2[inn, 0], emb2[inn, 1], label=pp, c=cc[-1])
                                        label1set=True
                                else:
                                    #cc.append(colors[ip * 2 + 1])
                                    cc.append(cmapbr2(float(timesteps[select_i[inn]]-first_tt) / (last_tt-first_tt)))
                                    if timesteps[select_i[inn]] == last_tt and not label2set:
                                        ax.scatter(emb2[inn, 0], emb2[inn, 1], label=pp, c=cc[-1])
                                        label2set=True
                            #cc = np.array(cc)
                            ax.scatter(emb2[:, 0], emb2[:, 1], c=cc)
                            
                            ax.legend()
                        elif "FAUST" in cfg["dataset"]["name"]:
                            
                            cc = []
                            p_markers = []
                            for ii in ids:
                                cc_id = int(keys[ii].split("_")[2]) % 10
                                cc.append(colors_CB[cc_id])
                                p_markers.append(markers[cc_id])
                                                        
                            for ii in range(len(ids)):
                                ax.scatter(emb2[ii, 0], emb2[ii, 1], c=cc[ii], marker = p_markers[ii],
                                           s=250, edgecolor = 'black', linewidth=0.5)
                                
                            for ind, tt_s in enumerate(tt_sort[:10]):
                                cc_id = int(keys[ids[tt_s]].split("_")[2]) % 10 
                                ax.scatter(
                                    emb2[tt_s, 0],
                                    emb2[tt_s, 1],
                                    label=faust_labels[cc_id],
                                    c=colors_CB[cc_id],
                                    marker = p_markers[cc_id],
                                    s=180,
                                    edgecolor = 'black', 
                                    linewidth=0.5
                                )
                                
                                ax.set_xticks([])
                                ax.set_yticks([])
                                font = font_manager.FontProperties(size=24)
                                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',prop=font)
                                plt.tight_layout()
                                if saveplots:
                                    plt.savefig('embedding_plots/2d_{}.png'.format(model_id))
                                    
                        elif "syn" in cfg["dataset"]["name"]:
                            
                            print('tt_sort', len(tt_sort))

                            cmap = truncate_colormap(matplotlib.cm.plasma, minval=0.0, maxval=1.0)
                            cmap2 = truncate_colormap(matplotlib.cm.Greys, minval=0.0, maxval=0.95)
                            for ind, tt_s in enumerate(tt_sort):
                                cc_1 = cmap(float(int((ind)/25)) / (len(tt_sort)/25))
                                cc_2 = cmap2((float((ind)%25)) / 25)
                                                        
                                ax.scatter(emb2[tt_s, 0], emb2[tt_s, 1], c=cc_1,
                                           s=180, edgecolor = 'black')
                                ax.scatter(emb2[tt_s, 0], emb2[tt_s, 1], c=cc_2, 
                                           s=180, edgecolor = 'black', alpha=0.6)
                                
                                
                            ax.set_xticks([])
                            ax.set_yticks([])
                            plt.tight_layout()
                            if saveplots:
                                plt.savefig('embedding_plots/2d_{}.png'.format(model_id),transparent=True,dpi=500)
                                
                        else:
                            ax.scatter(
                                emb2[tt_sort][:, 0], emb2[tt_sort][:, 1], label=pp
                            )
                            ax.legend()

                if "FAUST" not in cfg["dataset"]["name"]:
                    
                    # # one embedding for all
                    method = PCA(n_components=3)

                    if 'TRUCK' in cfg["dataset"]["name"]: 
                        select_parts_emb = ['000', '001']
                        
                        ids = np.asarray([ii for ii, kk in enumerate(keys) if "emb" in kk and kk.split('_')[2] in select_parts_emb])
                    else:
                        ids = np.asarray([ii for ii, kk in enumerate(keys) if "emb" in kk])

                    timesteps_all = np.asarray([int(keys[ii].split("_")[-2]) for ii in ids])

                    first_tt = 0
                    last_tt = len(cfg["dataset"]["select_tt"]) - 1
                    if "TRUCK" in cfg["dataset"]["name"]:
                        first_tt = 6
                        last_tt = len(cfg["dataset"]["select_tt"]) - 5

                    select_i = np.where(
                        (timesteps_all >= cfg["dataset"]["select_tt"][first_tt])
                        & (timesteps_all <= cfg["dataset"]["select_tt"][last_tt])
                    )[0]

                    emb_h = np.asarray([collect_results[keys[ii]] for ii in ids[select_i]])
                    emb2 = method.fit_transform(emb_h)

                    ax = plt.figure(figsize=(8, 8)).add_subplot(projection="3d")
                    
                    axmin = [100,100,100]
                    axmax = [-100,-100,-100]
                    font = font_manager.FontProperties(size=16)
                    for ip, pp in enumerate(cfg["dataset"]["select_parts"]):
                        ids_p = np.asarray(
                            [
                                ii_new
                                for ii_new, ii in enumerate(ids[select_i])
                                if pp in keys[ii]
                            ]
                        )
                        if len(ids_p):
                            emb2_p = emb2[ids_p]
                            timesteps = np.asarray(
                                [int(keys[ids[ii]].split("_")[-2]) for ii in ids_p]
                            )
                            tt_sort = np.argsort(timesteps)
                            
                            for ii in range(len(axmin)):
                                if np.min(emb2_p[:,ii]) < axmin[ii]:
                                    axmin[ii] = np.min(emb2_p[:,ii])
                                if np.max(emb2_p[:,ii]) > axmax[ii]:
                                    axmax[ii] = np.max(emb2_p[:,ii])  

                            if "GALLOP" in cfg["dataset"]["name"]:


                                # ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                                test_colors = ['midnightblue','darkred','darkslategray']
                                gall_colors = ['#377eb8', '#ff7f00', '#4daf4a',]



                                colorp = gall_colors[ip]
                                colort = test_colors[ip]

                                ax.plot(
                                    emb2_p[tt_sort][:, 0],
                                    emb2_p[tt_sort][:, 1],
                                    emb2_p[tt_sort][:, 2],
                                    label=pp, lw=2, c = colorp
                                )
                                for tt in range(1, 12, 3):
                                    ax.text(
                                        emb2_p[tt_sort][tt, 0],
                                        emb2_p[tt_sort][tt, 1],
                                        emb2_p[tt_sort][tt, 2],
                                        str(tt), fontsize= 20, c=colort
                                    )

                            elif "TRUCK" in cfg["dataset"]["name"]:
                                cc = []
                                p_markers = []
                                for inn,ii in enumerate(ids_p):

                                    cmapbr1 = truncate_colormap(cmaps[(ip*2)%len(cmaps)], minval=0.3, maxval=1.0)
                                    cmapbr2 = truncate_colormap(cmaps[(ip*2+1)%len(cmaps)], minval=0.3, maxval=1.0)
                                    label1set = False
                                    label2set = False
                                    tt_i  = keys[ids[select_i][ii]].split("_")[3]
                                    
                                    if keys[ids[select_i][ii]].split("_")[0] in br1:
                                        #cc.append(colors_CB[ip * 2])
                                        cc.append(cmapbr1((float(tt_i)/5-first_tt) / (last_tt-first_tt)))
                                    else:
                                        #cc.append(colors_CB[ip * 2 + 1])
                                        cc.append(cmapbr2((float(tt_i)/5-first_tt) / (last_tt-first_tt)))
                                    
                                    if keys[ids[select_i][ii]].split("_")[0] in br1:
                                        p_markers.append(markers[ip * 2])
                                    else:
                                        p_markers.append(markers[ip * 2 + 1]) 
                                colors_for_legend = []
                                cc = np.array(cc)
                                for ii in range(len(ids_p)):
                                    if p_markers[ii] in colors_for_legend:
                                        ax.scatter(
                                            emb2_p[ii, 0], emb2_p[ii, 1], emb2_p[ii, 2], c=cc[ii].reshape(1,-1), 
                                            marker=p_markers[ii], s = 100, edgecolors= "black", linewidth=0.5
                                        )
                                    elif timesteps[ii] == (last_tt-10)*5:
                                        labels_text = ['Part 0, Branch B', 'Part 0, Branch A', 'Part 1, Branch B', 'Part 1, Branch A']
                                        labels_index = markers.index(p_markers[ii])
                                        colors_for_legend += [p_markers[ii]]
                                        ax.scatter(
                                            emb2_p[ii, 0], emb2_p[ii, 1], emb2_p[ii, 2], label=labels_text[labels_index], 
                                            c=cc[ii], marker=p_markers[ii], s = 100, edgecolors= "black", linewidth=0.5
                                        )
                                    else:
                                        ax.scatter(
                                            emb2_p[ii, 0], emb2_p[ii, 1], emb2_p[ii, 2], c=cc[ii].reshape(1,-1), 
                                            marker=p_markers[ii], s = 100, edgecolors= "black", linewidth=0.5
                                        )
                                    

                            elif "FAUST" in cfg["dataset"]["name"]:
                                cc = []
                                for ii in ids_p:
                                    cc_id = int(keys[ids[ii]].split("_")[2]) % 10
                                    cc.append(colors[cc_id])
                                ax.scatter(emb2_p[:, 0], emb2_p[:, 1], emb2_p[:, 2], c=cc)
                                for ind, tt_s in enumerate(tt_sort[:10]):
                                    cc_id = (
                                        int(keys[ids[select_i][ids_p[tt_s]]].split("_")[2])
                                        % 10
                                    )
                                    ax.scatter(
                                        emb2_p[tt_s, 0],
                                        emb2_p[tt_s, 1],
                                        label=faust_labels[cc_id],
                                        c=colors[cc_id],
                                    )

                            else:
                                ax.scatter(
                                    emb2_p[tt_sort][:, 0],
                                    emb2_p[tt_sort][:, 1],
                                    emb2_p[tt_sort][:, 2],
                                    label=pp,
                                )




                    if "TRUCK" in cfg["dataset"]["name"]:
                        import math
                        ax.view_init(elev=10, azim=130)
                        ax.set_xticks([math.ceil(10*axmin[0])/10,math.floor(10*axmax[0])/10])
                        ax.tick_params(axis="x", labelsize=16)
                        ax.set_yticks([math.ceil(10*axmin[1])/10,math.floor(10*axmax[1])/10])
                        ax.tick_params(axis="y", labelsize=16)
                        #ax.set_ylabel('time')
                        ax.set_zticks([math.ceil(10*axmin[2])/10,math.floor(10*axmax[2])/10])
                        ax.tick_params(axis="z", labelsize=16)
                        #ax.tick_params(axis='both', which='major', labelsize=10)
                        plt.legend(prop=font)  


                    else:
                        #ax.axis('off')
                        import math
                        ax.set_xticks([math.ceil(10*axmin[0])/10,math.floor(10*axmax[0])/10])
                        ax.tick_params(axis="x", labelsize=16)
                        ax.set_yticks([math.ceil(10*axmin[1])/10,math.floor(10*axmax[1])/10])
                        ax.tick_params(axis="y", labelsize=16)
                        #ax.set_ylabel('time')
                        ax.set_zticks([math.ceil(10*axmin[2])/10,math.floor(10*axmax[2])/10])
                        ax.tick_params(axis="z", labelsize=16)
                        ax.view_init(elev=17, azim=10)
                        plt.legend(prop=font)  

                    if saveplots:
                        plt.savefig('embedding_plots/3d_{}.png'.format(model_id))


            else:
                print("calculate_errors must be TRUE")


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap