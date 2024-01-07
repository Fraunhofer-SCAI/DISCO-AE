#  # imports
import argparse
import yaml

import os
import os.path as osp
from tqdm import tqdm
from utils.FMN_Dataset import FMN_Dataset, collate_fn, shape_to_device
from utils.model_eval_functions import model_id_gen, FMN_id_gen, get_emb_size, def_model, eval_yalm, save_pred_emb_and_loss

from utils.model_eval_functions import eval_network
import time

import pickle

import torch
import random

import torch.profiler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

torch.backends.cudnn.deterministic = True


def train_network(cfg):
    ###################
    # # PREPARATION # #
    ###################

    # model reference
    model_id = model_id_gen(cfg)
    print("model reference:", model_id)

    # gpu or cpu
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device("cuda:{}".format(cfg["misc"]["device"]))
    else:
        device = torch.device("cpu")

    print("device", device)

    # get some important paths
    base_path = osp.dirname(__file__)
    cache_dir = osp.join(base_path, cfg["dataset"]["cache_dir"])
    cache_FMN = osp.join(cache_dir, FMN_id_gen(cfg))

    p2p_maps_path = osp.join(base_path, cfg["dataset"]["p2p_maps"])

    save_dir_name = "trained_{}".format(model_id)
    model_save_path = osp.join(base_path, "data", save_dir_name, "ep" + "_{}.pth")
    if not osp.exists(osp.join(base_path, "data", save_dir_name)):
        os.makedirs(osp.join(base_path, "data", save_dir_name))

    log_path = os.path.join(base_path, "data", save_dir_name, "log.txt")
    log_file = open(log_path, "w")

    # initializing writer for TensorBoard
    log_dir = "logs/fit/" + "{}_{}".format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), model_id
    )
    writer = SummaryWriter(log_dir)

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

    print("Loaded training dataset. {} shapes", train_dataset.FM_meshes.n_meshes)

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
    if cfg["training"]["batch_size"] > 1:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            collate_fn=collate_fn,
            shuffle=True,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True
        )

    # define model
    ae_network = def_model(cfg, device)

    lr = float(cfg["training"]["lr"])

    optimizer = torch.optim.Adam(
        ae_network.parameters(),
        lr=lr,
        betas=(cfg["training"]["b1"], cfg["training"]["b2"]),
    )
    criterion = torch.nn.MSELoss(reduction="sum").to(device)

    # some model and training stats for log file

    log_file.write(model_id + "\n\n")

    pytorch_total_params = sum(
        p.numel() for p in ae_network.parameters() if p.requires_grad
    )

    txt3 = "nCCLB = {}, DiffNet-nfeature = {}".format(
        cfg["fmnet"]["nCCLB"], cfg["diffnet"]["nfeature"]
    )
    txt1 = "Number of Parameters: {:,}".format(pytorch_total_params)
    # get embedding size
    txt2 = get_emb_size(cfg)
    print(txt1 + "\n" + txt2 + "\n" + txt3)
    log_file.write(txt1 + "\n" + txt2 + "\n" + txt3 + "\n")
    print("Training Samples:", len(train_loader))

    log_file.write("Training Samples: {:,}".format(len(train_loader)) + "\n\n")

    ##############
    # # TRAINING ##
    ##############

    collect_loss = []

    # Training loop
    print("start training")
    iterations = 0
    tmp_time = time.time()
    for epoch in tqdm(range(1, cfg["training"]["epochs"] + 1)):
        if epoch % cfg["training"]["decay_iter"] == 0:
            lr *= cfg["training"]["decay_factor"]
            print("Decaying learning rate, new one: {}".format(lr))
            log_file.write(" Decaying learning rate, new one: {}\n".format(lr))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        ae_network.train()

        avg_loss = 0

        for i, data in enumerate(train_loader):
            data = shape_to_device(data, device)

            batch_size = data["shape"]["xyz"].shape[0]

            # data augmentation

            # do iteration
            optimizer.zero_grad()

            recon = ae_network(data)

            # p2p map for error calculation
            # compare errors on template shape
            loss = 0
            for bs in range(batch_size):

                xyz_true_on_template = torch.gather(
                    data["shape"]["xyz"][bs],
                    0,
                    data["shape"]["p2p_temp"][bs].repeat(3, 1).T,
                )

                if cfg["loss_part"]:
                    if epoch > 100:
                        loss_small = (xyz_true_on_template - recon[bs]).square().sum(axis=1)
                        loss_small, _ = torch.sort(loss_small)
                        loss += loss_small[: int(0.8 * loss_small.shape[0])].mean()
                    else:
                        loss += (
                            criterion(xyz_true_on_template, recon[bs]) / recon[bs].shape[0]
                        )  # divide by number of vertices
                else:
                    loss += criterion(xyz_true_on_template, recon[bs]) / recon[bs].shape[0]

                if cfg["loss_rec"]:
                    map_template = data["shape"]["p2p_temp"][bs].to(xyz_true_on_template.device)
                    recon = recon[bs, map_template]
                    if xyz_true_on_template.shape[0] > 25000:
                        idx = torch.randperm(xyz_true_on_template.shape[0])[:20000]
                        xyz_true_on_template = xyz_true_on_template[idx]
                        recon = recon[idx]

                    matrix1 = torch.cdist(xyz_true_on_template, xyz_true_on_template)
                    matrix2 = torch.cdist(recon, recon)
                    loss += (matrix1 - matrix2).square().mean() * cfg["loss_rec"]

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        # log file
        iterations += 1

        log_iter = (iterations + 1) % cfg["misc"]["log_interval"] == 0
        if log_iter:
            txt1 = "#epoch:{}".format(epoch)
            txt2 = "#MSE-loss:{:10.8f}".format(avg_loss / len(train_loader))
            txt3 = "#Runtime:{:10.8f}".format(time.time() - tmp_time)
            tmp_time = time.time()
            log_file.write(txt1 + "\n" + txt2 + "\n" + txt3 + "\n")
            # print all losses

        # adding running loss to tensorboard
        writer.add_scalar("Loss/train", avg_loss / len(train_loader), epoch)
        collect_loss.append(avg_loss / len(train_loader))

        # save model
        if (epoch) % cfg["misc"]["checkpoint_interval"] == 0 or epoch == cfg[
            "training"
        ]["epochs"]:
            torch.save(ae_network.state_dict(), model_save_path.format(epoch))

    ################
    # # EVALUATION ##
    ################

    # # calculate errors and save the predictions

    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    ae_network.eval()

    print("\n-------\nRESULTS\n-------")

    print("\nTesting  Samples:", len(test_loader))
    log_file.write(
        "\n\n-------\nRESULTS\n-------\n\nTesting  Samples: {:,}".format(
            len(test_loader)
        )
        + "\n\n"
    )

    cache_dir_results = os.path.join(base_path, "data", save_dir_name, "cache")
    os.makedirs(cache_dir_results, exist_ok=True)
    load_cache_test = os.path.join(cache_dir_results, "test.pickle")
    load_cache_train = os.path.join(cache_dir_results, "train.pickle")

    with torch.no_grad():
        # test errors
        collect_test_loader, avg_loss, avg_limshape_loss = save_pred_emb_and_loss(
            ae_network, cfg, test_loader, criterion, shape_to_device, device
        )
        txt1a = "#final limitshape test  MSE-loss: {:06.6f}".format(
            avg_limshape_loss / len(test_loader)
        )

        print("Test Errors")
        print("LimitShape-Loss: {:08.8f}".format(avg_limshape_loss / len(test_loader)))
        print("Network-Loss   : {:08.8f}".format(avg_loss / len(test_loader)))

        txt1a = "final limitshape test  MSE-loss ALL:       {:08.8f}".format(
            avg_limshape_loss / len(test_loader)
        )
        txt2a = "final test  MSE-loss ALL     :       {:08.8f}".format(
            avg_loss / len(test_loader)
        )

        # train errors
        collect_train_loader, avg_loss, avg_limshape_loss = save_pred_emb_and_loss(
            ae_network, cfg, train_loader, criterion, shape_to_device, device
        )
        txt1b = "#final limitshape train  MSE-loss: {:06.6f}".format(
            avg_limshape_loss / len(train_loader)
        )

        print("Train Errors")
        print("LimitShape-Loss: {:08.8f}".format(avg_limshape_loss / len(train_loader)))
        print("Network-Loss   : {:08.8f}".format(avg_loss / len(train_loader)))

        txt1b = "final limitshape train MSE-loss ALL:       {:08.8f}".format(
            avg_limshape_loss / len(train_loader)
        )
        log_file.write("Erros using area normalization. not for paper.\n")
        log_file.write("\n" + txt1a + "\n" + txt1b + "\n" + "--------" + "\n")

        txt2b = "final train MSE-loss ALL     :       {:08.8f}".format(
            avg_loss / len(train_loader)
        )
        log_file.write(txt2a + "\n" + txt2b + "\n")

        # save calculated results
        with open(load_cache_test, "wb") as handle:
            pickle.dump(collect_test_loader, handle)
        with open(load_cache_train, "wb") as handle:
            pickle.dump(collect_train_loader, handle)

        # close log file
        log_file.close()
        writer.close()

        return ae_network, log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch the training of DiffNet Limit Shape Autoencoder model."
    )

    parser.add_argument("--config", type=str, default="horse", help="Config file name")
    parser.add_argument("--loss_rec", type=int, default=0, help="Config file name")
    parser.add_argument("--loss_part", type=int, default=0, help="Config file name")

    args = parser.parse_args()

    filename = osp.join('.', 'config', '{}.yalm'.format(args.config))

    if osp.isfile(filename):
        cfg = yaml.safe_load(open(filename, "r"))

        cfg["loss_rec"] = args.loss_rec
        cfg["loss_part"] = args.loss_part
                
        cfg = eval_yalm(cfg)
        print(cfg)

        torch.manual_seed(cfg["training"]["seed"])
        random.seed(cfg["training"]["seed"])

        trained_network, log_path = train_network(cfg)
        start = time.time()
        log_file = open(log_path, "a")
        eval_network(trained_network, cfg, calculate_errors=False, calculate_errors_p2s=True,
                     plot_reconstructions=False, plot_embeddings=False, use_cache=True,
                     logfile=log_file)
        print('Eval time:', time.time() - start, 'seconds')
        log_file.close()

    else:
        print("config file {} does not exist".format(filename))
