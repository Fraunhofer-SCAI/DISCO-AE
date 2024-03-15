import argparse
import yaml
import os
import torch
from faust_scape_dataset import FaustScapeDataset, shape_to_device
from gallop_dataset import GallopDataset
from model import GeomFMapNet
from utils import find_knn, zoomout_refine


def test_net(args):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path = os.path.join(base_path, cfg["dataset"]["root_train"])

    save_dir_name = f'saved_models_{cfg["dataset"]["name"]}'
    saved_maps = os.path.join(base_path, f"../data/{save_dir_name}/maps" + f"_{args.test_name}.pt")
    if not os.path.exists(os.path.join(base_path, f"../data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"../data/{save_dir_name}/"))

    # create dataset
    if cfg["dataset"]["name"] == "faust":
        train_dataset = FaustScapeDataset(dataset_path, name=cfg["dataset"]["name"], k_eig=cfg["fmap"]["k_eig"], train=False,
                                          n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False)
    elif cfg["dataset"]["name"] == "horse":
        train_dataset = GallopDataset(dataset_path, name=cfg["dataset"]["name"], k_eig=cfg["fmap"]["k_eig"], train=False,
                                      n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False)
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    fm_net = GeomFMapNet(cfg).to(device)
    fm_net.load_state_dict(torch.load(args.weights))
    n_fmap = cfg["fmap"]["n_fmap"]

    # Eval loop
    print("start evaluation...")
    to_save = {}

    fm_net.eval()
    for i, data in enumerate(train_loader):
        data = shape_to_device(data, device)
        evecs1, evecs2 = data["shape1"]["evecs"], data["shape2"]["evecs"]

        # do iteration
        C12, C21 = fm_net(data)
        C12, C21 = C12.squeeze(0), C21.squeeze(0)

        # maps from 2 to 1
        evec1_on_2 = evecs1[:, :n_fmap] @ C12.transpose(0, 1)
        _, pred_labels2to1 = find_knn(evecs2[:, :n_fmap], evec1_on_2, k=1, method='cpu_kd')
        map_21 = pred_labels2to1.flatten()

        # maps from 1 to 2
        evec2_on_1 = evecs2[:, :n_fmap] @ C21.transpose(0, 1)
        _, pred_labels1to2 = find_knn(evecs1[:, :n_fmap], evec2_on_1, k=1, method='cpu_kd')
        map_12 = pred_labels1to2.flatten()

        # zoomout refinement
        if args.do_zo:
            C12, C21, evecs1, evecs2 = C12.detach().cpu().numpy(), C21.detach().cpu().numpy(), evecs1.cpu().numpy(), evecs2.cpu().numpy()
            _, map_21_ref = zoomout_refine(evecs1, evecs2, C12, nit=args.num_zo_iters,
                                           step=(evecs1.shape[-1] - n_fmap) // args.num_zo_iters, return_p2p=True)
            _, map_12_ref = zoomout_refine(evecs2, evecs1, C21, nit=args.num_zo_iters,
                                           step=(evecs2.shape[-1] - n_fmap) // args.num_zo_iters, return_p2p=True)
        else:
            map_12_ref, map_21_ref = map_12, map_21

        to_save[f'{data["shape1"]["name"]}_{data["shape2"]["name"]}'] = [map_12, map_12_ref, map_21, map_21_ref]

    torch.save(to_save, saved_maps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of FMap model.")

    parser.add_argument("--config", type=str, default="faust", help="Config file name")
    parser.add_argument("--weights", type=str, default="NA", help="path to trained weights")
    parser.add_argument("--test_name", type=str, default="test1", help="name of the test run")
    parser.add_argument("--do_zo", type=bool, default=True, help="do zoomout refinement or not")
    parser.add_argument("--num_zo_iters", type=int, default=10, help="number of zoomout iterations")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"{args.config}.yaml", "r"))
    test_net(args)
