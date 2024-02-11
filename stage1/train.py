import argparse
import yaml
import os
import torch
from faust_scape_dataset import FaustScapeDataset, shape_to_device
from model import FMLoss, GeomFMapNet
from utils import augment_batch


def train_net(args):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path = os.path.join(base_path, cfg["dataset"]["root_train"])

    save_dir_name = f'saved_models_{cfg["dataset"]["name"]}'
    model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))

    # create dataset
    if cfg["dataset"]["name"] == "faust":
        train_dataset = FaustScapeDataset(dataset_path, name=cfg["dataset"]["name"], k_eig=cfg["fmap"]["k_eig"],
                                          n_fmap=cfg["fmap"]["n_fmap"], use_cache=True, op_cache_dir=op_cache_dir)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
    else:
        raise NotImplementedError("dataset not implemented!")

    # define model
    fm_net = GeomFMapNet(cfg).to(device)
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(fm_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = FMLoss(w_bij=cfg["loss"]["w_bij"], w_ortho=cfg["loss"]["w_ortho"]).to(device)

    # Training loop
    print("start training")
    iterations = 0
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if epoch % cfg["optimizer"]["decay_iter"] == 0:
            lr *= cfg["optimizer"]["decay_factor"]
            print(f"Decaying learning rate, new one: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        fm_net.train()
        for i, data in enumerate(train_loader):
            data = shape_to_device(data, device)

            # data augmentation
            data = augment_batch(data, rot_x=0, rot_y=0, rot_z=0, std=0.01, noise_clip=0.00, scale_min=1.0, scale_max=1.0)

            # do iteration
            C12, C21 = fm_net(data)
            loss = criterion(C12, C21)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            iterations += 1
            if iterations % cfg["misc"]["log_interval"] == 0:
                print(f"#epoch:{epoch}, #batch:{i + 1}, #iteration:{iterations}, loss:{loss}")

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(fm_net.state_dict(), model_save_path.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of FMap model.")

    parser.add_argument("--config", type=str, default="faust", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"{args.config}.yaml", "r"))
    train_net(cfg)
