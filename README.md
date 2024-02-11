# DISCO-AE
Autoencoder for Diverse 3D Shape Collections (DISCO)

**[Paper](https://arxiv.org/abs/2310.18141):** Unsupervised Representation Learning for Diverse Deformable Shape
Collections

**Authors:** Sara Hahner⋆, Souhaib Attaiki⋆, Jochen Garcke, Maks Ovsjanikov 

⋆: Equal contribution

**3D Vision Conference 2024**

## Packages

- pytorch (Tested with Pytorch 1.13 and CUDA 11.6)
- tqdm
- scikit-learn
- scipy
- igl 
- matplotlib
- plotly
- meshplot
- robust-laplacian 
- potpourri3d 
- trimesh

Source-Code:
- [pyFM](https://github.com/RobinMagnet/pyFM/tree/master/pyFM): This package provides the functions to calculate the Functional Maps and to set up the Functional Maps Network. Since we made small changes to the mesh class in an earlier version of pyFM, the code can be found in directory [pyFM](https://github.com/Fraunhofer-SCAI/DISCO-AE/tree/main/pyFM).
- [DiffusionNet](https://github.com/nmwsharp/diffusion-net): The implementation can be found in directory [diffusion_net](https://github.com/Fraunhofer-SCAI/DISCO-AE/tree/main/diffusion_net).

## Load Data

Download data and p2p maps: 

   ```sh
    ./00_get_data.sh gallop
    ./00_get_data.sh FAUST
   ```

   
## Get Point-to-Point Maps 
To train the first stage in our pipeline, and extract the point-to-point maps, run the following command:

   ```sh
   cd stage1
   python 01_train_FM.py --config faust
   ```


## DISCO-autoencoder

Supervised DISCO-AE for GALLOP and FAUST dataset. The config file (faust-extra) trains the "unknown poses" experiment setup. The config file (faust-inter) trains the "unknown individuals" experiment setup. 
The paper explains the experiment setups in detail.

   ```sh
    python 02_train_network.py --config gallop
    python 02_train_network.py --config faust-extra
    python 02_train_network.py --config faust-inter
   ```

Unsupervised DISCO-AE for selected GALLOP and FAUST setups.

   ```sh
    python 02_train_network.py --config horse-unsup
    python 02_train_network.py --config faust-unsup-inter
   ```


| **Layer**          | **Output Shape**   | **Trainable** |
|--------------------|--------------------|---------------|
| Input              | (n, 3)             |               |
| DiffusionNet       | (n, nfeature)      | X             |
| ProjToLimitShape   | **(nB, nfeature)** |               |
| ProjToTemplateMesh | (m, nfeature)      |               |
| Append TemplateShape 3D-Coord | (m, nfeature+3)      |               |
| DiffusionNet       | (m, 3)             | X             |

