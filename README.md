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
    ./00_get_data.sh TRUCK
   ```

## Get Point-to-Point Maps 
To train the first stage in our pipeline, and extract the point-to-point maps, run the following command:

   ```sh
   cd stage1
   python 01_train_FM.py --config faust
   # once the training ends, run the following command to extract the point-to-point maps
   # weights are saved in the same folder as the data under the name "saved_models_DatasetName"
   python get_p2p.py --config faust --weights path/to/weights
   ```

To speed up the p2p map extraction calculate only the maps necessary to define the FMN for the shape collection. 
We recommend calculating p2p maps from every shape to the corresponding template shape and connecting the template shapes to each other.
To reduce the runtime even further, reduce the number of ZoomOut interations.
For best results, we recommend rotating the shapes properly (facing the same direction), and for the car components, normalizing using L1 distance and mean-centering.

## DISCO-autoencoder

Supervised DISCO-AE for GALLOP and FAUST dataset. The config file (faust-extra) trains the "unknown poses" experiment setup. The config file (faust-inter) trains the "unknown individuals" experiment setup. 
The paper explains the experiment setups in detail.

   ```sh
    python 02_train_network.py --config gallop
    python 02_train_network.py --config faust-extra
    python 02_train_network.py --config faust-inter
    python 02_train_network.py --config TRUCK_pall 
   ```

Unsupervised DISCO-AE for selected GALLOP and FAUST setups.

   ```sh
    python 02_train_network.py --config horse-unsup --loss_rec 10
    python 02_train_network.py --config faust-unsup-inter --loss_rec 10
   ```


| **Layer**          | **Output Shape**   | **Trainable** |
|--------------------|--------------------|---------------|
| Input              | (n, 3)             |               |
| DiffusionNet       | (n, nfeature)      | X             |
| ProjToLimitShape   | **(nB, nfeature)** |               |
| ProjToTemplateMesh | (m, nfeature)      |               |
| Append TemplateShape 3D-Coord | (m, nfeature+3)      |               |
| DiffusionNet       | (m, 3)             | X             |


## Citation

```markdown
@InProceedings{Hahner2024,
    author    = {Hahner, Sara and Attaiki, Souhaib and Garcke, Jochen and Ovsjanikov, Maks},
    title     = {Unsupervised Representation Learning for Diverse Deformable Shape Collections},
    booktitle = {Proceedings of the International Conference on 3D Vision (3DV 2024)},
    year      = {2024},
}
```
