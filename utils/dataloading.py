import numpy as np
import pickle
import os
import os.path as osp

from tqdm import tqdm

from pyFM.mesh import TriMesh
from pyFM.FMN import FMN
from pyFM import spectral


class simmesh(TriMesh):
    # extend the TriMesh class by variables:
    # - partid
    # - simid
    # - timestep: integer
    def __init__(
        self, path, partid, simid, timestep, area_normalize=True, center=False
    ):
        super().__init__(path, area_normalize=area_normalize, center=center)
        self.partid = partid  # id of the part -> corresponding FM will be calculated based on this
        self.simid = simid  # id of the simulation
        self.timestep = timestep


def dataloader_GALLOP(select_parts, select_tt, datadir="data", verbose=False, ndim=100):
    nshapes_all = 0
    meshes = []
    if ndim < 100:
        ndim = 100

    if verbose:
        print("Parts: ", end="")
    for part in select_parts:
        datapath = "Data_{}".format(part)

        if verbose:
            print(part, end=" ")
        shapes = [
            f.name
            for f in os.scandir(osp.join(datadir, datapath))
            if "off" in f.name
            and int(f.name.split("-")[2]) in select_tt
            and "checkpoints" not in f.name
        ]
        shapes.sort()

        nshapes_all += len(shapes)

        # load data
        cache = "{}/cache".format(datadir)
        if not osp.exists(cache):
            os.makedirs(cache)
        cache = "{}/cache/meshes_{}_{}_{}.pickle".format(
            datadir, datapath, select_tt, ndim
        )

        if osp.exists(cache):
            with open(cache, "rb") as handle:
                meshes_p = pickle.load(handle)

        else:
            meshes_p = []

            for sh in tqdm(shapes):
                sh_var = sh.split("-")
                partid = sh_var[0]
                timestep = int(sh_var[2])
                simid = sh_var[3].split(".")[0].split("_")[-1]

                mesh = simmesh(
                    osp.join(datadir, datapath, sh),
                    partid,
                    simid,
                    timestep,
                    area_normalize=True,
                    center=True,
                )
                meshes_p += [mesh]

                # By default does not use the intrinsic delaunay Laplacian
                mesh.process(k=ndim, intrinsic=False, verbose=False)

            with open(cache, "wb") as handle:
                pickle.dump(meshes_p, handle)

        meshes.extend(meshes_p)

    if verbose:
        print("imported {} Meshes".format(len(meshes)))

    return meshes


def dataloader_FAUST(select_parts, select_tt, datadir="data", verbose=False, ndim=100):
    nshapes_all = 0
    meshes = []
    if ndim < 100:
        ndim = 100

    if verbose:
        print("Parts: ", end="")
    for pn, part in enumerate(select_parts):
        datapath = "Data_{}".format(part)
        
        if 'SCAPE' in select_parts and 'FAUST' in select_parts:
            select_tt_tmp = select_tt[pn]
        else:
            select_tt_tmp = select_tt

        if verbose:
            print(part, end=" ")
        shapes = [
            f.name
            for f in os.scandir(osp.join(datadir, datapath))
            if "off" in f.name
            and int(f.name.split("_")[2].split("-")[0]) in select_tt_tmp
            and "checkpoints" not in f.name
        ]
        shapes.sort()

        nshapes_all += len(shapes)

        # load data
        cache = "{}/cache".format(datadir)
        if not osp.exists(cache):
            os.makedirs(cache)
        cache = "{}/cache/meshes_{}_{}_{}.pickle".format(
            datadir, datapath, len(select_tt_tmp), ndim
        )

        if osp.exists(cache):
            with open(cache, "rb") as handle:
                meshes_p = pickle.load(handle)

        else:
            meshes_p = []

            for sh in tqdm(shapes):
                partid = part
                print(partid)
                timestep = int(sh.split("_")[2].split("-")[0])
                simid = 0

                mesh = simmesh(
                    osp.join(datadir, datapath, sh),
                    partid,
                    simid,
                    timestep,
                    area_normalize=True,
                    center=True,
                )
                meshes_p += [mesh]

                # By default does not use the intrinsic delaunay Laplacian
                mesh.process(k=ndim, intrinsic=False, verbose=False)

            with open(cache, "wb") as handle:
                pickle.dump(meshes_p, handle)

        meshes.extend(meshes_p)

    if verbose:
        print("imported {} Meshes".format(len(meshes)))

    return meshes


def dataloader_TRUCK(select_parts, select_tt, datadir="data", verbose=False, ndim=100, order_p_s_t=True):
    # order_p_t_s: order frist by parts, then by timestep, then by simulation. If false: part, simulation, timestep
    nshapes_all = 0
    meshes = []
    if ndim < 100:
        ndim = 100

    for part in select_parts:
        datapath = "Data_{}".format(part)

        shapes = [
            f.name
            for f in os.scandir(osp.join(datadir, datapath))
            if "off" in f.name
            and int(f.name.split("_")[3]) in select_tt
            and "checkpoints" not in f.name
        ]

        if order_p_s_t:  # "part" in shapes[0]:
            print('order_p_s_t')
            # # order by simulation and then by timestep
            tts = np.asarray(
                [
                    int(sn.split("_")[3]) + 10000 * int(sn.split("_")[5].split(".")[0])
                    for sn in shapes
                ]
            )
            tts = np.argsort(tts)
            shapes = [shapes[tt] for tt in tts]
            # print(shapes)
        else:
            print('order_p_t_s')
            # # order by timestep and then by simulation
            tts = np.asarray(
                [
                    10000 * int(sn.split("_")[3]) + int(sn.split("_")[5].split(".")[0])
                    for sn in shapes
                ]
            )
            tts = np.argsort(tts)
            shapes = [shapes[tt] for tt in tts]
            # print(shapes)

        nshapes_all += len(shapes)

        # load data
        cache = "{}/cache".format(datadir)
        if not osp.exists(cache):
            os.makedirs(cache)
        if order_p_s_t:
            order_marker = 's_t'
        else:
            order_marker = 't_s'
        cache = "{}/cache/meshes_{}_{}_{}_{}.pickle".format(
            datadir, datapath, select_tt, ndim, order_marker
        )

        if osp.exists(cache):
            with open(cache, "rb") as handle:
                meshes_p = pickle.load(handle)

        else:
            meshes_p = []

            for sh in tqdm(shapes):
                sh_var = sh.split("_")
                partid = "part_" + sh_var[1]
                timestep = int(sh_var[3])
                simid = sh_var[5].split(".")[0]

                mesh = simmesh(
                    osp.join(datadir, datapath, sh),
                    partid,
                    simid,
                    timestep,
                    center=True,
                )
                meshes_p += [mesh]

                # By default does not use the intrinsic delaunay Laplacian
                mesh.process(k=ndim, intrinsic=False, verbose=False)

            with open(cache, "wb") as handle:
                pickle.dump(meshes_p, handle)

        meshes.extend(meshes_p)

        if verbose:
            print("Imported {} Meshes for part: {}".format(len(meshes_p), part))

    if verbose:
        print("Imported {} Meshes in total".format(len(meshes)))

    return meshes

def dataloader_syn(select_parts, corners, select_tt, datadir="data", verbose=False, ndim=100):
    nshapes_all = 0
    meshes = []
    if ndim < 100:
        ndim = 100

    if verbose:
        print("Parts: ", end="")
    for part in select_parts:
        datapath = "Data_{}_{}".format(part, corners)

        if verbose:
            print(part, end=" ")
        shapes = [
            f.name
            for f in os.scandir(osp.join(datadir, datapath))
            if "off" in f.name
            and int(f.name.split("_")[1][:-4]) in select_tt
            and "checkpoints" not in f.name
        ]
        shapes.sort()
        
        nshapes_all += len(shapes)

        # load data
        cache = "{}/cache".format(datadir)
        if not osp.exists(cache):
            os.makedirs(cache)
        cache = "{}/cache/meshes_{}_{}_{}.pickle".format(
            datadir, datapath, len(select_tt), ndim
        )

        if osp.exists(cache):
            print('load data from cache:', cache)
            with open(cache, "rb") as handle:
                meshes_p = pickle.load(handle)

        else:
            meshes_p = []

            for sh in tqdm(shapes):
                partid = part
                timestep = int(sh.split("_")[1][:-4])
                simid = 0

                mesh = simmesh(
                    osp.join(datadir, datapath, sh),
                    partid,
                    simid,
                    timestep,
                    area_normalize=True,
                    center=True,
                )
                meshes_p += [mesh]

                # By default does not use the intrinsic delaunay Laplacian
                mesh.process(k=ndim, intrinsic=False, verbose=False)
            
            print('loaded {} meshes'.format(len(meshes_p)))

            with open(cache, "wb") as handle:
                pickle.dump(meshes_p, handle)

        meshes.extend(meshes_p)

    if verbose:
        print("imported {} Meshes".format(len(meshes)))
        
    return meshes

def add_edges_ij(FM_meshes, meshi, meshj, samples_p2p, ii, jj, template_ind=None):
    # add edges (j_id, i_id) and (i_id, j_id) if the map is given.
    # Else connect to template shape
    i_pid = meshi.partid
    j_pid = meshj.partid
    i_tid = meshi.timestep
    j_tid = meshj.timestep
    if 'triangle' in i_pid:
        i_tid = 0
        j_tid = 0
        i_pid = 'triangle'
        j_pid = 'triangle'
    i_sid = meshi.simid
    j_sid = meshj.simid
    i_id = "{}_{}_{}".format(i_sid, i_pid, i_tid)
    j_id = "{}_{}_{}".format(j_sid, j_pid, j_tid)

    failed = False

    # the p2p maps are always the other way around
    if (ii, jj) not in FM_meshes.p2p:
        if (j_id, i_id) in samples_p2p:
            FM_meshes.p2p[(ii, jj)] = samples_p2p[(j_id, i_id)]
            FM_meshes.edges += [(ii, jj)]
        else:
            # (j_id, i_id) not in samples_p2p
            print((j_id, i_id), ' not in samples_p2p ')
            # if this edge doesnt exist connect to the template shape
            failed = True

    # add also the edge in the opposite direction
    if (jj, ii) not in FM_meshes.p2p:
        if (i_id, j_id) in samples_p2p:
            FM_meshes.p2p[(jj, ii)] = samples_p2p[(i_id, j_id)]
            FM_meshes.edges += [(jj, ii)]
        else:
            # print( (i_id, j_id), ' not in samples_p2p ')
            failed = True

    return FM_meshes, failed


def def_FMN_from_p2p(
    meshes, select_parts, template_tt, samples_p2p, maxsize_circle=None, verbose=False
):
    FM_meshes = FMN(meshes)

    FM_meshes.p2p = dict()
    FM_meshes.edges = []
    FM_meshes.maps = dict()

    # # get maps across the shapes of interest, use loaded p2p maps for this

    # save position of meshes from simulation edge_sim at t=0
    edge_sim = meshes[0].simid
    print('template simulations', edge_sim)
    template_sim_id = []

    tmp = 0
    for part in select_parts:

        # a circle for every part
        for ii in range(tmp, len(meshes)):
            meshi = meshes[ii]
            
            if meshi.partid != part:
                # part completed
                if verbose:
                    print(
                        "  Add edges for {} samples of part {}".format(ii - tmp, part)
                    )
                break
            #print(meshi.simid, edge_sim, meshi.timestep, template_tt)
            if meshi.simid == edge_sim and meshi.timestep == template_tt:
                # keep track of one template shape per part
                template_sim_id += [ii]
                print("template mesh found", template_sim_id)

            if isinstance(maxsize_circle, str):
                if maxsize_circle == "star":
                    if len(template_sim_id):
                        jj = template_sim_id[-1]
                        FM_meshes, _ = add_edges_ij(
                            FM_meshes, meshi, meshes[jj], samples_p2p, ii, jj
                        )

            else:
                # add edge to the next shape
                jj = ii + 1
                if jj >= len(meshes):
                    jj = tmp  # got to last mesh, close the partwise circle in network
                elif meshes[jj].partid != part:
                    jj = tmp  # got to last mesh of part, close the partwise circle in network

                # # in both directions add the edge
                FM_meshes, failed = add_edges_ij(
                    FM_meshes, meshi, meshes[jj], samples_p2p, ii, jj
                )

                # connect to the last template shape if edge is missing
                if failed:
                    if len(template_sim_id):
                        #if failed: Add edges to template shape because missing p2p map in order to have a connected graph
                        jj = template_sim_id[-1]
                        FM_meshes, _ = add_edges_ij(
                            FM_meshes, meshi, meshes[jj], samples_p2p, ii, jj
                        )
                # connect every maxsize_circle-th shape to the last template shape
                elif maxsize_circle is not None:
                    if ii % maxsize_circle == 0 and len(template_sim_id):
                        #if failed: Add edges to template shape because missing p2p map in order to have a connected graph
                        jj = template_sim_id[-1]
                        FM_meshes, _ = add_edges_ij(
                            FM_meshes, meshi, meshes[jj], samples_p2p, ii, jj
                        )


            if ii == len(meshes) - 1 and verbose:
                print(
                    "  Add edges for {} samples of part {}".format(ii - tmp + 1, part)
                )
        tmp = ii

    ii = template_sim_id[0]
    for jj in template_sim_id:
        if ii == jj:
            continue

        FM_meshes, _ = add_edges_ij(FM_meshes, meshes[ii], meshes[jj], samples_p2p, ii, jj)

    if verbose:
        print(
            "Add a circle for every part (P2P-correcpondence) and connect every {}-th parts to template shape.\
                Always with edges in both directions.Resulting number of edges: ".format(
                maxsize_circle
            ),
            end="",
        )
        print(len(FM_meshes.edges))
        # print('2^( #parts - 1 ) + 2 * #meshes = {}'.format(2**(len(template_sim_id)-1) + len(meshes)*2))

    return FM_meshes, template_sim_id


def compute_new_maps(FM_meshes):
    """
    Convert pointwise maps into Functional Maps of size M.

    Parameters
    ------------------------
    M : int - size of the functional map to compute
    """

    for i, j in FM_meshes.edges:
        sub = None

        if (i, j) in FM_meshes.p2p:
            FM = spectral.mesh_p2p_to_FM(
                FM_meshes.p2p[(i, j)],
                FM_meshes.meshlist[i],
                FM_meshes.meshlist[j],
                dims=FM_meshes.M,
                subsample=sub,
            )
            FM_meshes.maps[(i, j)] = FM

    # Reset map-dependant variables
    FM_meshes.p2p = None  # Dictionnary of pointwise
    return FM_meshes


def extend_FMN_by_p2p(
    FM_meshes, meshes_ext, select_parts, template_sim_id, samples_p2p, verbose=False
):
    meshes = FM_meshes.meshlist
    nshapes_base = FM_meshes.n_meshes

    if FM_meshes.p2p is None:
        FM_meshes.p2p = dict()

    FM_meshes.meshlist.extend(meshes_ext)

    template_parts = []
    for template_id in template_sim_id:
        template_parts += [meshes[template_id].partid]
    print("Template parts:", template_parts)

    tmp = 0
    for part in select_parts:
        # connect new shapes to corresponding template shape
        for ii in range(tmp, len(meshes_ext)):
            meshi = meshes_ext[ii]
            if meshi.partid != part:
                if verbose:
                    print(
                        "  Add edges for {} samples of part {}".format(ii - tmp, part)
                    )
                break

            # find corresponding template shape and add edges in both directions
            jj = template_sim_id[template_parts.index(part)]

            FM_meshes, _ = add_edges_ij(
                FM_meshes, meshi, meshes[jj], samples_p2p, nshapes_base + ii, jj
            )

            if ii == len(meshes_ext) - 1 and verbose:
                print(
                    "  Add edges for {} samples of part {}".format(ii - tmp + 1, part)
                )

        # next part
        tmp = ii

    # # calculate new functional maps
    FM_meshes = compute_new_maps(FM_meshes)

    # # get the Y for test shapes using FM to template shape
    for ii in range(nshapes_base, nshapes_base + len(meshes_ext)):
        meshi = FM_meshes.meshlist[ii]
        # find corresponding template shape
        temp = template_sim_id[template_parts.index(meshi.partid)]
        # # calculate Y via FM to template shape
        FM_meshes.CCLB = np.append(
            FM_meshes.CCLB, [FM_meshes.maps[(temp, ii)] @ FM_meshes.CCLB[temp]], axis=0
        )

    if verbose:
        print(
            "Add an edge for every new part to the corresponding template shape (P2P-correcpondence). Always with edges in both directions. "
        )

    return FM_meshes
