from P2_create_dataset import main_step2, load_yaml
from P1_preprocess_data import main_step1
from dataset_load import load_h5_II, save_h5_II
from descriptif_data import plot_bilan

from os.path import join, dirname
from os import makedirs
from shutil import copyfile
from datetime import datetime
import numpy as np
import time


if __name__ == "__main__":
    path_final = "../dataset_/VMERGE_V8/"
    name_dataset = "dataset_AD_08200821_14Mas3Top3Phy_W15.hdf5"

    #### DESCENDING DATA ####
    yaml_param = "parameter/Y_dataset_parameter_DSC.yml"
    parameter = load_yaml(yaml_param)
    makedirs(parameter["ouput_dir"], exist_ok=True)
    now = datetime.now().strftime("%d%m%y_%HH%MM%S")
    copyfile(yaml_param, join(parameter["ouput_dir"], f"load_{now}.yml"))
    t = time.time()
    main_step1(**parameter)
    print("preprocess done in %s seconds" % (time.time() - t))
    t = time.time()
    main_step2(**parameter)
    print("create_dataset done in %s seconds" % (time.time() - t))

    #### ASCENDING DATA ####
    yaml_param2 = "parameter/Y_dataset_parameter_ASC.yml"
    parameter2 = load_yaml(yaml_param2)
    makedirs(parameter2["ouput_dir"], exist_ok=True)
    now = datetime.now().strftime("%d%m%y_%HH%MM%S")
    copyfile(yaml_param2, join(parameter2["ouput_dir"], f"load_{now}.yml"))
    t = time.time()
    main_step1(**parameter2)
    print("preprocess done in %s seconds" % (time.time() - t))
    t = time.time()
    main_step2(**parameter2)
    print("create_dataset done in %s seconds" % (time.time() - t))

    #### MERGE DATA ####
    path_asc = parameter2["ouput_dir"]
    path_desc = parameter["ouput_dir"]
    Xa, ya = load_h5_II(join(path_asc, "final_E4", "data_ASC_VX.h5"))
    Xd, yd = load_h5_II(join(path_desc, "final_E4", "data_DSC_VX.h5"))
    ya["metadata"] = np.hstack(
        [ya["metadata"], np.array(len(ya["metadata"]) * ["ASC"]).reshape(-1, 1)]
    )
    yd["metadata"] = np.hstack(
        [yd["metadata"], np.array(len(yd["metadata"]) * ["DSC"]).reshape(-1, 1)]
    )

    XX = np.vstack([Xa, Xd])
    yy = {
        "metadata": np.vstack([ya["metadata"], yd["metadata"]]),
        "topography": np.vstack([ya["topography"], yd["topography"]]),
        "physics": np.vstack([ya["physics"], yd["physics"]]),
    }
    yy["metadata"] = yy["metadata"].astype(np.string_)

    data_final = join(path_final, name_dataset)
    print(data_final)
    makedirs(path_final, exist_ok=True)
    save_h5_II(XX, yy, data_final)
    del XX, yy, Xa, ya, Xd, yd

    #### DESCRIPTIF ####
    XX, yy = load_h5_II(data_final)
    save_p = join(dirname(data_final), "bilan_dataset.pdf")
    print(save_p)
    plot_bilan([XX, yy], save_p)
