from joblib import Parallel, delayed
from os import makedirs
from os.path import join, basename, dirname
import glob, pickle
from tqdm import tqdm
from datetime import datetime
from shutil import copyfile

from dataset_load import save_h5_II, load_h5_II
from descriptif_data import plot_bilan
from geo_tools import load_data
from img_processing_II import SAR_patch_clean
from labelling_tools import *
from yaml import safe_load


def load_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def load_yaml(file_name):
    with open(file_name, "r") as f:
        opt = safe_load(f)
    return opt


def onedate2patchslabel(i, tel, windows_size, step, start=0, outdir=None, type_d="DSC"):
    x, _ = load_data(i)
    patchs = np.array(SAR_patch_clean(x, windows_size, step, start), dtype=np.float32)
    X, y = add_labels_II(patchs, i, tel, type_d)
    if outdir:
        np.save(join(outdir, basename(i)[:-4] + ".npy"), X)
        np.save(join(outdir, basename(i)[:-4] + "_label.npy"), y)
    return X, y


def create_dataset(
    path, output_dir, folders, csv_GT, winsize, step, type_d, workers, save=True
):
    print("================== Create Dataset ==================")
    print("crop original images and select the non-nan results")
    print("====================================================")
    start = step[0]
    step = step[1]
    output_path_save = join(output_dir, "final_E4")
    makedirs(output_path_save, exist_ok=True)
    out_temp = join(output_dir, "temp")
    makedirs(out_temp, exist_ok=True)

    for folder in folders[:2]:
        input_path = join(path, folder, "*.tif")
        tel = [load_pkl(csv_GT[m]) for m in range(len(csv_GT))]
        extraction = Parallel(n_jobs=workers)(
            delayed(onedate2patchslabel)(i, tel, winsize, step, start, out_temp, type_d)
            for i in tqdm(glob.glob(input_path), position=0, leave=False)
        )

        X, Y = select_non_empty_list(extraction)

        print("X :", X.shape, " Y :", Y.keys(), len(Y[list(Y.keys())[0]]))
        print("=============== Done ===============")

        if save:
            print("=============== Save ===============")
            print(f"save dataset {folder} to hdf5 file")
            print("===================================")
            output_path_z = join(output_path_save, f"data_{type_d}_VX.h5")
            save_h5_II(X, Y, output_path_z)
            print("============== Done ===============")
    return 1


def main_step2(**kwargs):
    ouput_dir = kwargs["ouput_dir"]
    folders = kwargs["folders"]
    csv_GT = kwargs["csv_GT"]
    patchs_size = kwargs["patchs_size"]
    step = kwargs["step"]
    save = kwargs["save"]
    ouput_stor = kwargs["ouput_stor"]
    workers = kwargs["workers"]
    input_path = join(ouput_stor, "pre_process_E0")
    type_d = kwargs["type_d"]

    create_dataset(
        input_path,
        ouput_dir,
        folders,
        csv_GT,
        patchs_size,
        step,
        type_d,
        workers,
        save,
    )


if "__main__" == __name__:
    path_final = "../dataset_/VMERGE_V8/"
    name_dataset = "dataset_AD_08200821_14Mas3Top3Phy_W15.hdf5"

    #### DESCENDING DATA ####
    yaml_param = "parameter/Y_dataset_parameter_DSC.yml"
    parameter = load_yaml(yaml_param)
    makedirs(parameter["ouput_dir"], exist_ok=True)
    now = datetime.now().strftime("%d%m%y_%HH%MM%S")
    copyfile(yaml_param, join(parameter["ouput_dir"], f"load_{now}.yml"))
    main_step2(**parameter)

    #### ASCENDING DATA ####
    yaml_param2 = "parameter/Y_dataset_parameter_ASC.yml"
    parameter2 = load_yaml(yaml_param2)
    makedirs(parameter2["ouput_dir"], exist_ok=True)
    now = datetime.now().strftime("%d%m%y_%HH%MM%S")
    copyfile(yaml_param2, join(parameter2["ouput_dir"], f"load_{now}.yml"))
    main_step2(**parameter2)

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
