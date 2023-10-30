from os.path import join, basename, dirname, abspath
from os import remove, rename, makedirs
from joblib import Parallel, delayed
from shutil import rmtree
from tqdm import tqdm
from osgeo import ogr
import numpy as np
import glob

from geo_tools import (
    gdal_clip_shp_raster,
    gdal_merge_rasters,
    check_data,
    check_ResProj_files,
    apply_function,
    mask_layover,
)


def clip_shp(inraster, shp, stations_id, ouput_path, add_aux):
    if add_aux:
        outraster = join(
            ouput_path, stations_id + "_" + basename(inraster)[:6] + ".tif"
        )
    else:
        outraster = join(
            ouput_path, stations_id + "_" + extract_date(basename(inraster)) + ".tif"
        )
    gdal_clip_shp_raster(inraster, shp, outraster, stations_id)


def clip_multiple_shp(
    data_path, name_shp, stations_id, ouput_path, add_aux, workers=-1
):
    Parallel(n_jobs=workers)(
        delayed(clip_shp)(inraster, name_shp, stations_id, ouput_path, add_aux)
        for inraster in glob.glob(data_path)
    )


def select_data(data_path, ouput_path, shptr_path, add_aux, workers):
    name_shp = abspath(glob.glob(shptr_path)[0])
    ds = ogr.Open(name_shp)
    lyr = ds.GetLayer(0)
    lyr.ResetReading()
    ft = lyr.GetNextFeature()
    while ft:
        stations_id = ft.GetFieldAsString("id")
        clip_multiple_shp(
            data_path, name_shp, stations_id, ouput_path, add_aux, workers
        )
        ft = lyr.GetNextFeature()


def data_clipping(data_path, shp_path, ouput_dir, folders, add_aux, workers):
    for folder in folders:
        ouput_path = join(ouput_dir, "pre_process_E0", folder)
        makedirs(ouput_path, exist_ok=True)
        shptr_path = shp_path + f"{folder}/*.shp"
        select_data(data_path, ouput_path, shptr_path, add_aux, workers)


def data_merge(ouput_dir, folders, typeaux):
    for folder in folders:
        ouput_path = join(ouput_dir, "pre_process_E0", folder, "*.tif")
        list_tif = glob.glob(ouput_path)
        list_tif.sort()
        for i in list_tif:
            sta = extract_stations_id(i)
            if check_current_file_not_aux(i, sta, typeaux):
                exist_aux, list_aux = check_exist_aux_files(i, list_tif, sta, typeaux)
                if exist_aux:
                    in_rast = i
                    m = "#"
                    for l_aux in list_aux:
                        out_rast = dirname(i) + f"/{sta}_{m}_temp.tif"
                        gdal_merge_rasters(in_rast, l_aux, out_rast)
                        remove(in_rast)
                        in_rast = out_rast
                        m += "#"
                    rename(in_rast, i)
        for i in list_tif:
            sta = extract_stations_id(i)
            if not (check_current_file_not_aux(i, sta, typeaux)):
                remove(i)


def check_exist_aux_files(i_path, listfiles, station, typeaux):
    cond0 = np.array(
        [(dirname(i_path) + f"/{station}_{j}.tif") in listfiles for j in typeaux]
    ).all()
    cond1 = np.array(
        [(dirname(i_path) + f"\{station}_{j}.tif") in listfiles for j in typeaux]
    ).all()
    cond2 = cond0 | cond1
    if cond2:
        if cond0:
            return cond2, [(dirname(i_path) + f"/{station}_{j}.tif") for j in typeaux]
        else:
            return cond2, [(dirname(i_path) + f"\{station}_{j}.tif") for j in typeaux]
    else:
        return cond2, []


def check_current_file_not_aux(i_path, station, typeaux):
    cond = np.array(
        [
            abspath(dirname(i_path) + f"/{station}_{j}.tif") != abspath(i_path)
            for j in typeaux
        ]
    ).all()
    return cond


def extract_date(a):
    return a[-19:-11]


def extract_stations_id(i_path):
    loc = basename(i_path).find("_")
    sta = basename(i_path)[:loc]
    return sta


def transformation_data(ouput_dir, folders, dic_bands, workers):
    for folder in folders:
        ouput_path = join(ouput_dir, "pre_process_E0", folder, "*.tif")
        list_tif = glob.glob(ouput_path)
        Parallel(n_jobs=workers)(
            delayed(apply_function)(i, dic_bands) for i in tqdm(list_tif)
        )
    return 1


def preprocess(
    aux_path,
    data_path,
    shp_path,
    mask_path,
    ouput_dir,
    folders,
    dic_bands,
    typeaux=["DEM10M", "EXP10M", "PEN10M"],
    add_aux=True,
    mask_data=True,
    workers=-1,
):
    print("================== Preprocessing S1 DATA ==================")
    if mask_data:
        print("---------------------------")
        print("masking data of the layover")
        data_path = mask_layover(data_path, mask_path)
    if add_aux:
        print("----------------------------------------------------------")
        print("checking data and auxiliary data resolution and projection")
        valid = check_ResProj_files(aux_path, data_path)
    else:
        print("---------------------------------------")
        print("checking data resolution and projection")
        valid = np.array(check_data(data_path)).size > 1
    if valid:
        print("-------------")
        print("clipping data")
        aux_proc = False
        data_clipping(data_path, shp_path, ouput_dir, folders, aux_proc, workers=24)
        if add_aux:
            print("-----------------------")
            print("clipping auxiliary data")
            data_clipping(aux_path, shp_path, ouput_dir, folders, add_aux, workers=24)
            print("-------------------------------")
            print("merging data and auxiliary data")
            data_merge(ouput_dir, folders, typeaux)
    if mask_data:
        print("----------------------------")
        print("removing temporary directory")
        rmtree(dirname(data_path))
    if dic_bands != None:
        print("----------------------")
        print("transformation of data")
        transformation_data(ouput_dir, folders, dic_bands, workers=12)
    print("================== Successfully preprocessed S1 DATA ================== ")


def main_step1(**kwargs):
    try:
        data_path = kwargs["data_path"]
        aux_path = kwargs["aux_path"]
        shp_path = kwargs["shp_path"]
        mask_path = kwargs["mask_path"]
        ouput_stor = kwargs["ouput_stor"]
        folders = kwargs["folders"]
        typeaux = kwargs["typeaux"]
        dicbands = kwargs["dicbands"]
        mask_data = kwargs["mask_data"]
        add_aux = kwargs["add_aux"]
        workers = kwargs["workers"]
    except KeyError as e:
        print("KeyError: %s undefine" % e)
    preprocess(
        aux_path,
        data_path,
        shp_path,
        mask_path,
        ouput_stor,
        folders,
        dicbands,
        typeaux,
        add_aux,
        mask_data,
        workers=workers,
    )


if __name__ == "__main__":
    shp_path = "shp/"
    aux_path = "auxiliary_data/*"
    data_path = "data_S1/*.tif"
    typeaux = ["DEM10M", "EXP10M", "PEN10M"]
    folders = ["train", "test", "validation"]
    ouput_dir = "dataset/"
    mask_path = "auxiliary_data/S1_mask_lay/ASCLAY_CRS32631.tif"
    workers = -1
    dicbands = {}
    preprocess(
        aux_path,
        data_path,
        shp_path,
        mask_path,
        ouput_dir,
        folders,
        dicbands,
        typeaux,
        add_aux=True,
        mask_data=True,
        workers=workers,
    )
