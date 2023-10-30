from os.path import basename
import pandas as pd
import numpy as np


def prepare_label(label_array, type_d):
    dic = {}
    mtd, tpg, phy = [], [], []
    for i in range(len(label_array)):
        phy.append(label_array[i][:3])
        tpg.append(label_array[i][3:6])
        mtd_temp = label_array[i][6:]
        mtd_temp.append(type_d)
        mtd.append(mtd_temp)
    dic["metadata"] = np.array(mtd, dtype=np.string_)
    dic["topography"] = np.array(tpg, dtype=np.float32)
    dic["physics"] = np.array(phy, dtype=np.float32)
    return dic


def process_file_II(array_R, i, tel):
    hpo = extract_topo(array_R)
    label_R = expert_label_snow(i, tel, hpo)
    return array_R, label_R


def mean_angle(deg):
    deg = np.deg2rad(deg)
    x = np.mean(np.cos(deg))
    y = np.mean(np.sin(deg))
    res = np.rad2deg(np.arctan2(y, x)).round(3)
    if res < 0:
        return max(res, 360 + res)
    else:
        return res


def altitude_plage(plg):
    h = np.arange(0, 5100, 300)
    altitude = h[np.argmin(np.abs(h - plg))]
    return altitude


def theta_plage(theta):
    t0 = np.arange(22.5, 360 + 22.5, 45)
    t = np.arange(0, 360, 45)
    if theta > t0[-1] or theta < t0[0]:
        ct = t[0]
    else:
        ind = np.where(theta < t0)[0][0]
        ct = t[ind]
    return ct


def pente_plage(p):
    if (p < 2) & (p >= 0):
        pente = 0
    elif (p < 30) & (p >= 2):
        pente = 20
    else:
        pente = 45
    return pente


def stations2crocus(o):
    high = o.altitude.values[0]
    theta = o.orientation.values[0]
    pente = o.pente.values[0]
    hpo = altitude_plage(high), pente_plage(pente), theta_plage(theta)
    return hpo


def extract_topo(img):
    info = pd.DataFrame()
    info["altitude"] = [img[:, :, 3].mean()]
    info["altitude_std"] = [img[:, :, 3].std()]
    info["orientation"] = [mean_angle(img[:, :, 4])]
    info["pente"] = [img[:, :, 5].mean()]
    return stations2crocus(info)


def expert_label_snow(i, tel, hpo):
    name_base = basename(i)[:-4]
    massif = name_base.split("_")[0]
    date_sample = name_base.split("_")[1]
    if (hpo in list(tel[0][massif].keys())) and (hpo in list(tel[1][massif].keys())):
        try:
            dframe_sample = (
                tel[0][massif][hpo].loc[date_sample],
                tel[1][massif][hpo].loc[date_sample],
                tel[2][massif][hpo].loc[date_sample],
            )
        except:
            return -1
    else:
        return -1
    cond_ex = (
        (len(dframe_sample[0]) > 0)
        and (len(dframe_sample[1]) > 0)
        and len(dframe_sample[2]) > 0
    )
    cond_nan = (
        (dframe_sample[0].isnull().values.any())
        or (dframe_sample[1].isnull().values.any())
        or (dframe_sample[2].isnull().values.any())
    )
    if cond_ex and not cond_nan:
        return [
            dframe_sample[0].tmin,
            dframe_sample[1].hs,
            dframe_sample[2].tel,
            hpo[0],
            hpo[1],
            hpo[2],
            date_sample,
            massif,
        ]
    else:
        return -1


def add_labels_II(data, name_file, tel, type_d):
    results = [
        process_file_II(raster_a, name_file, tel)
        for raster_a in data
        if process_file_II(raster_a, name_file, tel)[1] != -1
    ]
    if len(results) > 1:
        X = np.array([i[0] for i in results], dtype=np.float32)
        y = [i[1] for i in results]
        y = prepare_label(y, type_d)  # results[1])
    else:
        X, y = np.array([]), np.array([])
    return X, y


def select_non_empty_list(extr):
    x, y = list(zip(*extr))
    X = []
    Y = {}

    for i in x:
        if len(i) > 0:
            X.extend(i)
    X = np.array(X, dtype=np.float32)

    for i in y:
        if len(i) > 0:
            for k, v in i.items():
                if k in Y.keys():
                    Y[k] = np.concatenate((Y[k], v), axis=0)
                else:
                    Y[k] = v
    return X, Y
