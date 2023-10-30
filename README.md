LSD4WSD: 
========
**L**earning **S**AR **D**ataset for **W**et **S**now **D**etection - Full Analysis Version. 
The aim of this dataset is to provide a basis for automatic learning to detect wet snow.
It is based on Sentinel-1 SAR satellite images acquired between August 2020 and August 2021 over the French Alps (tiles 31 TGL). The new version of this dataset is no longer simply restricted to a classification task, and provides a set of metadata for each sample. 

![image-20231030121103383](/home/listic/Téléchargements/README.assets/image-20231030121103383.png)

| Types            | Improvements                                                 |
| ---------------- | ------------------------------------------------------------ |
| Number of massif | add 7 new massif to cover the all Sentinel-1 images (cf `info.pdf`). |
| Acquisition      | add images of the descending pass in addition to those originally used in the ascending pass. |
| Sample           | reduction in the size of the samples considered to 15 by 15 to facilitate evaluation at the central pixel. |
| Sample           | increased density of extracted windows, with a distance of approximately 500 meters between the centers of the windows. |
| Sample           | removal of the pre-processing involving the use of logarithms. |
| Sample           | removal of the pre-processing involving the normalisation.   |
| Labels           | new structure for the labels part: dictionary with keys: `topography`, `metadata` and `physics`. |
| Labels           | `physics`: addition of direct information from the CROCUS model for 3 simulations: Liquid Water Content, snow height and minimum snowpack temperature . |
| Labels           | `topography`: information on the slope, altitude and average orientation of the sample. |
| Labels           | `metadata` : information on the date of the sample, the mountain massif and the run (ascending or descending). |
| Dataset          | removal of the train/test split*                             |

*We leave it up to the user to use the Group Kfold method to validate the models using the mass information.

Finally, it consists of 2467516 samples of size 15 by 15 by 9. For each sample, the 9 metadata are provided, using in particular the [Crocus](https://www.umr-cnrm.fr/spip.php?article265&lang=en) physical model:

- `topography`:

  - elevation (meters) (average),
  - orientation (degrees) (average),
  - slope (degrees) (average),

-  `metadata`:

  - name of the alpine massif,
  - date of acquisition,
  - type of acquisition (ascending/descending),

- `physics`

  - Liquid Water Content (km/m2),

  - snow height (m),

  - minimum snowpack temperature (Celsius degree).


The 9 channels are in the following order:

- Sentinel-1 polarimetric channels: VV, VH and the combination C: VV/VH in linear,
- Topographical features: altitude, orientation, slope
- Polarimetric ratio with a reference summer image: VV/VVref, VH/VHref, C/Cref

An overview of the distribution and a summary of the sample statistics can be found in the file  `info.pdf`.

The data is stored in .hdf5 format with gzip compression. The structure is as follows:

```bash
dataset.hdf5
  ├── img (float32)
  ├── metadata (string)
  ├── topography (float32)
  └── physics (float32)
```
We provide a python script to read and request the data. The script is `dataset_load.py`. It is based on the `h5py`, `numpy` and `pandas` libraries. It allows to select a part or the whole dataset using requests on the metadata. The script is documented and can be used as follows:

```python
import dataset_load as dl

# initialize the loader
path = 'dataset.hdf5'
dataset = dl.Dataset_loader(
  path,
  shuffle=False,
  descrp=[
      "date",
      "massif",
      "elevation",
      "slope",
      "orientation",
      "tmin",
      "tel",
      "hsnow",
  ],
)

# print the infos
print(dataset.infos)

# request the data
rq1 = "massif == 'VERCORS' and \
      ((date.dt.month == 3 and date.dt.day== 1) or \
      (elevation > 3000 and hsnow < 0.25))"

rq2 = "massif == 'ARAVIS' & aquisition == 'ASC' & \
        elev == 900.0 & slope == 20 & theta == 45 "

rq3 = "massif == 'ARAVIS' | date.dt.month == 1"

# load the requested data
x, y = dataset.request_data(rq1)
print(x.shape)
```

The processing chain is available at the following [Github](https://github.com/Matthieu-Gallet/LSD4WSD-dataset) address.

------

The authors would like to acknowledge the support from the National Centre for Space Studies (CNES) in providing computing facilities and access to SAR images via the PEPS platform.

The authors would like to deeply thank Mathieu Fructus for running the Crocus simulations

