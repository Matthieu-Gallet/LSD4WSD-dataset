import pandas as pd
import numpy as np
import h5py


def save_h5_II(img, y, filename, additional_info=True):
    """Save data in hdf5 format with a data part (in float32),
    a metadata part to describe data and two additional fields  (topography and physics)

    Parameters
    ----------
    img : np.array
        dataset to save (float32)
    y : dict
        dictionary of description of the dataset
        :warning: the mandatory key is "metadata", the other keys are optional
    filename : str
        path to save the data
    additional_info : bool, optional
        give the possibility to save additional information (topography and physics), by default True
    """
    metadata = y["metadata"]
    if additional_info:
        topo = y["topography"]
        label = y["physics"]
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("img", np.shape(img), compression="gzip", data=img)
        hf.create_dataset(
            "metadata", np.shape(metadata), compression="gzip", data=metadata
        )
        if additional_info:
            hf.create_dataset(
                "topography", np.shape(topo), compression="gzip", data=topo
            )
            hf.create_dataset(
                "physics", np.shape(label), compression="gzip", data=label
            )


def load_h5_II(filename):
    y = {}
    with h5py.File(filename, "r") as hf:
        data = np.array(hf["img"][:])
        y["metadata"] = np.array(hf["metadata"][:]).astype(str)
        y["topography"] = np.array(hf["topography"][:]).astype(np.float32)
        y["physics"] = np.array(hf["physics"][:]).astype(np.float32)
    return data, y


def load_data_h5(filename, idx=None):
    """Load the dataset part of the hdf5 file

    Parameters
    ----------
    filename : str
        path to the hdf5 file
    idx : np.array, optional
        index of the dataset to load, by default None

    Returns
    -------
    np.array
        dataset in float32
    """
    with h5py.File(filename, "r") as hf:
        if idx is not None:
            return np.array(hf["img"][idx]).astype(np.float32)
        else:
            return np.array(hf["img"][:]).astype(np.float32)


def load_info_h5(filename, idx=None, type_metadata="str"):
    """Load the informations part of the hdf5 file

    Parameters
    ----------
    filename : str
        path to the hdf5 file
    idx : np.array, optional
        index of the dataset to load, by default None
    type_metadata : str, optional
        type of the metadata, by default "str"
        :info: for direct integer label, use "int", for other type see numpy dtype

    Returns
    -------
    np.array
        metadata in the type_metadata format
    """
    with h5py.File(filename, "r") as hf:
        metadata = np.array(hf["metadata"][:]).astype(type_metadata)
        try:
            topography = np.array(hf["topography"][:]).astype(np.float32)
            physics = np.array(hf["physics"][:]).astype(np.float32)
        except Exception as e:
            print(e)
            print(
                "No additional information: topography and physics\n Be careful to the description (columns) used for the dataloader and the metadata type)"
            )
            topography = np.array([None] * len(metadata))
            physics = np.array([None] * len(metadata))
    if idx is not None:
        return metadata[idx], topography[idx], physics[idx]
    else:
        return metadata, topography, physics


def shuffle_data(X, y, seed=42):
    """Shuffle the dataset and the metadata

    Parameters
    ----------
    X : np.array
        dataset
    y : dict
        dictionary of description of the dataset

    Returns
    -------
    np.array
        shuffled dataset
    dict
        shuffled metadata
    """
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx], {k: v[idx] for k, v in y.items()}


class Dataset_loader:
    def __init__(
        self,
        path,
        shuffle=True,
        descrp=[
            "date",
            "massif",
            "aquisition",
            "elevation",
            "slope",
            "orientation",
            "tmin",
            "tel",
            "hsnow",
        ],
        print_info=True,
    ):
        self.path = path
        self.descrp = descrp
        self.shuffle = shuffle
        self.seed = 42
        self.X = None
        self.y = {}
        self.print_info = print_info
        self.init_info()

    def init_info(self):
        """Load the metadata and prepare the information for the request"""
        temp = load_info_h5(self.path)
        self.infos = pd.concat([pd.DataFrame(i) for i in temp], axis=1)
        self.infos.columns = self.descrp
        self.infos.date = pd.to_datetime(self.infos.date, format="%Y%m%d")
        self.idx_request = self.infos.index.values

    def check_data(self):
        """Check if the dataset and the metadata have the same dimension"""
        return np.all(
            [self.dim[0] == self.y[key].shape[0] for key in list(self.y.keys())]
        )

    def load_data(self):
        """Load the dataset and the metadata with respect to the request and shuffle the data if needed"""
        self.X = load_data_h5(self.path, self.idx_request)
        self.dim = self.X.shape
        print(self.dim)
        self.y["metadata"], self.y["topography"], self.y["physics"] = load_info_h5(
            self.path, self.idx_request
        )
        if self.check_data():
            if self.shuffle:
                self.X, self.y = shuffle_data(self.X, self.y, seed=self.seed)
        else:
            if self.print_info:
                print("Error in dimension")
        return self.X, self.y

    def request_data(self, condition):
        """Request the dataset with respect to the condition

        Parameters
        ----------
        condition : str
            SQL like request to select the data in the pandas dataframe
        """
        try:
            self.idx_request = self.infos.query(condition).index.values
            if self.print_info:
                print(f"Request: {condition} with {len(self.idx_request)} samples")
        except Exception as e:
            if self.print_info:
                print(e)
                print("Error in request")
        return self.load_data()

    def __repr__(self):
        return f"Dataset_loader: ({self.path}) with {len(self.idx_request)} samples"

    def __str__(self):
        return f"Dataset_loader: ({self.path}) with {len(self.idx_request)} samples"


if __name__ == "__main__":
    path_dataset = "../dataset_/V4v5_dataset_A_HD15_REG_LIN/final_E4/data_train_V4.h5"
    dtst_ld = Dataset_loader(
        path_dataset,
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
    print(dtst_ld.infos)

    # Example of request
    rq1 = "massif == 'VERCORS' and \
          ((date.dt.month == 3 and date.dt.day== 1) or \
          (elevation > 3000 and hsnow < 0.25))"

    rq2 = "massif == 'ARAVIS' & aquisition == 'ASC' & \
           elev == 900.0 & slope == 20 & theta == 45 "

    rq3 = "massif == 'ARAVIS' | date.dt.month == 1"

    x, y = dtst_ld.request_data(rq1)
    print(x.shape)
