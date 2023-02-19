"""
Dataset class for the multimodal BigEarthNet data set. The implementations build up on the data loader provided here:

https://git.tu-berlin.de/rsim/BigEarthNet-MM_tools (MIT License)
"""

import os
import json
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import csv
import rasterio
import pickle as pkl
from rasterio.enums import Resampling
from tempfile import NamedTemporaryFile
import shutil
import argparse
import yaml

damaged_files = ["S1BS2B_MSIL2A_20180220T114339_50_85.tif",
                 "S1BS2A_MSIL2A_20180205T100211_4_84.tif",
                 "S1BS2A_MSIL2A_20180205T100211_89_46.tif"]

class BigEarthDataLoader(Dataset):
    """Big Earth dataset."""

    def __init__(self, split_file, root_folder, label_path=None, transform=None, class_filter=range(19)):
        """
        Args:
            split_file (string): Path csv file containing all samples for a specific split
            root_folder (string): Path to the root folder of the data
            label_path (string): Path to file containing sample labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.s1_folder = os.path.join(root_folder, "BigEarthNet-S1-v1.0")
        self.s2_folder = os.path.join(root_folder, "BigEarthNet-v1.0")
        self.merged_folder = os.path.join(root_folder, "merged")
        self.label_path = label_path
        self.class_filter = class_filter

        self.samples = []
        self.label_conversion = None
        self.original_labels = None

        self.split_name = ""

        self.split_file = split_file
        self.s1s2_mapping_file = split_file.replace(split_file.split("/")[-1], "s1s2_mapping.csv")
        self.transform = transform

        self.split_name = os.path.basename(self.split_file).split(".")[0]
        self.pkl_path = self.split_file.replace(".csv", ".pkl")

        self.prepare_label_list()
        self.prepare_sample_list()

        self.merge_bands()

        self.filter_samples()

    def filter_samples(self) -> None:
        """
        Filters samples according to the objects class_filter property.
        """

        num_label_list = np.array([sample["label_numeric"] for sample in self.samples])

        print("\n#############################")
        print(f"Data set before filter: {len(num_label_list)} samples")
        print(np.sum(num_label_list, 0))
        print("#############################\n")

        print("Applying filter: ", self.class_filter)
        filter_invert = [i for i in range(19) if i not in self.class_filter]

        if len(filter_invert) != 0:
            mask_list = [num_label_list[:,c] == 1 for c in filter_invert]
            self.samples = [self.samples[i] for i in np.arange(len(self))[np.sum(mask_list, axis=0) == 0]]

        label_map = np.zeros((len(self.class_filter), 19))

        counter = 0
        for c in range(19):
            if c in self.class_filter:
                label_map[counter,c] = 1
                counter+=1

        for sample in self.samples:
            sample["label_numeric"] = np.matmul(label_map,sample["label_numeric"])

        print("#############################")
        print(f"Data after class filter: {len(self.samples)} samples")
        print(np.sum(num_label_list, 0))
        print(len(self))
        print("#############################")

    def prepare_label_list(self) -> None:
        """
        Loads label json file based on label_path property and sets corresponding properties.
        """
        with open(self.label_path, "rb") as file:
            d = json.load(file)
            self.label_dict = d["BigEarthNet-19_labels"]
            self.original_labels = d["original_labels"]
            self.label_conversion = d["label_conversion1"]

    def fill_table(self) -> None:
        """
        Builds and fills table mapping S1 and S2 samples.
        """
        print("##### Start filling table #####")
        label_dict = {}
        with open(self.s1s2_mapping_file, "r+b") as file:
            csv_reader = csv.reader(file, delimiter=",")
            for row in csv_reader:
                label_dict[row[0]] = row[1]

        with NamedTemporaryFile(mode="w", delete=False) as tempfile:
            with open(self.split_file, "r+b") as file:
                csv_reader = csv.reader(file, delimiter=",")
                writer = csv.writer(tempfile, delimiter=",")

                for row in csv_reader:
                    row.append(label_dict[row[0]])
                    writer.writerow(row)

            shutil.move(tempfile.name, self.split_file)

    def prepare_sample_list(self) -> None:
        """
        Reads csv containing S1 and S2 mapping and builds internal list of samples
        containing paths to tifs and label information.
        """

        if os.path.exists(self.pkl_path):
            with open(self.pkl_path, "rb") as file:
                self.samples = pkl.load(file)
        else:
            with open(self.split_file, "r+b") as file:
                csv_reader = csv.reader(file, delimiter=",")
                for row in csv_reader:
                    only_s2 = len(row)==1
                    break

            if only_s2:
                self.fill_table()

            with open(self.split_file, "r+b") as file:
                csv_reader = csv.reader(file, delimiter=",")
                for row in csv_reader:
                    merged_name = os.path.join(self.merged_folder, "S1B"+row[0] + ".tif")
                    self.samples.append({"Merged": merged_name, "S1": row[1], "S2": row[0], "label": None})
                np.random.shuffle(self.samples)

            for i, sample in enumerate(self.samples):
                print(f"Prepare sample {i} of {len(self.samples)}         ", end="\r")
                if not os.path.exists(os.path.join(self.s1_folder, sample["S1"])) \
                    or not os.path.exists(os.path.join(self.s2_folder, sample["S2"])):
                    continue

                s1_json_file = [os.path.join(self.s1_folder, sample["S1"], p) \
                                    for p in os.listdir(os.path.join(self.s1_folder, sample["S1"])) \
                                        if p.endswith(".json")]

                assert len(s1_json_file)==1
                s1_json_file = s1_json_file[0]

                with open(s1_json_file, "r+b") as file:
                    s1_json = json.load(file)

                s2_json_file = [os.path.join(self.s2_folder, sample["S2"], p) \
                                    for p in os.listdir(os.path.join(self.s2_folder, sample["S2"])) \
                                        if p.endswith(".json")]
                assert len(s2_json_file)==1
                s2_json_file = s2_json_file[0]

                with open(s2_json_file, "r+b") as file:
                    s2_json = json.load(file)

                assert s2_json["labels"] == s1_json["labels"]

                sample["label_text"] = s2_json["labels"]
                sample["label_numeric"] = np.zeros(19)
                for i in [self.label_conversion[l] for l in sample["label_text"] \
                                                   if self.label_conversion[l] != 1000]:

                    sample["label_numeric"][i] = 1

            with open(self.pkl_path, "wb") as file:
                pkl.dump(self.samples, file)

        for i in np.arange(len(self.samples)-1, -1, -1):
            if self.samples[i]["Merged"].split("/")[-1] in damaged_files:
                self.samples.remove(self.samples[i])


        print(f"Samples for {self.split_name}-split loaded:     {len(self.samples)} Samples")

    def __len__(self) -> int:
        """
        Returns the number of samples in this data set.
        """
        return len(self.samples)

    def merge_bands(self) -> None:
        """
        Preprocessing method:
        Merges individual S1 polarization and S2 band tifs into one single tif file.
        """
        if not os.path.exists(os.path.join(self.merged_folder)):
            os.mkdir(self.merged_folder)
        for i, sample in enumerate(self.samples):
            merged_name = sample["Merged"]

            if os.path.exists(merged_name):
                print(f"\r Checked Sample {i}/{len(self.samples)}        -"\
                       +"      {merged_name}        ", end="\r")
                continue

            print(f"\r Prepare Sample {i}/{len(self.samples)}        -"\
                   +"         {merged_name}")
            s1_files = [os.path.join(self.s1_folder, sample["S1"], p)
                            for p in os.listdir(os.path.join(self.s1_folder,
                                                             sample["S1"])) if p.endswith(".tif")]
            s2_files = [os.path.join(self.s2_folder, sample["S2"], p)
                            for p in os.listdir(os.path.join(self.s2_folder,
                                                             sample["S2"])) if p.endswith(".tif")]

            s1_files = np.sort(s1_files)
            s2_root_file = s2_files[0][:-6]
            s2_files = [s2_root_file + p + ".tif" for p in \
                ["01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "11", "12"]]

            merged_files = np.concatenate([s1_files, s2_files])

            with rasterio.open(merged_name,
                               "w",
                               height=120,
                               width=120,
                               count=len(merged_files),
                               dtype=np.float32) as dst:

                for idx, layer in enumerate(merged_files, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(idx, src1.read(1,out_shape=(
                                            1,120,120),
                                        resampling=Resampling.cubic))

        print("Bands merged!                                                     "\
             +"                                                                       ")

    def __getitem__(self, idx) -> dict:
        """
        Loads and returns a SAR-optical data sample for a given id.

        idx (int)   :   ID of the sample to be loaded.
        """

        sample = self.samples[idx]

        merged_name = sample["Merged"]

        img = torch.moveaxis(torch.tensor(io.imread(merged_name)), source=2, destination=0)

        if self.transform:
            img = self.transform(img)

        return {"label_numeric": sample["label_numeric"], "img": img, "merged_name": merged_name}

    def get_data_set_stats(self):
        """
        Returns the channel-wise mean and standard deviation for the loaded data split.
        """
        mu = np.zeros(14, dtype=np.float64)
        mu_sqr = np.zeros(14, dtype=np.float64)

        if self.mean is None:
            for idx in range(len(self.samples)):
                print(f"{idx+1}/{len(self.samples)}         ",\
                     +f"{self.samples[idx]['Merged']}                     ", end="\r")
                sample = self[idx]

                mu += np.mean(sample["img"], axis=(-1,-2))
                mu_sqr += np.mean(sample["img"]**2, axis=(-1,-2))

            mean_val = mu / len(self.samples)
            std_val = (mu_sqr / len(self.samples) - self.mean**2)**0.5

        return mean_val, std_val

    def print_rgb(self, ids=None, num_samples=None, save_path="./rgb") -> None:
        """
        Saves a given number of random samples or a given number of ids into optical RGB and SAR files.

        ids (list[int])     :   List of integer ids that shall be printed.
        num_samples (int)   :   Number of samples to print (only used when ids==None).
        sava_path (str)     :   Folder where images are saved to.
        """

        if ids is None:
            assert num_samples is not None, \
                   "Either explicit sample ids or a number of "\
                   + "random samples needs to be provided."
            idx = np.random.choice(np.arange(len(self)),
                                   size=num_samples,
                                   replace=False)
        else:
            idx = ids

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i, id_sub in enumerate(idx):
            print(f"{i}/{len(idx)}")
            sample = self[id_sub]

            name = sample["merged_name"].split("/")[-1].split(".")[0]

            img = io.imread(sample["merged_name"])[:,:,2:5]
            img = np.stack([img[:,:,2], img[:,:,1], img[:,:,0]], axis=-1)
            img = (255 * (np.clip(img, 0, 8000) / 8000)).astype(np.uint8)
            io.imsave(os.path.join(save_path,name + ".png"), img)

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-cfg",
                        "--config_file",
                        help="path to configuration json file.")
    args = parser.parse_args()
    cfg_path = args.config_file
    with open(cfg_path, "rb") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    data_root_path = cfg["data_root_path"]
    data_splits = cfg["data"]["splits"]
    cfg_label_path = cfg["data"]["label_path"]

    for data_split in data_splits:
        print("#######################", data_split, "########################")
        _ = BigEarthDataLoader(split_file=data_splits[data_split],
                               root_folder=data_root_path,
                               label_path=cfg_label_path,
                               transform=None,
                               class_filter=np.arange(19))
