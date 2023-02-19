"""
Script for testing the fusion network with integrated OOD detection.

Arguments:
    -cfg    Path to config yaml file of the experiment.
    -ts     Test scenario (see docu for options).
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
from sklearn.metrics import  f1_score, fbeta_score
from tqdm import tqdm
import yaml
from utils.utils import set_seed
from models.mm_resnet import ResNetFuse
from data.bigearthnet_dataloader import BigEarthDataLoader
from data.bigearthnet_utils import normalize

parser = argparse.ArgumentParser()

parser.add_argument("-cfg", "--config_file", help="path to configuration json file.")
parser.add_argument("-ts", "--test_set", help="Which test set to use for evaluation.",
                    choices=[
                        "train",
                        "val",
                        "test",
                        "ood_cloudy",
                        "ood_cloudy_handpicked",
                        "ood_ice_and_snow",
                        "left_out_classes",
                        "missing_sar",
                        "missing_opt",
                        "corrupted_sar",
                        "corrupted_opt"
                        ])
parser.add_argument("-s", "--seed", help="seed for evaluation run.", default=42)
args = parser.parse_args()
cfg_file = args.config_file

test_set = args.test_set

assert test_set in [
            "train",
            "val",
            "test",
            "ood_cloudy",
            "ood_cloudy_handpicked",
            "ood_ice_and_snow",
            "left_out_classes",
            "missing_sar",
            "missing_opt",
            "corrupted_sar",
            "corrupted_opt",
            ], test_set

metrics = [
          "f1_micro",
          "f1_macro",
          "f1_sample",
          "f2_micro",
          "f2_macro",
          "f2_sample"
          ]

assert os.path.exists(cfg_file), f"Configuration file '{cfg_file}' does not exist!"

with open(cfg_file, "rb") as file:
    cfg = yaml.safe_load(file)

device="cuda" if torch.cuda.is_available() else "cpu"

# Put this in config file
num_workers = cfg["num_workers"]
experiment_root_path = cfg["experiment_root_path"]
data_root_path = cfg["data_root_path"]
train_split = cfg["data"]["splits"]["train_split"]
val_split = cfg["data"]["splits"]["val_split"]
test_split = cfg["data"]["splits"]["test_split"]
cloudy_split = cfg["data"]["splits"]["cloudy_split"]
ice_and_snow_split = cfg["data"]["splits"]["ice_and_snow_split"]
cloudy_handpicked_split = cfg["data"]["splits"]["cloudy_handpicked_split"]
label_path = cfg["data"]["label_path"]
classes_ood_training = cfg["data"]["class_splits"]["classes_ood_training"]
classes_ood_testing = cfg["data"]["class_splits"]["classes_ood_testing"]
classes_in_distribution = cfg["data"]["class_splits"]["classes_in_distribution"]

seed = cfg["setup"]["seed"]
num_epochs = cfg["main_training"]["num_epochs"]
batch_size = cfg["main_training"]["batch_size"]
learning_rate = cfg["main_training"]["learning_rate"]

input_dim = cfg["data"]["splits"]
resnet_type = cfg["model"]["type"]
fusion_stage = cfg["model"]["fusion_stage"]
branch_split = cfg["model"]["branch_split"]
save_path_model = cfg["save_paths"]["save_path_models"]

if save_path_model[0] == "/":
    assert save_path_model.startswith(experiment_root_path), \
        "Model path should be relative or subfolder of experiment path."
else:
    save_path_model = os.path.join(experiment_root_path, save_path_model)
os.makedirs(save_path_model, exist_ok=True)
save_path_sar = os.path.join(save_path_model, "ood_detector_sar.pth")
save_path_optical = os.path.join(save_path_model, "ood_detector_optical.pth")
fusion_model_path = os.path.join(save_path_model, "final_model.pth")

set_seed(seed)

res_dict = {}

num_classes = len(classes_in_distribution)

sigmoid = Sigmoid()

with torch.no_grad():

    model_checkpoint = torch.load(fusion_model_path, map_location="cpu")
    fusion_model = ResNetFuse(res_type=resnet_type,
                            fusion_stage=fusion_stage,
                            branch_split=branch_split,
                            device=device,
                            num_classes=num_classes)

    fusion_model.load_state_dict(model_checkpoint["model_state_dict"])

    fusion_model.to(device)
    fusion_model.eval()

    if test_set not in ["missing_opt", "missing_sar"]:

        _, sar_branch_out, opt_branch_out =\
            fusion_model(torch.rand([14, 120, 120], \
                         device=device).unsqueeze(0), give_branch_out=True)

        in_features_opt = opt_branch_out.flatten().shape[0]
        in_features_sar = sar_branch_out.flatten().shape[0]

        ood_detector_sar = torch.nn.Sequential(torch.nn.Flatten(start_dim=1),
                                               torch.nn.Linear(in_features=in_features_opt,
                                                               out_features=512),
                                               torch.nn.Dropout(0.1),
                                               torch.nn.Tanh(),
                                               torch.nn.Linear(512,64),
                                               torch.nn.Tanh(),
                                               torch.nn.Linear(64,1))
        ood_detector_sar.load_state_dict(torch.load(save_path_sar, map_location="cpu"))
        ood_detector_sar.to(device)
        ood_detector_sar.eval()

        ood_detector_opt = torch.nn.Sequential(torch.nn.Flatten(start_dim=1),
                                               torch.nn.Linear(in_features=in_features_sar,
                                                               out_features=512),
                                               torch.nn.Dropout(0.1),
                                               torch.nn.Tanh(),
                                               torch.nn.Linear(512,64),
                                               torch.nn.Tanh(),
                                               torch.nn.Linear(64,1))
        ood_detector_opt.load_state_dict(torch.load(save_path_optical, map_location="cpu"))
        ood_detector_opt.to(device)
        ood_detector_opt.eval()
    else:
        ood_detector_opt = None
        ood_detector_sar = None

    if test_set == "train":
        split = train_split
    elif test_set == "val":
        split = val_split
    elif test_set == "ood_cloudy":
        split = cloudy_split
    elif test_set == "ood_cloudy_handpicked":
        split = cloudy_handpicked_split
    elif test_set == "ood_ice_and_snow":
        split = ice_and_snow_split
    else: split = test_split

    if test_set == "left_out_classes":
        class_filter = classes_ood_testing
    else:
        class_filter = classes_in_distribution

    # training data loader
    data_set = BigEarthDataLoader(split_file=train_split,
                                  root_folder=data_root_path,
                                  label_path=label_path,
                                  transform=normalize,
                                  class_filter=class_filter)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)

    pred_fuse_list = []
    pred_m1_list = []
    pred_m2_list = []
    y_true_list = []
    pred_ood_optical_list = []
    pred_ood_sar_list = []

    for data in tqdm(data_loader, test_set):

        x = data["img"]
        y = data["label_numeric"]

        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        y_true_list.append(y)

        if ood_detector_opt is None or ood_detector_sar is None:

            if test_set == "missing_opt":
                pred_m1 = fusion_model(x)[len(x):2*len(x)]
                pred_m1_list.append(pred_m1.cpu().detach().numpy())
            elif test_set == "missing_sar":
                pred_m2 = fusion_model(x)[2*len(x):]
                pred_m2_list.append(pred_m2.cpu().detach().numpy())
            else:
                raise RuntimeError("Test set should be 'missing_opt' or 'missing_sar' here.")

        else:
            if test_set == "corrupted_sar":
                x1 = torch.randn(np.shape(x[:,:2,:,:]), device=device)
                x2 = torch.as_tensor(x[:,2:,:,:], dtype=torch.float32, device=device)
                x = torch.cat([x1,x2], dim=-3)
            elif test_set == "corrupted_opt":
                x1 = torch.as_tensor(x[:,:2,:,:], dtype=torch.float32, device=device)
                x2 = torch.randn(np.shape(x[:,2:,:,:]), device=device)
                x = torch.cat([x1,x2], dim=-3)

            pred_fuse, sar_branch_out, opt_branch_out = fusion_model(x, give_branch_out=True)
            pred_m1 = pred_fuse[len(x):2*len(x)]
            pred_m2 = pred_fuse[2*len(x):]
            pred_fuse = pred_fuse[:len(x)]

            pred_fuse_list.append(pred_fuse.cpu().detach().numpy())
            pred_m1_list.append(pred_m1.cpu().detach().numpy())
            pred_m2_list.append(pred_m2.cpu().detach().numpy())

            pred_ood_optical_list.append(
                sigmoid(ood_detector_opt(opt_branch_out)).detach().cpu().numpy())
            pred_ood_sar_list.append(
                sigmoid(ood_detector_sar(sar_branch_out)).detach().cpu().numpy())

    del data_set
    del data_loader

    if len(pred_fuse_list) > 0:
        pred_fuse_list = sigmoid(torch.as_tensor(np.concatenate(pred_fuse_list,0))).numpy()

    if len(pred_m1_list) > 0:
        pred_m1_list = sigmoid(torch.as_tensor(np.concatenate(pred_m1_list,0))).numpy()

    if len(pred_m2_list) > 0:
        pred_m2_list = sigmoid(torch.as_tensor(np.concatenate(pred_m2_list,0))).numpy()

    if len(pred_ood_sar_list) > 0 and len(pred_ood_optical_list) > 0:
        weight_predictions = np.concatenate([np.concatenate(pred_ood_sar_list,0),
                                             np.concatenate(pred_ood_optical_list,0)], axis=-1)
    else:
        weight_predictions = []

    if len(weight_predictions) > 0:
        pred_weighted = (1-weight_predictions[:,0:1])*weight_predictions[:,1:]*pred_m1_list \
                +weight_predictions[:,0:1]*(1-weight_predictions[:,1:])*pred_m2_list \
                +(1-weight_predictions[:,0:1])*(1-weight_predictions[:,1:])*pred_fuse_list\
                +weight_predictions[:,0:1]*weight_predictions[:,1:]*0.5*np.ones_like(pred_m1_list)
    else:
        pred_weighted = []


    y_true = np.concatenate(y_true_list)

    if len(pred_fuse_list) > 0:
        pred_fuse_list = np.round(pred_fuse_list)
        F1_MICRO_FUSE = \
            f1_score(y_true=y_true, y_pred=pred_fuse_list, average="micro", zero_division=0)
        F1_MACRO_FUSE = \
            f1_score(y_true=y_true, y_pred=pred_fuse_list, average="macro", zero_division=0)
        F1_SAMPLE_FUSE = \
            f1_score(y_true=y_true, y_pred=pred_fuse_list, average="samples", zero_division=0)
        F2_MICRO_FUSE = \
            fbeta_score(y_true=y_true, y_pred=pred_fuse_list, average="micro", beta=2, zero_division=0)
        F2_MACRO_FUSE = \
            fbeta_score(y_true=y_true, y_pred=pred_fuse_list, average="macro", beta=2, zero_division=0)
        F2_SAMPLE_FUSE = \
            fbeta_score(y_true=y_true, y_pred=pred_fuse_list, average="samples", beta=2, zero_division=0)
    else:
        F1_MICRO_FUSE=F1_MACRO_FUSE=F1_SAMPLE_FUSE=F2_MICRO_FUSE=F2_MACRO_FUSE=F2_SAMPLE_FUSE=None

    if len(pred_m1_list) > 0:
        pred_m1_list = np.round(pred_m1_list)
        F1_MICRO_BRANCH_1 = \
            f1_score(y_true=y_true, y_pred=pred_m1_list, average="micro", zero_division=0)
        F1_MACRO_BRANCH_1 = \
            f1_score(y_true=y_true, y_pred=pred_m1_list, average="macro", zero_division=0)
        F1_SAMPLE_BRANCH_1 = \
            f1_score(y_true=y_true, y_pred=pred_m1_list, average="samples", zero_division=0)
        F2_MICRO_BRANCH_1 = \
            fbeta_score(y_true=y_true, y_pred=pred_m1_list, average="micro", beta=2, zero_division=0)
        F2_MACRO_BRANCH_1 = \
            fbeta_score(y_true=y_true, y_pred=pred_m1_list, average="macro", beta=2, zero_division=0)
        F2_SAMPLE_BRANCH_1 = \
            fbeta_score(y_true=y_true, y_pred=pred_m1_list, average="samples", beta=2, zero_division=0)
    else:
        F1_MICRO_BRANCH_1=F1_MACRO_BRANCH_1=F1_SAMPLE_BRANCH_1\
                         =F2_MICRO_BRANCH_1=F2_MACRO_BRANCH_1=F2_SAMPLE_BRANCH_1=None

    if len(pred_m2_list) > 0:
        pred_m2_list = np.round(pred_m2_list)
        F1_MICRO_BRANCH_2 = \
            f1_score(y_true=y_true, y_pred=pred_m2_list, average="micro", zero_division=0)
        F1_MACRO_BRANCH_2 = \
            f1_score(y_true=y_true, y_pred=pred_m2_list, average="macro", zero_division=0)
        F1_SAMPLE_BRANCH_2 = \
            f1_score(y_true=y_true, y_pred=pred_m2_list, average="samples", zero_division=0)
        F2_MICRO_BRANCH_2 = \
            fbeta_score(y_true=y_true, y_pred=pred_m2_list, average="micro", beta=2, zero_division=0)
        F2_MACRO_BRANCH_2 = \
            fbeta_score(y_true=y_true, y_pred=pred_m2_list, average="macro", beta=2, zero_division=0)
        F2_SAMPLE_BRANCH_2 = \
            fbeta_score(y_true=y_true, y_pred=pred_m2_list, average="samples", beta=2, zero_division=0)
    else:
        F1_MICRO_BRANCH_2=F1_MACRO_BRANCH_2=F1_SAMPLE_BRANCH_2=F2_MICRO_BRANCH_2\
                         =F2_MACRO_BRANCH_2=F2_SAMPLE_BRANCH_2=None

    if len(pred_weighted) > 0:
        pred_weighted = np.round(pred_weighted)
        F1_MICRO_WEIGHTED = \
            f1_score(y_true=y_true, y_pred=pred_weighted, average="micro", zero_division=0)
        F1_MACRO_WEIGHTED = \
            f1_score(y_true=y_true, y_pred=pred_weighted, average="macro", zero_division=0)
        F1_SAMPLE_WEIGHTED = \
            f1_score(y_true=y_true, y_pred=pred_weighted, average="samples", zero_division=0)
        F2_MICRO_WEIGHTED = \
            fbeta_score(y_true=y_true, y_pred=pred_weighted, average="micro", beta=2, zero_division=0)
        F2_MACRO_WEIGHTED = \
            fbeta_score(y_true=y_true, y_pred=pred_weighted, average="macro", beta=2, zero_division=0)
        F2_SAMPLE_WEIGHTED = \
            fbeta_score(y_true=y_true, y_pred=pred_weighted, average="samples", beta=2, zero_division=0)
    else:
        F1_MICRO_WEIGHTED=F1_MACRO_WEIGHTED=F1_SAMPLE_WEIGHTED=F2_MICRO_WEIGHTED\
                         =F2_MACRO_WEIGHTED=F2_SAMPLE_WEIGHTED=None

    result_dict = {
        "fused":{
            "f1_micro": F1_MICRO_FUSE,
            "f1_macro": F1_MACRO_FUSE,
            "f1_sample": F1_SAMPLE_FUSE,
            "f2_micro": F2_MICRO_FUSE,
            "f2_macro": F2_MACRO_FUSE,
            "f2_sample": F2_SAMPLE_FUSE,
        },

        "weighted":
            {
            "f1_micro": F1_MICRO_WEIGHTED,
            "f1_macro": F1_MACRO_WEIGHTED,
            "f1_sample": F1_SAMPLE_WEIGHTED,
            "f2_micro": F2_MICRO_WEIGHTED,
            "f2_macro": F2_MACRO_WEIGHTED,
            "f2_sample": F2_SAMPLE_WEIGHTED,
            },

        "branch_1":
            {
            "f1_micro": F1_MICRO_BRANCH_1,
            "f1_macro": F1_MACRO_BRANCH_1,
            "f1_sample": F1_SAMPLE_BRANCH_1,
            "f2_micro": F2_MICRO_BRANCH_1,
            "f2_macro": F2_MACRO_BRANCH_1,
            "f2_sample":  F2_SAMPLE_BRANCH_1,
            },

        "branch_2":
            {
            "f1_micro": F1_MICRO_BRANCH_2,
            "f1_macro": F1_MACRO_BRANCH_2,
            "f1_sample": F1_SAMPLE_BRANCH_2,
            "f2_micro": F2_MICRO_BRANCH_2,
            "f2_macro": F2_MACRO_BRANCH_2,
            "f2_sample": F2_SAMPLE_BRANCH_2
            }
    }

print(f"\n\n ###### RESULTS {test_set} ########")
for k in result_dict.items():
    print(k[0])
    for m in k[1].items():
        print("    ", m[0], m[1])
    print("----------------------------------")
