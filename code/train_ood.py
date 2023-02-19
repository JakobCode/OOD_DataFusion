"""
Script for training the fusion network.

Call with arguments -cfg with path to config yaml file of the experiment.
"""
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch
import yaml
from data.bigearthnet_utils import normalize
from data.bigearthnet_dataloader import BigEarthDataLoader
from models.mm_resnet import ResNetFuse
from utils.utils import set_seed

parser = argparse.ArgumentParser()

parser.add_argument("-cfg", "--config_file", help="path to configuration json file.")
args = parser.parse_args()
cfg_file = args.config_file

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
cloudy_handpicked_split = cfg["data"]["splits"]["cloudy_handpicked_split"]
label_path = cfg["data"]["label_path"]
classes_ood_training = cfg["data"]["class_splits"]["classes_ood_training"]
classes_in_distribution = cfg["data"]["class_splits"]["classes_in_distribution"]
save_path_model = cfg["save_paths"]["save_path_models"]
batch_size = cfg["ood_training"]["batch_size"]
seed = cfg["setup"]["seed"]
num_epochs = cfg["ood_training"]["num_epochs"]
input_dim = cfg["data"]["splits"]
resnet_type = cfg["model"]["type"]
fusion_stage = cfg["model"]["fusion_stage"]
branch_split = cfg["model"]["branch_split"]

if save_path_model[0] == "/":
    assert save_path_model.startswith(experiment_root_path), \
        "Model path should be relative or subfolder of experiment path."
else:
    save_path_model = os.path.join(experiment_root_path, save_path_model)

os.makedirs(save_path_model, exist_ok=True)
save_path_sar = os.path.join(save_path_model, "ood_detector_sar.pth")
save_path_optical = os.path.join(save_path_model, "ood_detector_optical.pth")
model_path = os.path.join(save_path_model, "final_model.pth")

# Setup experiments
num_classes = len(classes_in_distribution)

set_seed(seed=seed)


# in-distribution
train_gen = BigEarthDataLoader(split_file=train_split,
                               root_folder=data_root_path,
                               label_path=label_path,
                               transform=normalize,
                               class_filter=classes_in_distribution)

in_generator = DataLoader(dataset=train_gen,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=False)

ood_dl = BigEarthDataLoader(split_file=train_split,
                            root_folder=data_root_path,
                            label_path=label_path,
                            transform=normalize,
                            class_filter=classes_ood_training)


ood_generator = DataLoader(dataset=ood_dl,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           drop_last=False)


print("OOD Training for model '", model_path, "'")

# Load trained fusion model, extract modality branches and build ood detectors
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
fusion_model = ResNetFuse(res_type=resnet_type,
                           fusion_stage=fusion_stage,
                           branch_split=branch_split,
                           num_classes=num_classes,
                           device=device)

fusion_model.load_state_dict(checkpoint["model_state_dict"])
fusion_model.to(device)

optical_branch = fusion_model.branch_list[1]
optical_branch.to(device)
optical_branch.eval()
sar_branch = fusion_model.branch_list[0]
sar_branch.to(device)
sar_branch.eval()

in_features_sar = np.array(sar_branch(torch.zeros([1, 2, 120, 120],\
                                      device=device)).detach().shape).prod()
in_features_opt = np.array(optical_branch(torch.zeros([1, 12,120, 120],\
                                          device=device)).detach().shape).prod()

optical_ood_model = torch.nn.Sequential(
                                torch.nn.Flatten(start_dim=1),
                                torch.nn.Linear(in_features=in_features_opt,
                                                out_features=512),
                                torch.nn.Dropout(0.1),
                                torch.nn.Tanh(),
                                torch.nn.Linear(512,64),
                                torch.nn.Tanh(),
                                torch.nn.Linear(64,1))

sar_ood_model = torch.nn.Sequential(torch.nn.Flatten(start_dim=1),
                                torch.nn.Linear(in_features=in_features_sar,
                                                out_features=512),
                                torch.nn.Dropout(0.1),
                                torch.nn.Tanh(),
                                torch.nn.Linear(512,64),
                                torch.nn.Tanh(),
                                torch.nn.Linear(64,1))

optical_ood_model.to(device)
sar_ood_model.to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(2))
optimizer_opt = torch.optim.Adam(params=optical_ood_model.parameters(), lr=1e-4)
optimizer_sar = torch.optim.Adam(params=sar_ood_model.parameters(), lr=1e-4)


# Train OOD detector for optical feature space
for epoch in range(num_epochs):
    num_steps = min(len(ood_generator), len(in_generator))
    counter = 0
    for samples_in, samples_ood in zip(in_generator, ood_generator):
        counter += 1
        print(f"Epoch [{epoch}/{num_epochs}]    -   \
                Step [{counter}/{num_steps}]           ", end="\r")

        optimizer_opt.zero_grad()

        x_in = torch.as_tensor(samples_in["img"], \
                               device=device).type(torch.FloatTensor).to(device)
        x_ood = (1+(torch.rand((samples_ood["img"].shape[0], 1, 1, 1),device=device)-0.5))*\
                torch.as_tensor(samples_ood["img"], \
                    device=device).type(torch.FloatTensor).to(device)
        y_true = torch.cat([torch.zeros((len(x_in),1)),\
                            torch.ones((len(x_ood),1))], dim=0).to(device)

        with torch.no_grad():
            f = torch.flatten(optical_branch(torch.cat([x_in[:,2:,:,:],\
                                                        x_ood[:,2:,:,:]], dim=0)),start_dim=1)

        y = optical_ood_model(f)

        loss = criterion(y, y_true)

        loss.backward()

        optimizer_opt.step()

        y_true = y_true.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

torch.save(optical_ood_model.state_dict(), save_path_optical)


# Train OOD detector for SAR feature space
for epoch in range(num_epochs):
    a = iter(in_generator)
    b = iter(ood_generator)

    num_steps = min(len(ood_generator), len(in_generator))
    counter = 0
    for samples_in, samples_ood in zip(in_generator, ood_generator):
        optimizer_sar.zero_grad()

        counter += 1
        print(f"Epoch [{epoch}/{num_epochs}]    -   \
                Step [{counter}/{num_steps}]           ", end="\r")

        x_in = torch.as_tensor(samples_in["img"], \
                               device=device).type(torch.FloatTensor).to(device)
        x_ood = torch.as_tensor(samples_ood["img"], \
                                device=device).type(torch.FloatTensor).to(device)
        y_true = torch.cat([torch.zeros((len(x_in),1), device=device),\
                            torch.ones((len(x_ood),1), device=device)], dim=0).to(device)

        with torch.no_grad():
            f = torch.flatten(sar_branch(torch.cat([x_in[:,:2,:,:],\
                                                    x_ood[:,:2,:,:]], dim=0)),start_dim=1)

        y = sar_ood_model(f)

        loss = criterion(y, y_true)

        loss.backward()

        optimizer_sar.step()

        y_true = y_true.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

torch.save(sar_ood_model.state_dict(), save_path_sar)

print(f"\n\nTraining finished. Models saved into '{save_path_model}.")
