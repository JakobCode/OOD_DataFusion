"""
Script for training the fusion network.

Call with arguments -cfg with path to config yaml file of the experiment.
"""

import argparse
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import yaml
from data.bigearthnet_utils import normalize
from data.bigearthnet_dataloader import BigEarthDataLoader
from models.mm_resnet import ResNetFuse
from utils.utils import set_seed

parser = argparse.ArgumentParser()

parser.add_argument("-cfg",
                    "--config_file", 
                    help="path to configuration json file.")
args = parser.parse_args()
cfg_file = args.config_file

assert os.path.exists(cfg_file), \
       f"Configuration file '{cfg_file}' does not exist!"

with open(cfg_file, "rb") as file:
    cfg = yaml.safe_load(file)

device="cuda" if torch.cuda.is_available() else "cpu"

# Put this in config file
num_workers = cfg["num_workers"]
experiment_root_path = cfg["experiment_root_path"]
data_root_path = cfg["data_root_path"]
train_split = cfg["data"]["splits"]["train_split"]
val_split = cfg["data"]["splits"]["val_split"]
label_path = cfg["data"]["label_path"]
classes_in_distribution = cfg["data"]["class_splits"]["classes_in_distribution"]
seed = cfg["setup"]["seed"]
num_epochs = cfg["main_training"]["num_epochs"]
batch_size = cfg["main_training"]["batch_size"]
learning_rate = cfg["main_training"]["learning_rate"]
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

# Setup experiments
num_classes = len(classes_in_distribution)

set_seed(seed=seed)

train_gen = BigEarthDataLoader(split_file=train_split,
                               root_folder=data_root_path,
                               label_path=label_path,
                               transform=normalize,
                               class_filter=classes_in_distribution)

train_data_loader = DataLoader(dataset=train_gen,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)

# Prepare validation data loader
val_gen = BigEarthDataLoader(split_file=val_split,
                             root_folder=data_root_path,
                             label_path=label_path,
                             transform=normalize,
                             class_filter=classes_in_distribution)

val_data_loader = DataLoader(dataset=val_gen,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

# Load model
model = ResNetFuse(res_type=resnet_type,
                    fusion_stage=fusion_stage,
                    branch_split=branch_split,
                    num_classes=num_classes,
                    device=device)
model.to(device)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())

criterion = torch.nn.BCEWithLogitsLoss(reduce=False)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

best_val_loss = 99999999

# Training loop
for epoch in range(num_epochs):
    # Evaluate on training split
    print("START EPOCH ", epoch+1)

    pred_list_train_both = []
    pred_list_train_sar = []
    pred_list_train_optical = []
    true_list_train = []

    pred_list_val_both = []
    pred_list_val_sar = []
    pred_list_val_optical = []
    true_list_val = []
    train_loss = []

    model.train()
    for data in tqdm(train_data_loader, "Training Data"):

        optimizer.zero_grad()

        x = data["img"]
        y_true = data["label_numeric"]

        x = torch.as_tensor(x, device=device, dtype=torch.float32)
        y_true = torch.as_tensor(y_true, device=device, dtype=torch.float32)

        y_pred_fuse = model(x)

        num_samples_batch = y_true.shape[0]
        loss1 = criterion(y_pred_fuse[:num_samples_batch], y_true)
        loss2 = criterion(y_pred_fuse[num_samples_batch:-num_samples_batch], y_true)
        loss3 = criterion(y_pred_fuse[-num_samples_batch:], y_true)

        loss = 2 * loss1 + loss2 + loss3

        loss.mean().backward()

        optimizer.step()

        train_loss += list(loss.detach())

    model.eval()
    val_loss = []
    with torch.no_grad():
        for data in tqdm(val_data_loader, "Validation Data"):

            x = data["img"]
            y_true = data["label_numeric"]

            x = torch.as_tensor(x, device=device, dtype=torch.float32)
            y_true = torch.as_tensor(y_true, device=device, dtype=torch.float32)

            y_pred_fuse = model(x)

            num_samples_batch = y_true.shape[0]
            loss1 = criterion(y_pred_fuse[:num_samples_batch], y_true)
            loss2 = criterion(y_pred_fuse[num_samples_batch:-num_samples_batch], y_true)
            loss3 = criterion(y_pred_fuse[-num_samples_batch:], y_true)

            val_loss += list(2 * loss1 + loss2 + loss3)

        cur_val_loss = torch.cat(val_loss, 0).mean()
        train_loss = torch.cat(train_loss, 0).mean()

        print(f"\nTraining Loss Epoch {epoch+1}:   {train_loss}")
        if cur_val_loss < best_val_loss:
            print(f"Validation Loss Epoch {epoch+1}:"+
                  f" {cur_val_loss}    (improved from {best_val_loss})")
            best_val_loss = cur_val_loss
            torch.save({
                "epoch": epoch,
                "val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                }, os.path.join(save_path_model, f"epoch{epoch}_loss_improvement.pth"))
        else:
            print(f"Validation Loss Epoch {epoch+1}:   "+
                  f"{cur_val_loss}    (best loss: {best_val_loss})")
        print()

torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    }, os.path.join(save_path_model, "final_model.pth"))

print(f"\n\nTraining finished. Models saved into '{save_path_model}.")

