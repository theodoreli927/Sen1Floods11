import csv
import os
import numpy as np
import rasterio
import torch
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import random
from torchgeo.trainers import SemanticSegmentationTask
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning.pytorch as pl
import matplotlib.pyplot as plt

# Returns a numpy array
def getArrFlood(fname):
    return rasterio.open(fname).read()

# To search through to find proper directory containing necessary data
from pathlib import Path
def find_directory(dirName):
    start_path = os.getcwd()
    for path in Path(start_path).rglob(dirName):
        return str(path)
    
    raise FileNotFoundError(dirName, " directory not found")


class InMemoryDataset(torch.utils.data.Dataset):

    def __init__(self, data_list, preprocess_func):
        self.data_list = data_list
        self.preprocess_func = preprocess_func

    def __getitem__(self, i):
        return self.preprocess_func(self.data_list[i])

    def __len__(self):
        return len(self.data_list)

# Used to preprocess training data
# Adds random variability for better generalization
def processAndAugment(data):
    (x, y) = data
    im, label = x.copy(), y.copy()

    # convert to PIL for easier transforms
    im1 = Image.fromarray(im[0])
    im2 = Image.fromarray(im[1])
    label = Image.fromarray(label.squeeze())

    # Get params for random transforms
    i, j, h, w = transforms.RandomCrop.get_params(im1, (256, 256))

    im1 = F.crop(im1, i, j, h, w)
    im2 = F.crop(im2, i, j, h, w)
    label = F.crop(label, i, j, h, w)
    if random.random() > 0.5:
        im1 = F.hflip(im1)
        im2 = F.hflip(im2)
        label = F.hflip(label)
    if random.random() > 0.5:
        im1 = F.vflip(im1)
        im2 = F.vflip(im2)
        label = F.vflip(label)
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])
    im = torch.stack([transforms.ToTensor()(im1).squeeze(), transforms.ToTensor()(im2).squeeze()])
    im = norm(im)
    label = transforms.ToTensor()(label).squeeze()
    if torch.sum(label.gt(0.003) * label.lt(0.004)):
        label *= 255
    label = label.round()

    return {"image": im, "mask": label}

def processTestIm(data):
    (x, y) = data
    im, label = x.copy(), y.copy()
    norm = transforms.Normalize([0.6851, 0.5235], [0.0820, 0.1102])

    # convert to PIL for easier transforms
    im_c1 = Image.fromarray(im[0]).resize((512, 512))
    im_c2 = Image.fromarray(im[1]).resize((512, 512))
    label = Image.fromarray(label.squeeze()).resize((512, 512))

    im_c1s = [
        F.crop(im_c1, 0, 0, 256, 256),
        F.crop(im_c1, 0, 256, 256, 256),
        F.crop(im_c1, 256, 0, 256, 256),
        F.crop(im_c1, 256, 256, 256, 256),
    ]
    im_c2s = [
        F.crop(im_c2, 0, 0, 256, 256),
        F.crop(im_c2, 0, 256, 256, 256),
        F.crop(im_c2, 256, 0, 256, 256),
        F.crop(im_c2, 256, 256, 256, 256),
    ]
    labels = [
        F.crop(label, 0, 0, 256, 256),
        F.crop(label, 0, 256, 256, 256),
        F.crop(label, 256, 0, 256, 256),
        F.crop(label, 256, 256, 256, 256),
    ]

    ims = [
        torch.stack((transforms.ToTensor()(x).squeeze(), transforms.ToTensor()(y).squeeze()))
        for (x, y) in zip(im_c1s, im_c2s)
    ]

    ims = [norm(im) for im in ims]
    ims = torch.stack(ims)

    labels = [(transforms.ToTensor()(label).squeeze()) for label in labels]
    labels = torch.stack(labels)

    labels = (labels > 0.5).float()

    labels = labels.unsqueeze(1)

    return {"image": ims, "mask": labels}

# %%
# Returns a list of tuples (input, label) pairs
def download_flood_water_data_from_list(l):
    i = 0
    tot_nan = 0
    tot_good = 0
    flood_data = []

    for im_fname, mask_fname in l:
        if not os.path.exists(im_fname):
            print("Couldn't find file, here is the path: ", im_fname)
            continue
        arr_x = np.nan_to_num(getArrFlood(im_fname))
        arr_y = getArrFlood(os.path.join(mask_fname))
        #arr_y[arr_y == -1] = 255
        arr_y[arr_y == -1] = 0
        
       
        arr_x = np.clip(arr_x, -50, 1)
        arr_x = (arr_x + 50) / 51

        if i % 100 == 0:
            print(im_fname, mask_fname)
        i += 1
        flood_data.append((arr_x, arr_y))
        
    all_arr_x = [data[0] for data in flood_data]

    # Flatten the arrays and find the max and min values
    all_arr_x_flat = np.concatenate([arr.flatten() for arr in all_arr_x])

    max_value = all_arr_x_flat.max()
    min_value = all_arr_x_flat.min()

    print("Max input training value:", max_value)
    print("Min value of training data:", min_value)
    return flood_data

def load_flood_train_data(root):
    subdir = find_directory("flood_handlabeled")
    fname = os.path.join(subdir, "flood_train_data.csv")
    
    training_files = []
    with open(fname) as f:
        input_subdir = "/S1Hand/"
        mask_subdir = "/LabelHand/"
        for line in csv.reader(f):
            training_files.append(
                tuple((root + input_subdir + line[0], root + mask_subdir + line[1]))
            )
    print(len(training_files))  # Prints out 252
    return download_flood_water_data_from_list(training_files)


def load_flood_valid_data(root):
    subdir = find_directory("flood_handlabeled")
    fname = os.path.join(subdir, "flood_valid_data.csv")
    validation_files = []
    with open(fname) as f:
        input_subdir = "/S1Hand/"
        mask_subdir = "/LabelHand/"
        for line in csv.reader(f):
            validation_files.append(
                tuple((root + input_subdir + line[0], root + mask_subdir + line[1]))
            )

    return download_flood_water_data_from_list(validation_files)

train_data_dir = find_directory("HandLabeled")
train_data = load_flood_train_data(train_data_dir)

train_dataset = InMemoryDataset(train_data, processAndAugment)
iter_dataset = iter(train_dataset)
next_element = next(iter_dataset)


# Collate function to create a dictionary of images and masks
def collate_fn(batch):

    images = [item["image"] for item in batch]
    masks = [item["mask"] for item in batch]

    # Stack images and masks
    images = torch.stack(images)
    masks = torch.stack(masks).float()  # Ensure masks are float
    masks = masks.unsqueeze(1)

   

    # Return a dictionary of batched images and masks
    return {"image": images, "mask": masks}


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
)

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

train_iter = iter(train_loader)
batch = next(train_iter)

images = batch["image"]
masks = batch["mask"]


dirPath = find_directory("HandLabeled")
valid_data = load_flood_valid_data(dirPath)

valid_dataset = InMemoryDataset(valid_data, processTestIm)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=True,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=lambda x: {
        "image": torch.cat([a["image"] for a in x], 0),
        "mask": torch.cat([a["mask"] for a in x], 0),
    },
    pin_memory=True,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
)
valid_iter = iter(valid_loader)

batch = next(valid_iter)
images = batch["image"]
masks = batch["mask"]

class_weights = torch.tensor([1.0, 9.0], dtype=torch.float32)
loaded_task = SemanticSegmentationTask(
    model="unet",
    backbone="resnet50",
    weights=True,
    in_channels=2,
    num_classes=1,
    lr=0.001,
    patience=10,
    freeze_backbone=False,
    loss="bce",
    class_weights=class_weights,
)
loaded_task.monitor = "train_loss"

version = "real_image_bce_w_weights_SGD"
logs_dir = os.path.join(os.getcwd(), 'logs')

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    print(f"Created 'logs' directory at: {logs_dir}")
else:
    print(f"'logs' directory already exists at: {logs_dir}")

logger = TensorBoardLogger(
    logs_dir, name="bce", version=version
)

checkpoint_callback = ModelCheckpoint(
    filename="{epoch}-{val_loss:.2f}", monitor="val_loss", save_top_k=-1
)

trainer = pl.Trainer(
    default_root_dir=os.getcwd(),
    logger=logger,
    accelerator="gpu",
    devices=[4],
    min_epochs=1,
    max_epochs=100,
    log_every_n_steps=10,
    check_val_every_n_epoch=3,
    callbacks=[checkpoint_callback],
    enable_progress_bar=False,
)

trainer.fit(loaded_task, train_dataloaders=train_loader, val_dataloaders = valid_loader)

model_dir = os.path.join(os.getcwd(), 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"Created 'models' directory at: {model_dir}")
else:
    print(f"'models' directory already exists at: {model_dir}")

model_filename = version + ".pth"
model_path = os.path.join(model_dir, model_filename)
print("FINISHED TRAINING AND SAVED THE MODEL")

torch.save(loaded_task.model.state_dict(), model_path)

