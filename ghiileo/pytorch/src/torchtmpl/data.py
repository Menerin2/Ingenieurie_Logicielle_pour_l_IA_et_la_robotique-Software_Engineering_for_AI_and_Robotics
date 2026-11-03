# coding: utf-8

import os
import pathlib
import numpy as np
import torch
import torch.utils.data
from enum import Enum
from torchvision import transforms
import matplotlib.pyplot as plt
import sys


class Fold(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


#  Dataset 

class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, root, fold: Fold, transform=None, valid_ratio=0.2):

        # Convertir en Path si string
        if isinstance(root, str):
            root = pathlib.Path(root)

        self.transform = transform
        self.datafiles = {}       
        self.num_frames = 0       
        self.fold = fold

        # Charger tous les fichiers .npz dans sous-dossiers
        for f in root.glob("**/*.npz"):
            data = np.load(f, allow_pickle=True)
            frames = data["frames"]      # liste de dict {left,front,right}
            steerings = data["steerings"]

            # Split train / valid
            n = len(frames)
            split_idx = int((1 - valid_ratio) * n)

            if fold == Fold.TRAIN:
                frames = frames[:split_idx]
                steerings = steerings[:split_idx]

            elif fold == Fold.VALID:
                frames = frames[split_idx:]
                steerings = steerings[split_idx:]

            # Fold.TEST 

            self.datafiles[f] = {
                "frames": frames,
                "steerings": steerings,
                "num_frames": len(frames),
            }
            self.num_frames += len(frames)

        # Cache 
        self.cached = {
            "filename": None,
            "frames": None,
            "steerings": None,
        }

    def __len__(self):
        return self.num_frames
    

    def __getitem__(self, idx):

        # Fonction interne simple de préprocessing
        def preprocess(frame):
            frame = torch.from_numpy(frame).float() / 255.0  
            frame = frame.permute(2, 0, 1)                   
            return frame

        # Parcourt des fichiers jusqu’à trouver celui qui contient l’index idx
        for f, block in self.datafiles.items():
            if idx < block["num_frames"]:

                # Mettre à jour le cache si ce n’est plus le même fichier
                if self.cached["filename"] != f:
                    self.cached = {
                        "filename": f,
                        "frames": block["frames"],
                        "steerings": block["steerings"],
                    }

                frames = self.cached["frames"]
                steerings = self.cached["steerings"]

                # Charger left, front, right
                sample = frames[idx]
                X_left = preprocess(sample["left"])
                X_forward = preprocess(sample["front"])
                X_right = preprocess(sample["right"])

                # Commande à prédire
                Y = float(steerings[idx])

                break

            idx -= block["num_frames"]

        # Appliquer transformation si fournie
        if self.transform:
            X_left = self.transform(X_left)
            X_forward = self.transform(X_forward)
            X_right = self.transform(X_right)

        # Retour : un dictionnaire 
        X = {
            "left": X_left,
            "forward": X_forward,
            "right": X_right
        }

        return X, Y



# show_samples 

def show_samples(datapath):

    input_transform = transforms.Compose([
        transforms.Resize((128, 128))
    ])

    dataset = NPZDataset(
        root=datapath,
        fold=Fold.TRAIN,
        transform=input_transform,
        valid_ratio=0.2,
    )

    nsamples = 5
    fig = plt.figure(constrained_layout=True, figsize=(10, 30))
    fig.suptitle("Samples from the dataset")

    subfigs = fig.subfigures(nrows=nsamples, ncols=1)

    for row, subfig in enumerate(subfigs):
        idx = np.random.randint(0, len(dataset))
        X, y = dataset[idx]

        subfig.suptitle(f"Sample {idx}, Command to predict : {y}")

        axs = subfig.subplots(nrows=1, ncols=3)
        axs[0].imshow(X["left"].permute(1, 2, 0))
        axs[1].imshow(X["forward"].permute(1, 2, 0))
        axs[2].imshow(X["right"].permute(1, 2, 0))

    plt.show()



def get_dataloaders(config, use_cuda):
    """
    Build train and validation dataloaders
    """

    datapath = config.get("datapath", "~/ros2_ws/dataset")
    batch_size = config.get("batch_size", 16)
    valid_ratio = config.get("valid_ratio", 0.2)

    datapath = os.path.expanduser(datapath)

    # Transform (resize only, normalization already done)
    transform = transforms.Compose([
        transforms.Resize((128, 128))
    ])

    # Datasets
    train_dataset = NPZDataset(
        root=datapath,
        fold=Fold.TRAIN,
        transform=transform,
        valid_ratio=valid_ratio,
    )

    valid_dataset = NPZDataset(
        root=datapath,
        fold=Fold.VALID,
        transform=transform,
        valid_ratio=valid_ratio,
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    # Model info
   
    sample_X, _ = train_dataset[0]
    C, H, W = sample_X["forward"].shape
    
    input_size = (3 * C, H, W)
    num_classes = 1              # steering angle (regression)

    return train_loader, valid_loader, input_size, num_classes



if __name__ == "__main__":

    if len(sys.argv) == 2:
        datapath = sys.argv[1]
    else:
        datapath = "~/ros2_ws/dataset" ####

    datapath = os.path.expanduser(datapath)

    print(f"Loading dataset from: {datapath}")
    show_samples(datapath)
