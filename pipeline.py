import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import resnet18, ResNet18_Weights

import pytorch_lightning as pl

from models import Classifier

def main():
    BS = 128

    transform = T.Compose([
        T.Resize(126, T.InterpolationMode.NEAREST_EXACT),
        T.ToTensor()
    ])

    dataset = ImageFolder('dataset', transform=transform)
    train_set, test_set = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_set, BS, shuffle=True)
    train_loader = DataLoader(test_set, BS, shuffle=False)

    backbone = resnet18(ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    model = Classifier(backbone, 512)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=1_000,
    )

    trainer.fit(model, train_loader)

if __name__ == '__main__':
    main()