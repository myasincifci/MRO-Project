import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

class Classifier(pl.LightningModule):
    def __init__(self, backbone, bb_out_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(bb_out_dim, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)