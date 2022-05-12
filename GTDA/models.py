import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from .GTDA_utils import *
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torchmetrics import Accuracy
import torchvision.models as torch_models
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn


class GenericCNNData(pl.LightningDataModule):
    def __init__(
        self, args, train_transform=None, test_transform=None, 
        train_dataset=None, test_dataset=None, train_subdir='train', test_subdir='test'):
        super().__init__()
        self.args = args
        if train_dataset is None:
            self.train_dataset = ImageFolder(root=f"{self.args.data_root}/{train_subdir}/", transform=train_transform)
        else:
            self.train_dataset = train_dataset
        if test_dataset is None:
            self.test_dataset = ImageFolder(root=f"{self.args.data_root}/{test_subdir}/", transform=test_transform)
        else:
            self.test_dataset = test_dataset

    def train_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=self.args.shuffle,
            drop_last=self.args.drop_last,
            pin_memory=self.args.pin_memory,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=self.args.drop_last,
            pin_memory=self.args.pin_memory,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class ResNetModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        model = torch_models.resnet50(pretrained=True)
        num_filters = model.fc.in_features
        layers = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_filters, args.num_classes)

    def forward(self, batch):
        images, labels = batch
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(images).flatten(1)
        predictions = self.classifier(representations)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss, on_step=False, on_epoch=True)
        self.log("acc/train", accuracy, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss, on_step=False, on_epoch=True)
        self.log("acc/val", accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        parameters = self.classifier.parameters()
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        scheduler = {
            "scheduler": StepLR(
                optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma,
            ),
            "interval": "epoch",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

class LinearClassifier(torch.nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x