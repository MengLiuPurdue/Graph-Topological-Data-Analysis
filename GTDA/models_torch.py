import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from .GTDA_utils import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import pytorch_lightning as pl
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.nn import GCNConv
import copy
from torchmetrics import AUROC, Accuracy
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torchvision.models as torch_models
import sys
sys.path.append('PyTorch_CIFAR10')
# sys.path.append('/home/meng/Dropbox/Mengs_Files/Research/understand-GNN/understand_GNN/pytorch-cifar100')
# sys.path.append('/home/meng/Dropbox/Mengs_Files/Research/understand-GNN/understand_GNN/DrNAS/DARTS-space')
# sys.path.append('/home/meng/Dropbox/Mengs_Files/Research/understand-GNN/understand_GNN/DrNAS')
# from model_search import Network
from schduler import WarmupCosineLR
from cifar10_models.resnet import resnet50 as cifar10_resnet50
# from DARTS_utils import *
# from architect import Architect
# from net2wider import configure_optimizer, configure_scheduler
# from cifar100_utils import WarmUpLR
import logging
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
import sys
sys.path.append("esvit")
from esvit.config import config
from esvit.config import update_config
from esvit.models import build_model
import esvit.utils as esvit_utils
import os


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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

def select_classifier(name, args, criterion=None):
    if name == "SAGE":
        return SAGE(
            args.in_channels, 
            args.hidden_channels, 
            args.out_channels, 
            args.num_layers,
            args.dropout
        )
    elif name == "cifar10_resnet50":
        return cifar10_resnet50(pretrained=args.pretrained)
    elif name == "torch_resnet50":
        model = torch_models.resnet50(pretrained=args.pretrained)
        return model
    elif name == "torch_alexnet":
        model = torch_models.alexnet(pretrained=args.pretrained)
        return model
    elif name == "torch_resnet18":
        model = torch_models.resnet18(pretrained=args.pretrained)
        return model
    elif name == "DARTS":
        model = Network(
            args.init_channels, 
            args.nclass, 
            args.layers, 
            criterion=criterion, 
            k=args.k,
            reg_type=args.reg_type, 
            reg_scale=args.reg_scale)
        return model
    else:
        raise NotImplementedError

class OGBNProteinsData(pl.LightningDataModule):
    def __init__(self, args, split_idx=None):
        super().__init__()
        self.args = args
        if split_idx is None:
            dataset = PygNodePropPredDataset(
                name = 'ogbn-proteins', 
                transform=T.ToSparseTensor(), 
                root=f"{args.data_root}/dataset")
            self.split_idx = dataset.get_idx_split()
            del dataset
        else:
            self.split_idx = split_idx

    def train_dataloader(self):
        train_ids = torch.tensor(
            self.split_idx['train'],dtype=torch.long)
        dataloader = torch.utils.data.DataLoader(
            train_ids,
            batch_size=train_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        valid_ids = torch.tensor(
            self.split_idx['valid'],dtype=torch.long)
        dataloader = torch.utils.data.DataLoader(
            valid_ids,
            batch_size=valid_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        test_ids = torch.tensor(
            self.split_idx['valid'],dtype=torch.long)
        dataloader = torch.utils.data.DataLoader(
            test_ids,
            batch_size=test_ids.shape[0],
            shuffle=False,
        )
        return dataloader

class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args, trainnodes = None, validnodes = None, testnodes = None):
        super().__init__()
        self.args = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.trainnodes = trainnodes
        self.validnodes = validnodes
        self.testnodes = testnodes

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.args.data_root, train=True, transform=transform, download=self.args.download)
        if self.trainnodes is not None:
            dataset = [dataset[i] for i in self.trainnodes]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.args.data_root, train=False, transform=transform, download=self.args.download)
        if self.validnodes is not None:
            dataset = [dataset[i] for i in self.validnodes]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR100Data(pl.LightningDataModule):
    def __init__(self, args, trainnodes = None, validnodes = None, testnodes = None):
        super().__init__()
        self.args = args
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.trainnodes = trainnodes
        self.validnodes = validnodes
        self.testnodes = testnodes

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root=self.args.data_root, train=True, transform=transform, download=self.args.download)
        if self.trainnodes is not None:
            dataset = [dataset[i] for i in self.trainnodes]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root=self.args.data_root, train=False, transform=transform, download=self.args.download)
        if self.validnodes is not None:
            dataset = [dataset[i] for i in self.validnodes]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class SBMData(pl.LightningDataModule):
    def __init__(self, args, split_idx):
        super().__init__()
        self.args = args
        self.split_idx = split_idx

    def train_dataloader(self):
        train_ids = self.split_idx['train']
        dataloader = torch.utils.data.DataLoader(
            train_ids,
            batch_size=train_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        valid_ids = self.split_idx['valid']
        dataloader = torch.utils.data.DataLoader(
            valid_ids,
            batch_size=valid_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        test_ids = self.split_idx['test']
        dataloader = torch.utils.data.DataLoader(
            test_ids,
            batch_size=test_ids.shape[0],
            shuffle=False,
        )
        return dataloader


class ComputersData(pl.LightningDataModule):
    def __init__(self, args, split_idx):
        super().__init__()
        self.args = args
        self.split_idx = split_idx

    def train_dataloader(self):
        train_ids = self.split_idx['train']
        dataloader = torch.utils.data.DataLoader(
            train_ids,
            batch_size=train_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        valid_ids = self.split_idx['valid']
        dataloader = torch.utils.data.DataLoader(
            valid_ids,
            batch_size=valid_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        test_ids = self.split_idx['test']
        dataloader = torch.utils.data.DataLoader(
            test_ids,
            batch_size=test_ids.shape[0],
            shuffle=False,
        )
        return dataloader


class GenericGNNData(pl.LightningDataModule):
    def __init__(self, args, split_idx):
        super().__init__()
        self.args = args
        self.split_idx = split_idx

    def train_dataloader(self):
        train_ids = self.split_idx['train']
        dataloader = torch.utils.data.DataLoader(
            train_ids,
            batch_size=train_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def val_dataloader(self):
        valid_ids = self.split_idx['valid']
        dataloader = torch.utils.data.DataLoader(
            valid_ids,
            batch_size=valid_ids.shape[0],
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        test_ids = self.split_idx['test']
        dataloader = torch.utils.data.DataLoader(
            test_ids,
            batch_size=test_ids.shape[0],
            shuffle=False,
        )
        return dataloader

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

class GenericData(pl.LightningDataModule):
    def __init__(
        self, args, train_dataset, test_dataset):
        super().__init__()
        self.args = args
        self.train_dataset = train_dataset
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


class OGBNProteinsModule(pl.LightningModule):
    def __init__(self, args, node_feats = None, data = None, model = None):
        super().__init__()
        self.args = args
        if model is None:
            self.model = select_classifier(self.args.classifier, args)
        else:
            self.model = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        if data is None:
            dataset = PygNodePropPredDataset(name = 'ogbn-proteins', transform=T.ToSparseTensor(), root=f"{args.data_root}/dataset")
            self.data = dataset[0]
        else:
            self.data = data
        if node_feats is None:
            self.data.x = self.data.adj_t.mean(dim=1)
        else:
            self.data.x = node_feats
        self.data.adj_t.set_value_(None)
        self.labels = self.data.y.detach()
        self.data.y = self.data.y.to(torch.float)
        self.data.to(f"cuda:{args.gpu_id}")
        self.labels = self.labels.to(f"cuda:{args.gpu_id}")
        self.evaluator = Evaluator(name='ogbn-proteins')
        self.rocauc = [0,0,0]

    def forward(self, batch, is_test):
        y_pred = self.model(self.data.x, self.data.adj_t)[batch]
        loss = self.criterion(y_pred, self.data.y[batch])
        if self.current_epoch%self.args.compute_auc_every_n_step == 0 or is_test:
            rocauc = self.evaluator.eval({
                'y_true': self.labels[batch],
                'y_pred': y_pred,
            })['rocauc']
        else:
            rocauc = 0
        return loss, rocauc

    def training_step(self, batch, batch_nb):
        loss,rocauc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.rocauc[0] = rocauc
        self.log("loss/train", loss, prog_bar=True)
        self.log("auc/train", self.rocauc[0], prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_nb):
        loss,rocauc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.rocauc[1] = rocauc
        self.log("loss/valid", loss, prog_bar=True)
        self.log("auc/valid", self.rocauc[1], prog_bar=True)

    def test_step(self, batch, batch_nb):
        _,rocauc = self.forward(batch,True)
        self.rocauc[2] = rocauc
        self.log("auc/test", self.rocauc[2], prog_bar=True)
    
    def free_gpu_memory(self):
        self.data.to('cpu')
        self.model.to('cpu')
        self.labels = self.labels.to('cpu')
        self.to('cpu')
        torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr, 
            weight_decay=self.args.weight_decay)

class SBMModule(pl.LightningModule):
    def __init__(self, args, data, node_feats = None):
        super().__init__()
        self.args = args
        self.model = select_classifier(self.args.classifier, args)
        self.criterion = torch.nn.NLLLoss()
        self.data = data
        if node_feats is not None:
            self.data.x = node_feats
        self.labels = self.data.y.detach()
        self.data.to(f"cuda:{args.gpu_id}")
        self.labels = self.labels.to(f"cuda:{args.gpu_id}")
        self.evaluator = Accuracy()
        self.acc = [0,0,0]

    def forward(self, batch, is_test):
        m = nn.LogSoftmax(dim=1)
        y_pred = m(self.model(self.data.x, self.data.adj_t)[batch])
        loss = self.criterion(y_pred, self.data.y[batch])
        if self.current_epoch%self.args.compute_auc_every_n_step == 0 or is_test:
            acc = self.evaluator(torch.argmax(y_pred,1), self.labels[batch])
        else:
            acc = 0
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss,acc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.acc[0] = acc
        self.log("loss/train", loss, prog_bar=True)
        self.log("acc/train", self.acc[0], prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_nb):
        loss,acc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.acc[1] = acc
        self.log("loss/valid", loss, prog_bar=True)
        self.log("acc/valid", self.acc[1], prog_bar=True)

    def test_step(self, batch, batch_nb):
        _,acc = self.forward(batch,True)
        self.acc[2] = acc
        self.log("acc/test", self.acc[2], prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)
        # scheduler = ExponentialLR(optimizer, gamma=self.args.gamma)
        # return ([optimizer],[scheduler])
        return optimizer

class ComputersModule(pl.LightningModule):
    def __init__(self, args, data, node_feats = None):
        super().__init__()
        self.args = args
        self.model = select_classifier(self.args.classifier, args)
        self.criterion = torch.nn.NLLLoss()
        self.data = data
        if node_feats is not None:
            self.data.x = node_feats
        self.labels = self.data.y.detach()
        self.data.to(f"cuda:{args.gpu_id}")
        self.labels = self.labels.to(f"cuda:{args.gpu_id}")
        self.evaluator = Accuracy()
        self.acc = [0,0,0]

    def forward(self, batch, is_test):
        m = nn.LogSoftmax(dim=1)
        y_pred = m(self.model(self.data.x, self.data.adj_t)[batch])
        loss = self.criterion(y_pred, self.data.y[batch])
        if self.current_epoch%self.args.compute_auc_every_n_step == 0 or is_test:
            acc = self.evaluator(torch.argmax(y_pred,1), self.labels[batch])
        else:
            acc = 0
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss,acc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.acc[0] = acc
        self.log("loss/train", loss, prog_bar=True)
        self.log("acc/train", self.acc[0], prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_nb):
        loss,acc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.acc[1] = acc
        self.log("loss/valid", loss, prog_bar=True)
        self.log("acc/valid", self.acc[1], prog_bar=True)

    def test_step(self, batch, batch_nb):
        _,acc = self.forward(batch,True)
        self.acc[2] = acc
        self.log("acc/test", self.acc[2], prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)
        # scheduler = ExponentialLR(optimizer, gamma=self.args.gamma)
        # return ([optimizer],[scheduler])
        return optimizer


class CIFAR10Module(pl.LightningModule):
    def __init__(self, args, model = None):
        super().__init__()
        self.args = args

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        if model is None:
            self.model = select_classifier(self.args.classifier, args)
        else:
            self.model = model

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.args.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]


class CIFAR100Module(pl.LightningModule):
    def __init__(self, args, model = None):
        super().__init__()
        self.args = args

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        if model is None:
            self.model = select_classifier(self.args.classifier, args)
        else:
            self.model = model

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
        )
        total_steps = self.args.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
    

class GenericGNNModule(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        self.args = args
        self.model = select_classifier(self.args.classifier, args)
        self.criterion = torch.nn.NLLLoss()
        self.data = data
        self.labels = self.data.y.detach()
        self.data.to(f"cuda:{args.gpu_id}")
        self.labels = self.labels.to(f"cuda:{args.gpu_id}")
        self.evaluator = Accuracy()
        self.acc = [0,0,0]

    def forward(self, batch, is_test):
        m = nn.LogSoftmax(dim=1)
        y_pred = m(self.model(self.data.x, self.data.adj_t)[batch])
        loss = self.criterion(y_pred, self.data.y[batch])
        if self.current_epoch%self.args.compute_auc_every_n_step == 0 or is_test:
            acc = self.evaluator(torch.argmax(y_pred,1), self.labels[batch])
        else:
            acc = 0
        return loss, acc

    def training_step(self, batch, batch_nb):
        loss,acc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.acc[0] = acc
        self.log("loss/train", loss, prog_bar=True)
        self.log("acc/train", self.acc[0], prog_bar=True)
        return loss
        

    def validation_step(self, batch, batch_nb):
        loss,acc = self.forward(batch,False)
        if self.current_epoch%self.args.compute_auc_every_n_step == 0:
            self.acc[1] = acc
        self.log("loss/valid", loss, prog_bar=True)
        self.log("acc/valid", self.acc[1], prog_bar=True)

    def test_step(self, batch, batch_nb):
        _,acc = self.forward(batch,True)
        self.acc[2] = acc
        self.log("acc/test", self.acc[2], prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)
        return optimizer

class DARTSModule(object):
    def __init__(self, args, model = None, logger = None, train_queue = None, valid_queue = None):
        self.args = args
        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(sum(args.train_epochs)), eta_min=args.learning_rate_min)
        self.model = model
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.architect = Architect(model, args)
    
    def fit(self):
        epoch = 0
        for i, current_epochs in enumerate(self.args.train_epochs):
            for e in range(current_epochs):
                lr = self.scheduler.get_lr()[0]
                metrics = {
                    "epoch": epoch,
                    "lr": lr,
                }
                self.logger.log_metrics(metrics, step=epoch)
                self.logger.save()

                genotype = self.model.genotype()
                print(f'genotype = {genotype}')
                self.model.show_arch_parameters()

                # training
                train_acc, train_obj = self.train(
                    self.train_queue, self.valid_queue, self.model, self.architect, self.criterion, self.optimizer, lr, e)

                # validation
                valid_acc, valid_obj = self.infer(self.valid_queue, self.model, self.criterion, e)
                metrics = {
                    'acc_train': train_acc,
                    'acc_valid': valid_acc,
                }
                self.logger.log_metrics(metrics,step=epoch)
                self.logger.save()
                
                epoch += 1
                self.scheduler.step()
            
            if not i == len(self.args.train_epochs) - 1:
                self.model.pruning(self.args.num_keeps[i+1])
                # architect.pruning([model.mask_normal, model.mask_reduce])
                self.model.wider(self.args.ks[i+1])
                # self.optimizer = configure_optimizer(self.optimizer, torch.optim.SGD(
                #     self.model.parameters(),
                #     self.args.learning_rate,
                #     momentum=self.args.momentum,
                #     weight_decay=self.args.weight_decay))
                self.scheduler = configure_scheduler(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, float(sum(self.args.train_epochs)), eta_min=self.args.learning_rate_min))
                print('pruning finish, %d ops left per edge'.format(self.args.num_keeps[i+1]))
                print('network wider finish, current pc parameter %d'.format(self.args.ks[i+1]))

        genotype = self.model.genotype()
        print(f'genotype = {genotype}')
        self.model.show_arch_parameters()

    def train(self, train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):
            model.train()
            n = input.size(0)
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)

            if epoch >= 10:
                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=self.args.unrolled)
            optimizer.zero_grad()
            architect.optimizer.zero_grad()

            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            architect.optimizer.zero_grad()
            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % self.args.report_freq == 0:
                metrics = {
                    'train_objs': objs.avg,
                    'train_top1': top1.avg,
                    'train_top5': top5.avg,
                }
                self.logger.log_metrics(metrics,step=step+epoch*len(train_queue))
                self.logger.save()
            if 'debug' in self.args.save:
                break

        return top1.avg, objs.avg

    def infer(self, valid_queue, model, criterion, epoch):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        model.eval()

        with torch.no_grad():
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda(non_blocking=True)
                
                logits = model(input)
                loss = criterion(logits, target)

                prec1, prec5 = accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)

                if step % self.args.report_freq == 0:
                    metrics = {
                        'valid_objs': objs.avg,
                        'valid_top1': top1.avg,
                        'valid_top5': top5.avg,
                    }
                    self.logger.log_metrics(metrics,step=step+epoch*len(valid_queue))
                    self.logger.save()
                if 'debug' in self.args.save:
                    break

        return top1.avg, objs.avg

class GenericCNNModule(pl.LightningModule):
    def __init__(self, args, model = None, eval_mode = False):
        super().__init__()
        self.args = args
        self.eval_mode = eval_mode

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        if model is None:
            self.model = select_classifier(self.args.classifier, args)
        else:
            self.model = model
        if self.eval_mode:
            if 'resnet' in self.args.classifier:
                num_filters = self.model.fc.in_features
                layers = list(self.model.children())[:-1]
                self.feature_extractor = nn.Sequential(*layers)
                self.classifier = nn.Linear(num_filters, args.num_classes)
            elif 'alexnet' in self.args.classifier:
                new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
                self.model.classifier = new_classifier
                self.feature_extractor = self.model
                self.classifier = nn.Sequential(
                    nn.Linear(in_features=4096, out_features=args.num_classes, bias=True)
                )

    def forward(self, batch):
        images, labels = batch
        if self.eval_mode:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(images).flatten(1)
            predictions = self.classifier(representations)
        else:
            predictions = self.model(images)
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
        if self.eval_mode:
            parameters = self.classifier.parameters()
        else:
            parameters = self.model.parameters()
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        if self.args.scheduler == "StepLR":
            scheduler = {
                "scheduler": StepLR(
                    optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma,
                ),
                "interval": "epoch",
                "name": "learning_rate",
            }
        elif self.args.scheduler == "WarmupCosineLR":
            total_steps = self.args.max_epochs * len(self.train_dataloader())
            scheduler = {
                "scheduler": WarmupCosineLR(
                    optimizer, warmup_epochs=total_steps * self.args.lr_warmup, max_epochs=total_steps
                ),
                "interval": "step",
                "name": "learning_rate",
            }
        return [optimizer], [scheduler]

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class ESVITLinear(object):
    def __init__(self,args):
        checkpoint = torch.load(args.checkpoint)
        encoder_args = checkpoint['args']
        encoder_args.cfg = f"esvit/{encoder_args.cfg}"
        update_config(config, encoder_args)
        swin_spec = config.MODEL.SPEC
        embed_dim=swin_spec['DIM_EMBED']
        self.depths=swin_spec['DEPTHS']
        self.num_heads=swin_spec['NUM_HEADS'] 
        self.args = args
        self.encoder_model = build_model(config, is_teacher=True)
        esvit_utils.load_pretrained_weights(
            self.encoder_model, "models/EsViT/checkpoint_best.pth", "teacher", encoder_args.arch, encoder_args.patch_size)
        num_features = []
        for i, d in enumerate(self.depths):
            num_features += [int(embed_dim * 2 ** i)] * d 
        
        num_features_linear = sum(num_features[-args.n_last_blocks:])
        self.linear_classifier = LinearClassifier(num_features_linear, args.num_labels)
        # set optimizer
        self.optimizer = torch.optim.SGD(
            self.linear_classifier.parameters(),
            args.lr * (args.batch_size * esvit_utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs, eta_min=0)
        self.encoder_model.cuda()
        self.linear_classifier.cuda()
    
    def train(self,trainloader,epoch):
        self.linear_classifier.train()
        metric_logger = esvit_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', esvit_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        for (inp, target) in metric_logger.log_every(trainloader, 20, header):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # forward
            with torch.no_grad():
                output = self.encoder_model.forward_return_n_last_blocks(inp, self.args.n_last_blocks, self.args.avgpool_patchtokens, self.depths)
            
            # print(f'output {output.shape}')
            output = self.linear_classifier(output)

            # compute cross entropy loss
            loss = nn.CrossEntropyLoss()(output, target)

            # compute the gradients
            self.optimizer.zero_grad()
            loss.backward()

            # step
            self.optimizer.step()

            # log 
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    @torch.no_grad()
    def validate_network(self,val_loader):
        self.linear_classifier.eval()
        metric_logger = esvit_utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        for inp, target in metric_logger.log_every(val_loader, 20, header):
            # move to gpu
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = self.encoder_model.forward_return_n_last_blocks(inp, self.args.n_last_blocks, self.args.avgpool_patchtokens, self.depths)
            output = self.linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, target)

            acc1, acc5 = esvit_utils.accuracy(output, target, topk=(1, 5))

            batch_size = inp.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def fit(self,train_loader,val_loader):
        best_acc = 0
        if self.args.load_from_file == False:
            for epoch in range(self.args.epochs):

                train_stats = self.train(train_loader, epoch)
                self.scheduler.step()

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch}
                if epoch % self.args.val_freq == 0 or epoch == self.args.epochs - 1:
                    test_stats = self.validate_network(val_loader)
                    print(f"Accuracy at epoch {epoch} of the network on the test images: {test_stats['acc1']:.1f}%")
                    best_acc = max(best_acc, test_stats["acc1"])
                    print(f'Max accuracy so far: {best_acc:.2f}%')
                    log_stats = {**{k: v for k, v in log_stats.items()},
                                **{f'test_{k}': v for k, v in test_stats.items()}}
                    if best_acc == test_stats["acc1"]:
                        save_dict = {
                            "epoch": epoch + 1,
                            "state_dict": self.linear_classifier.state_dict(),
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "best_acc": best_acc,
                        }
                    torch.save(save_dict, os.path.join(self.args.output_dir, "checkpoint.pth.tar"))
        best_state_dict = torch.load(os.path.join(self.args.output_dir, "checkpoint.pth.tar"))['state_dict']
        self.linear_classifier.load_state_dict(best_state_dict)
        self.linear_classifier.cuda()
        test_stats = self.validate_network(val_loader)
        best_acc = max(best_acc, test_stats["acc1"])
        print("Training of the supervised linear classifier on frozen features completed.\n"
                    "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


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