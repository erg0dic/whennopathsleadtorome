from tkinter import X
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import math
from src.train import device
# from the graphlog repo

import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from src.train import DataLoader
from pytorch_lightning.core import LightningModule
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch.nn import Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GATv2Conv, RGATConv
from src.utils import get_acc_multihot
import torchmetrics


class SupervisedRGCN(LightningModule):
    """
    Sample model to show how to define a template.
    """
    def __init__(self, config=None):
        """
        Pass in hyperparameters as a `argparse.Namespace` or a `dict` to the model.
        """
        # init superclass
        super().__init__()
        # self.save_hyperparameters()
        self.config = config
        self.batch_size = 16

        # if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

        self.num_layers=8
        # build model
        self.__build_model()
        self.lossf = nn.BCEWithLogitsLoss(reduction='mean')

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout the model.
        """
        dim=64
        self.dim=dim
        self.rgcn_layers = []
        for l in range(self.num_layers):
            in_channels = dim
            out_channels = dim
            num_bases = dim

            self.rgcn_layers.append(
                self.config.convtype(
                    in_channels,
                    out_channels,
                    self.config.num_relations,
                    num_bases,
                    root_weight=True,
                    bias=True,
                ).to(device)
            )

        self.rgcn_layers = nn.ModuleList(self.rgcn_layers).to(device)
        self.classfier = []
        inp_dim = (
            dim * 1
            + dim
        )
        outp_dim = dim
        for l in range(2 - 1):
            self.classfier.append(nn.Linear(inp_dim, outp_dim, device=device))
            self.classfier.append(nn.ReLU())
            inp_dim = outp_dim
        self.classfier.append(nn.Linear(inp_dim, self.config.num_relations, device=device))
        self.classfier = nn.Sequential(*self.classfier)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, batch):
        """
        We use random node embeddings for each forward call.
        """
        data = batch
        # initialize nodes randomly
        node_emb = torch.zeros(size=(100, self.dim), device=device)
        torch.nn.init.xavier_uniform_(node_emb, gain=1.414)
        x = F.embedding(data.x, node_emb)
        x = x.squeeze(1)

        # get edge attributes
        edge_types = data.edge_type
        for nr in range(self.num_layers - 1):
            # x = F.dropout(x, p=0., training=self.training)
            x = self.rgcn_layers[0](x, data.edge_index, edge_types)
            x = F.relu(x)
        x = self.rgcn_layers[0](
            x, data.edge_index, edge_types
        )
        query_emb = torch.concat([x[batch.target_edge_index[0]],x[batch.target_edge_index[1]]], axis=-1)

        node_avg = x.mean(axis=0)
        # concat the query
        edges = query_emb  # B x (dim + dim x num_q)
        return self.classfier(edges)

    def loss(self, labels, logits):
        ce = self.lossf(logits, labels)
        return ce

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        targets = batch.target_edge_type
        y_hat = self(batch)
        targets = targets.view(y_hat.size(0), -1)
        # calculate loss
        loss_val = self.loss(targets, y_hat)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        targets = batch.target_edge_type
        y_hat = self(batch)
        targets = targets.view(y_hat.size(0), -1)

        loss_val = self.loss(targets, y_hat)

        # acc
        val_acc = get_acc_multihot(y_hat, targets, threshold=0.5)
        val_acc = torch.tensor(val_acc)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc,})
        print(val_acc)
        # can also return just a scalar instead of a dict (return loss_val)
        return output

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]


    def test_step(self, batch, batch_idx):
        """
        Lightning calls this during testing, similar to `validation_step`,
        with the data from the test dataloader passed in as `batch`.
        """
        output = self.validation_step(batch, batch_idx)
        # Rename output keys
        output["test_loss"] = output.pop("val_loss")
        output["test_acc"] = output.pop("val_acc")

        return output

from train import load_files, cl_args
import pytorch_lightning as pl
import numpy as np
import re
import json
from src.train import get_data_loaders, get_dataset_test, get_acc
import json


def train(data_train_path, dataset_type, seed=0, epochs=10):

    train_loader, val_loader, unique_edge_labels, unique_query_labels = get_data_loaders(fname=data_train_path, fp_bp=False, dataset_type='clutrr' if 'clutrr' in dataset_type else dataset_type)
    optimizer_args = {'lr':0.001}
    num_training_steps = epochs *len(train_loader)
    scheduler_args = {'num_warmup_steps':cl_args.num_warmup_steps,'num_training_steps':num_training_steps}
    cl_args.optimizer_args = optimizer_args
    cl_args.scheduler_args = scheduler_args
    cl_args.num_relations = len(unique_edge_labels)
    cl_args.convtype = FastRGCNConv #RGCNConv

    print('num of unique relations: ', len(unique_edge_labels))
    print('------------------------------------------------------')
    
    trainer = pl.Trainer(
            # gpus=1,
            max_epochs=epochs,
            # gradient_clip_val=cl_args.max_grad_norm,
            # progress_bar_refresh_rate=1,
            # precision=cl_args.precision
        )

    model = SupervisedRGCN(cl_args)
    trainer.fit(model,train_loader,val_loader)

    with torch.no_grad():
        model.eval()
        model.to(device)
        if dataset_type == 'ambiguity':
            import time
            start = time.time()
            test_out = {'test_d':{}, 'test_w':{}, 'test_bl':{}, 'test_opec':{}}
            for data_filename in ['test_d', 'test_w', 'test_bl', 'test_opec']:
                fname = f'../../data/ambig/test_ambig_{data_filename}.csv'
                dataset_test = get_dataset_test(f"{fname}", 
                                                unique_edge_labels, unique_query_labels, 
                                                remove_not_chains=False,
                                                fp_bp=False, dataset_type=dataset_type, batch_size=None)
                dataset_test_loader =  DataLoader(dataset_test, batch_size=128, shuffle=True)

                f1_metric_macro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='weighted').to(device)
                f1_metric_micro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='micro').to(device)

                with torch.no_grad():
                    num_data_points = 0.
                    corrects = 0.
                    for batch in dataset_test_loader:
                        batch = batch.to(device)
                        pred_bh = model(batch)
                        acc = get_acc_multihot(pred_bh, batch.target_edge_type.reshape(pred_bh.shape[0], -1), threshold=0.5)
                        num_data_points += batch.target_edge_type.shape[0]
                        corrects += acc*batch.target_edge_type.shape[0]

                        preds = (pred_bh.sigmoid() >= 0.5).to(pred_bh.dtype)
                        f1_macro = f1_metric_macro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                        f1_micro = f1_metric_micro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                    
                    test_acc = corrects/num_data_points


                f1_macro_full = f1_metric_macro.compute()
                f1_micro_full = f1_metric_micro.compute()
                test_out[data_filename]['test_acc'] = test_acc.item()
                test_out[data_filename]['f1_macro'] = f1_macro_full.item()
                test_out[data_filename]['f1_micro'] = f1_micro_full.item()
                print(f'{data_filename}: {test_acc}')
                print(f'{data_filename} f1_macro: {f1_macro_full}')
                print(f'{data_filename} f1_micro: {f1_micro_full}')
            print(test_out)
            end = time.time()
        elif dataset_type == 'no_ambiguity':
            import time
            start = time.time()
            test_out = {'test_d':{}, 'test_w':{}, 'test_bl':{}, 'test_opec':{}}
            for data_filename in ['test_d', 'test_bl', 'test_opec']:
                fname = f'../../data/ambig/test_no_ambig_{data_filename}.csv'
                dataset_test = get_dataset_test(f"{fname}", 
                                                unique_edge_labels, unique_query_labels, 
                                                remove_not_chains=False,
                                                fp_bp=False, dataset_type=dataset_type, batch_size=None)
                dataset_test_loader =  DataLoader(dataset_test, batch_size=128, shuffle=True)
                f1_metric_macro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='weighted').to(device)
                f1_metric_micro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='micro').to(device)

                with torch.no_grad():
                    num_data_points = 0.
                    corrects = 0.
                    for batch in dataset_test_loader:
                        batch = batch.to(device)
                        pred_bh = model(batch)
                        acc = get_acc_multihot(pred_bh, batch.target_edge_type.reshape(pred_bh.shape[0], -1), threshold=0.5)
                        num_data_points += batch.target_edge_type.shape[0]
                        corrects += acc*batch.target_edge_type.shape[0]

                    preds = (pred_bh.sigmoid() >= 0.5).to(pred_bh.dtype)
                    f1_macro = f1_metric_macro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                    f1_micro = f1_metric_micro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                    
                    test_acc = corrects/num_data_points

                f1_macro_full = f1_metric_macro.compute()
                f1_micro_full = f1_metric_micro.compute()
                print(f1_macro_full, f1_micro_full)
                test_out[data_filename]['test_acc'] = test_acc.item()
                test_out[data_filename]['f1_macro'] = f1_macro_full.item()
                test_out[data_filename]['f1_micro'] = f1_micro_full.item()
                print(f'{data_filename}: {test_acc}')
                print(f'{data_filename} f1_macro: {f1_macro_full}')
                print(f'{data_filename} f1_micro: {f1_micro_full}')
        print(test_out)
        
        print('-----------------------------------------')
        print('test accs', test_out)
    # save
    exp_dir = f"../../results/paper_rgcn_{dataset_type}_{seed}"
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        # save model.test_accs as a json file
    json.dump(test_out, open(os.path.join(exp_dir, 'results.json'), 'w'))

if __name__ == '__main__':
    from src.utils import set_seed
    import argparse

    os.makedirs('results', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--dataset_type',type=str, default='clutrr_2')
    # parser.add_argument('--lr',type=float, default=1e-3)
    # parser.add_argument('--epochs',type=int, default=50)
    args = parser.parse_args()

    if args.dataset_type in ['ambiguity', 'no_ambiguity']:
        seeds = [1,2,3]
        data_train_path = f'../../data/ambig/train_ambig.csv' if args.dataset_type == 'ambiguity' else f'../../data/ambig/train_no_ambig.csv' 
        dataset_type=args.dataset_type
        # input_dim=16
        # num_layers=6
    for seed in seeds:
        set_seed(seed)
        train(data_train_path, dataset_type, seed=seed, epochs=100)

