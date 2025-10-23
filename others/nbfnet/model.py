from email import message
from psutil import sensors_temperatures
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from collections.abc import Sequence
from src.train import get_data_loaders, ClutrrDataset, device, get_dataset_test
from src.model_nbf_fb import kl_div, xent, get_negative_relations
from torch_geometric.loader import DataLoader
import re
import os
import json
import torchmetrics


def margin_loss(score_negative_b, score_positive_b, margin=0.1):
    # sum over the facet dim and the batch dim
    margin_1 = torch.clamp_min(score_positive_b - score_negative_b + margin, 0)
    return margin_1.mean()

def get_score(left_arg: torch.Tensor, 
              right_arg: torch.Tensor, 
              score_fn: str ='kl', 
              ):
    if score_fn == 'kl':
        score_bn = kl_div(left_arg, right_arg)
    elif score_fn == 'xent':
        score_bn = xent(left_arg, right_arg)
    else:
        raise ValueError(f"score_fn must be one of ['kl', 'xent'] but got {score_fn}")
    
    return score_bn

def get_negative_relations(relations,
                            target_mask,          # (B, R)  0/1 or bool
                            num_negative_samples=10,
                            replacement=True):

    # torch.multinomial draws row-wise
    neg_inds = torch.multinomial((~target_mask.bool()).float(),
                                 num_samples=num_negative_samples,
                                 replacement=replacement)        # (B, K) 

    # sanity check: no positives slipped in
    assert not target_mask.gather(1, neg_inds).any(), \
           "positive label appeared in negative sample!"

    # lookup
    negative_relations = relations[neg_inds]                    # (B, K, …)
    return negative_relations

def get_margin_loss_term(outs_bh, rs_rh, targets, num_negative_samples=10, margin=0.1, score_fn='kl', 
                         outs_as_left_arg=True, pooling_layer=None):
    targets = targets.reshape(outs_bh.shape[0], -1)
    neg_samples_bnh = get_negative_relations(rs_rh, targets, num_negative_samples=num_negative_samples)
    if outs_as_left_arg:
        left_neg, right_neg = outs_bh.unsqueeze(1), neg_samples_bnh
        right_pos = targets @ rs_rh
        right_pos /= right_pos.sum(axis=-1).unsqueeze(-1)
        left_pos = outs_bh
    else:
        right_neg, left_neg = outs_bh.unsqueeze(1), neg_samples_bnh
        left_pos = targets @ rs_rh
        left_pos /= left_pos.sum(axis=-1).unsqueeze(-1)
        right_pos = outs_bh

    neg_score_bn = get_score(left_neg, right_neg, score_fn=score_fn)
    neg_score_b = neg_score_bn.mean(axis=-1)

    pos_score_bf = get_score(left_pos, right_pos, score_fn=score_fn)
    pos_score_b = pos_score_bf

    margin_1 = margin_loss(neg_score_b, pos_score_b, margin=margin)

    return margin_1

def out_score(outs_bh, rs_rh, score_fn='kl', outs_as_left_arg=True, pooling_layer=None):
    num_relations = rs_rh.shape[0]
    batch_size = outs_bh.shape[0]
    rs_brh = rs_rh.unsqueeze(0).repeat(batch_size, 1, 1)
    outs_brh = outs_bh.unsqueeze(1).repeat(1, num_relations, 1)
    if outs_as_left_arg:
        score_br = get_score(outs_brh, rs_brh, score_fn=score_fn)
    else:
        score_br = get_score(rs_brh, outs_brh, score_fn=score_fn)
    return score_br

class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim, device=device)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim, device=device)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim, device=device)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim, device=device)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim, device=device)

    def forward(self, input, boundary, edge_index, edge_type, size, edge_weight=None):

            
        relation = self.relation.weight
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)


        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index, edge_type=edge_type, size=size, edge_weight=edge_weight)
        return output

    def message(self, input_j, relation, boundary, edge_type):
        relation_j = relation.index_select(0, edge_type)
        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=0)  # (num_edges + num_nodes, batch_size, input_dim)
        
        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition

        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        shape = [1] * input.ndim
        shape[0] = -1
        edge_weight = edge_weight.view(shape)
        # print("edge weight is", edge_weight)
        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=0, dim_size=dim_size, reduce="mean")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=0, dim_size=dim_size, reduce="mean")
            max = scatter(input * edge_weight, index, dim=0, dim_size=dim_size, reduce="max")
            min = scatter(input * edge_weight, index, dim=0, dim_size=dim_size, reduce="min")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=0, dim_size=dim_size,
                             reduce=self.aggregate_func)
            
        if output.shape[0] == 1:
            output = output.squeeze(0)
        return output

    def update(self, update, input):
        # node update as a function of old states (input) [THIS IS THE INPUT TO `forward`] and this layer output (update)
        
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class NBFNet_og(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="pna",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, ):
        super(NBFNet_og, self).__init__()
        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]

        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.short_cut = short_cut  # whether to use residual connections between GNN layers
        self.concat_hidden = concat_hidden  # whether to compute final states as a function of all layer outputs or last

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, dependent))

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1])

        # additional relation embedding which serves as an initial 'query' for the NBFNet_og forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(num_relation, input_dim, device=device)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim, device=device))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, num_relation, device=device))
        self.mlp = nn.Sequential(*mlp)

    def bellmanford(self, batch):
        batch_size = batch.target_edge_index.shape[-1]
        # get the source node indices
        r_index = batch.target_edge_index[0]
        # # initialize queries (relation types of the given triples)
        # query = self.query(r_index)
        # index = r_index.unsqueeze(-1).tile(query.shape[-1])
        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch.num_nodes, self.dims[0], device=device)
        boundary[r_index] = torch.ones(self.dims[0], device=device)

        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        # boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (batch.num_nodes, batch.num_nodes)
        edge_weight = torch.ones(batch.edge_index.shape[-1], device=device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for i in range(len(self.layers)):
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = self.layers[0](layer_input, boundary, batch.edge_index, batch.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden


        output = hiddens[-1]

        return output

    def forward(self, batch, train=True):

        # message passing and updated node representations
        
        output = self.bellmanford(batch)  # (num_nodes, batch_size, feature_dim）
        feature = output

        feature_bh = feature[batch.target_edge_index[1]] # (batch_size, num_negative + 1, feature_dim)

        score = self.mlp(feature_bh).squeeze(-1)

        return score
    
from src.train import get_acc, get_acc_multihot

def train(input_dim=16, num_layers=6, message_func='transe', aggregate_func='mean', 
          dataset_type=None,
          data_train_path=None, seed=None,
          epochs=20,
          exp_tag='',
          ):
    train_loader, val_loader, unique_edge_labels, unique_query_labels = get_data_loaders(fname=data_train_path, fp_bp=False, dataset_type='clutrr' if 'clutrr' in dataset_type else dataset_type)
    


    separator = ">" * 30
    line = "-" * 30
    model = NBFNet_og(input_dim=input_dim, 
                    hidden_dims =  [input_dim]*num_layers,#[32, 32, 32, 32, 32, 32],
                    message_func=message_func,
                    aggregate_func=aggregate_func,
                    short_cut=True,
                    layer_norm=True,
                    dependent=False,
                    num_relation=len(unique_query_labels)
                    )
    import numpy as np
    def get_param_count(params):
        param_count = 0
        for param in params:
            param_count += np.array(param.shape).prod()
        return param_count
    print('param count', get_param_count(model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    # lossf = nn.CrossEntropyLoss()
    lossf = nn.BCEWithLogitsLoss()

    if not load_old_model:
        best_result = float("-inf")
        best_epoch = -1
        model.train()
        for epoch in range(epochs):
            losses = []
            print(separator)
            print("Epoch %d begin" % epoch)
            for batch in train_loader:
                batch = batch.to(device)
                pred_br = model(batch, train=True)
                # relations_rh = torch.softmax(model.query.weight, axis=-1)
                # margin_loss = get_margin_loss_term(pred_br, relations_rh, batch.target_edge_type, 
                #                                    num_negative_samples=1, margin=1., score_fn='xent', outs_as_left_arg=False)
                # loss = margin_loss
                loss = lossf(pred_br, batch.target_edge_type.reshape(pred_br.shape[0], -1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                # print("binary cross entropy: %g" % loss)
            with torch.no_grad():
                num_data_points = 0.
                corrects = 0.
                for batch in val_loader:
                    batch = batch.to(device)
                    pred_bh = model(batch, train=False)
                    # rs_rh = torch.softmax(model.query.weight, axis=-1)
                    # pred = -1*out_score(pred_bh, rs_rh, score_fn='xent', outs_as_left_arg=False)
                    
                    target_edge_type = batch.target_edge_type.reshape(-1, pred_bh.shape[-1])
                    acc = get_acc_multihot(pred_bh, target_edge_type, threshold=0.5)
                    num_data_points += target_edge_type.shape[0]
                    corrects += acc*target_edge_type.shape[0]
            
            avg_loss = sum(losses) / len(losses)
            val_acc = corrects/num_data_points
            print("average binary cross entropy: %g" % avg_loss)
            print("val acc: %g" % val_acc)
            print(separator)
            print("Epoch %d end" % epoch)
            print(line)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(state, f"model_epoch_{best_epoch}_{dataset_type}")

    if load_old_model:

        state = torch.load(f"model_epoch_{best_epoch}_{dataset_type}.pth", map_location=device)
        model.load_state_dict(state["model"])

    
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

            with torch.no_grad():
                num_data_points = 0.
                corrects = 0.
                f1_metric_macro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='weighted').to(device)
                f1_metric_micro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='micro').to(device)
                for batch in dataset_test_loader:
                    batch = batch.to(device)
                    pred_bh = model(batch, train=False)
                    acc = get_acc_multihot(pred_bh, batch.target_edge_type.reshape(pred_bh.shape[0], -1), threshold=0.5)
                    num_data_points += batch.target_edge_type.shape[0]
                    corrects += acc*batch.target_edge_type.shape[0]

                    preds = (pred_bh.sigmoid() >= 0.5).to(pred_bh.dtype)
                    f1_macro = f1_metric_macro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                    f1_micro = f1_metric_micro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                
                avg_loss = sum(losses) / len(losses)
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

            with torch.no_grad():
                num_data_points = 0.
                corrects = 0.
                f1_metric_macro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='weighted').to(device)
                f1_metric_micro = torchmetrics.F1Score(task='multilabel', num_labels=len(unique_query_labels), 
                                                average='micro').to(device)
                for batch in dataset_test_loader:
                    batch = batch.to(device)
                    pred_bh = model(batch, train=False)
                    acc = get_acc_multihot(pred_bh, batch.target_edge_type.reshape(pred_bh.shape[0], -1), threshold=0.5)
                    num_data_points += target_edge_type.shape[0]
                    corrects += acc*target_edge_type.shape[0]
                    
                    preds = (pred_bh.sigmoid() >= 0.5).to(pred_bh.dtype)
                    f1_macro = f1_metric_macro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                    f1_micro = f1_metric_micro(preds, batch.target_edge_type.reshape(pred_bh.shape[0], -1))
                    
                    num_data_points += batch.target_edge_type.shape[0]
                    corrects += acc*batch.target_edge_type.shape[0]
                
                avg_loss = sum(losses) / len(losses)
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
        end = time.time()
        print('inference time taken:', end-start)
    json.dump(test_out, open(f'results/paper_nbfnet_{dataset_type}_{seed}{exp_tag}.json', 'w'))

if __name__ == '__main__':
    from src.utils import set_seed
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--dataset_type',type=str, default='clutrr_3')
    parser.add_argument('--exp_tag',type=str, default='')
    # parser.add_argument('--lr',type=float, default=1e-3)
    # parser.add_argument('--epochs',type=int, default=50)
    args = parser.parse_args()


  
    if args.dataset_type in ['ambiguity', 'no_ambiguity']:
        seeds = [1]
        data_train_path = f'../../data/ambig/train_ambig.csv' if args.dataset_type == 'ambiguity' else f'../../data/ambig/train_no_ambig.csv' 
        dataset_type=args.dataset_type
        input_dim=256
        num_layers=10
    epochs = 100
    load_old_model=False
    # for input_dim in [16, 32, 64, 128]:
    #     for num_layers in [6, 7, 8, 9, 10,11]:
    #         for message_func in ['distmult', 'transe', 'rotate']:
    #             for aggregate_func in ['pna', 'mean', 'min', 'mul']:
    #                 print('current combo d, l, mf, af: ', input_dim, num_layers, message_func, aggregate_func)
    for seed in seeds:
        set_seed(seed)
        train(input_dim=input_dim, num_layers=num_layers, message_func='transe', aggregate_func='min', 
                dataset_type=dataset_type,
                data_train_path=data_train_path, seed=seed,
                epochs=epochs, 
                exp_tag=args.exp_tag
                )

