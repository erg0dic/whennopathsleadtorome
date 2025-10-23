import copy
import math
from regex import T
import torch
from torch import nn
from torch.nn import Parameter, ModuleList, LayerNorm, Dropout
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

from transformers import get_linear_schedule_with_warmup

# import pytorch_lightning as pl
import lightning.pytorch as pl

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

#from the original Sinha et al. CLUTRR code
def get_mlp(input_dim, output_dim, num_layers=2, dropout=0.0):
    network_list = []
    assert num_layers > 0
    if num_layers > 1:
        for _ in range(num_layers-1):
            network_list.append(nn.Linear(input_dim, input_dim))    
            network_list.append(nn.ReLU())
            network_list.append(nn.Dropout(dropout))
    network_list.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(
        *network_list
    )

class EdgeAttentionFlat(nn.Module):
    def __init__(self,d_model, num_heads, dropout,model_config):
        super(EdgeAttentionFlat, self).__init__()
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model,bias=False), 4)
        
        self.dropout = nn.Dropout(p=dropout)

        
    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes = query.size(1)
        
        
        k,v,q = [l(x) for l, x in zip(self.linears, (key, value, query))]
        k = k.view(num_batches,num_nodes,num_nodes,self.num_heads,self.d_k)
        v = v.view_as(k)
        q = q.view_as(k)

        scores_r = torch.einsum("bxyhd,bxzhd->bxyzh",q,k) / math.sqrt(self.d_k)
        scores_r = scores_r.masked_fill(mask.unsqueeze(4), -1e9)
        scores_l = torch.einsum("bxyhd,bzyhd->bxyzh",q,k) / math.sqrt(self.d_k)
        scores_l = scores_l.masked_fill(mask.unsqueeze(4), -1e9)
        scores = torch.cat((scores_r,scores_l),dim=3)
        
        att = F.softmax(scores,dim=3)
        att = self.dropout(att)
        att_r,att_l = torch.split(att,scores_r.size(3),dim=3)

        x_r = torch.einsum("bxyzh,bxzhd->bxyhd",att_r,v)
        x_l = torch.einsum("bxyzh,bzyhd->bxyhd",att_l,v)

        x = x_r+x_l
        x = torch.reshape(x,(num_batches,num_nodes,num_nodes,self.d_model))

        return self.linears[-1](x)


        

class EdgeAttention(nn.Module):
    def __init__(self,d_model, num_heads, dropout,model_config):
        super(EdgeAttention, self).__init__()
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.linears = _get_clones(nn.Linear(d_model, d_model,bias=False), 6)
        
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.lesion_scores = model_config.lesion_scores
        self.lesion_values = model_config.lesion_values
        
    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)
        num_nodes = query.size(1)
        
        left_k, right_k, left_v, right_v, query = [l(x) for l, x in zip(self.linears, (key, key, value, value, key))]
        left_k = left_k.view(num_batches,num_nodes,num_nodes,self.num_heads,self.d_k)
        right_k = right_k.view_as(left_k)
        left_v = left_v.view_as(left_k)
        right_v = right_v.view_as(left_k)
        query = query.view_as(left_k)

   
        if self.lesion_scores:
            query = right_k
            scores = torch.einsum("bxahd,bxyhd->bxayh",left_k,query) / math.sqrt(self.d_k)
        else:
            scores = torch.einsum("bxahd,bayhd->bxayh",left_k,right_k) / math.sqrt(self.d_k)

        scores = scores.masked_fill(mask.unsqueeze(4), -1e9)

        val = torch.einsum("bxahd,bayhd->bxayhd",left_v,right_v)

        att = F.softmax(scores,dim=2)
        att = self.dropout(att)
        if self.lesion_values:
            x = torch.einsum("bxayh,bxahd->bxyhd",att,left_v)
            x=x.contiguous()
            x = x.view(num_batches,num_nodes,num_nodes,self.d_model)
        else:
            x = torch.einsum("bxayh,bxayhd->bxyhd",att,val)
            x = x.view(num_batches,num_nodes,num_nodes,self.d_model)

        return self.linears[-1](x)


        
        
        

class EdgeTransformerLayer(nn.Module):

    def __init__(self,model_config ,activation="relu"):
        super().__init__()

        self.num_heads = model_config.num_heads

        dropout = model_config.dropout
        

        d_model = model_config.dim
        d_ff = model_config.ff_factor * d_model


        self.flat_attention = model_config.flat_attention

        if self.flat_attention:
            self.edge_attention = EdgeAttentionFlat(d_model,self.num_heads, dropout,model_config)
        else:
            self.edge_attention = EdgeAttention(d_model,self.num_heads, dropout,model_config)


        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)




    def forward(self, batched_graphs, mask=None):

        batched_graphs = self.norm1(batched_graphs)
        batched_graphs2 = self.edge_attention(batched_graphs,batched_graphs,batched_graphs,mask=mask)
        batched_graphs = batched_graphs + self.dropout1(batched_graphs2)
        batched_graphs = self.norm2(batched_graphs)
        batched_graphs2 = self.linear2(self.dropout2(self.activation(self.linear1(batched_graphs))))
        batched_graphs = batched_graphs + self.dropout3(batched_graphs2)
            

        return batched_graphs


class EdgeTransformerEncoder(nn.Module):

    def __init__(self,model_config, shared_embeddings=None,activation="relu"):
        super().__init__()
        self.model_config = model_config

        self.num_heads = self.model_config.num_heads
        
        self.embedding = torch.nn.Embedding(num_embeddings=self.model_config.edge_types+1,
                                            embedding_dim=self.model_config.dim)
        
        self.deep_residual = False

        self.share_layers = self.model_config.share_layers



        self.num_layers = self.model_config.num_message_rounds

        encoder_layer = EdgeTransformerLayer(self.model_config)
        self.layers = _get_clones(encoder_layer, self.num_layers)

        self._reset_parameters()


    def forward(self, batch):
        
        batched_graphs = batch['batched_graphs']
        if self.model_config.input_rep == 'multiedge':
            batched_graphs_bnnl = batched_graphs.permute(0,3,2,1)
            batched_graphs_bnnl = F.pad(batched_graphs_bnnl, (0,1), value=0)
            # get the embedding vectors using a matmul
            batched_graphs = torch.einsum("bxyl,lh->bxyh",batched_graphs_bnnl,self.embedding.weight)
            zero_locs = torch.where((batched_graphs == 0).all(axis=-1))
            # set 0s to the zeroth embedding
            batched_graphs[zero_locs] = self.embedding.weight[0]
        else:
            batched_graphs = self.embedding(batched_graphs) #B x N x N x node_dim
        
    
        mask = batch['masks']
        
        if mask is not None:
            new_mask = mask.unsqueeze(2)+mask.unsqueeze(1)
            new_mask = new_mask.unsqueeze(3)+mask.unsqueeze(1).unsqueeze(2)
            mask = new_mask


        all_activations = [batched_graphs]

        if not self.share_layers:
            for mod in self.layers:
                batched_graphs = mod(batched_graphs, mask=mask)
        else:
            for i in range(self.model_config.num_message_rounds):
                batched_graphs = self.layers[0](batched_graphs,mask=mask)


        return batched_graphs

    def _reset_parameters(self):

        # for n,p in self.named_parameters():
        #     if ("linear" in n and "weight" in n) or ("embedding" in n):
        #         torch.nn.init.orthogonal_(p)
        #     else:
        #         if p.dim()>1:
        #             nn.init.xavier_uniform_(p)
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torchmetrics

class EdgeTransformer(pl.LightningModule):
    def __init__(self,model_config):
        super().__init__()

        self.save_hyperparameters()
        self._create_model(model_config)
        self.test_accs = []
        self.macro_f1s = []
        self.micro_f1s = []
        self.f1_metric_macro = torchmetrics.F1Score(task='multilabel', num_labels=model_config.target_size, 
                                                average='weighted')
        self.f1_metric_macro_ = torchmetrics.F1Score(task='multilabel', num_labels=model_config.target_size, 
                                                average='none')
        self.f1_metric_micro = torchmetrics.F1Score(task='multilabel', num_labels=model_config.target_size, 
                                                average='micro')
        self.accuracy_metric = torchmetrics.ExactMatch(task='multilabel', num_labels=model_config.target_size,
                                                )

    def _create_model(self,model_config):
        self.model_config = model_config

        self.encoder = EdgeTransformerEncoder(model_config)
        input_dim = model_config.dim
        self.decoder2vocab = get_mlp(
            input_dim,
            model_config.target_size
        )

        if model_config.dataset_type in ['ambiguity', 'no_ambiguity']:
            self.crit = nn.BCEWithLogitsLoss(reduction='mean')   # sigmoid + BCE in one call
        else:
            self.crit = nn.CrossEntropyLoss(reduction='mean')


    def configure_optimizers(self):
        # We will support Adam or AdamW as optimizers.
        if self.model_config.optimizer=="AdamW":
            opt = AdamW
        elif self.model_config.optimizer=="Adam":
            opt = Adam
        optimizer = opt(self.parameters(), **self.model_config.optimizer_args)
        

        if self.model_config.scheduler=='linear_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,**self.model_config.scheduler_args)
        
        

        return {'optimizer':optimizer, 'lr_scheduler':{'scheduler':scheduler,'interval':'step'}}

        
        #return {'optimizer':optimizer}


    def _calculate_loss(self, batch):
        batched_graphs = self.encoder(batch)

        query_edges = batch['target_edge_index']

        logits = self.decoder2vocab(batched_graphs[query_edges[:,0],query_edges[:,1],query_edges[:,2]])


        loss = self.crit(logits,batch['target_edge_type'])

        return loss, logits

    def training_step(self,batch,batch_idx):
        loss, _ = self._calculate_loss(batch)
        scheduler = self.lr_schedulers()

        return loss

    def compute_acc_scalar(self,batch,scores):
        preds = scores.max(-1)[1]

        labels = batch['target_edge_type']

        acc = ((torch.eq(preds,labels).sum(0))/preds.size(0)).detach()

        return acc
    def compute_acc(self, batch, scores, *, threshold=0.5, exact_match=True):
        """
        Parameters
        ----------
        scores   : Tensor (B, C)   raw logits for each class
        labels   : Tensor (B, C)   multi‑hot ground truth in batch dict
        threshold: float           sigmoid cutoff for positive prediction
        exact_match : bool
            • False (default) → micro‑accuracy across all bits
            • True            → sample‑level exact‑match accuracy
        """
        labels = batch["target_edge_type"].to(scores.dtype)  # (B, C) 0/1

        # 1) convert logits → 0/1 predictions
        preds = (scores.sigmoid() >= threshold).to(scores.dtype)

        if exact_match:
            # every bit must match in the row to be counted correct
            acc = (preds == labels).all(dim=1).float().mean().detach()
            assert np.allclose(acc.cpu(), accuracy_score(labels.cpu(), preds.cpu()))
        else:
            # micro‑accuracy: fraction of correctly predicted bits overall
            acc = (preds == labels).float().mean().detach()
        macro_f1 = f1_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        micro_f1 = f1_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)

        return acc, macro_f1, micro_f1

    def validation_step(self,batch,batch_idx):
        loss, logits = self._calculate_loss(batch)

        if self.model_config.dataset_type not in ['ambiguity', 'no_ambiguity']:
            acc = self.compute_acc_scalar(batch,logits)
        else:
            acc, macro_f1, micro_f1 = self.compute_acc(batch,logits)

        self.log("val_loss",loss,prog_bar=True)
        self.log("val_acc",acc,prog_bar=True)

    def test_step(self,batch,batch_idx):
        loss, logits = self._calculate_loss(batch)

        if self.model_config.dataset_type not in ['ambiguity', 'no_ambiguity']:
            acc = self.compute_acc_scalar(batch,logits)
        else:
             acc, macro_f1, micro_f1 = self.compute_acc(batch,logits)
        self.log("test_loss",loss,prog_bar=True)
        self.log("test_acc",acc,prog_bar=True)
        
        # self.test_accs[-1].append(acc.item())
        # self.macro_f1s[-1].append(macro_f1)
        # self.micro_f1s[-1].append(micro_f1)

        preds = (logits.sigmoid() >= 0.5).to(logits.dtype)
        targets = batch["target_edge_type"].to(logits.dtype)

        self.f1_metric_macro(preds, targets)
        self.f1_metric_micro(preds, targets)
        self.accuracy_metric(preds, targets)
        self.f1_metric_macro_(preds, targets)

        # print(f1_score(targets.cpu(), preds.cpu(), average='macro', zero_division=np.nan))

        # print('macro_f1', macro_f1)

    def on_test_epoch_end(self):
        f1_val_macro = self.f1_metric_macro.compute()
        f1_val_micro = self.f1_metric_micro.compute()
        acc = self.accuracy_metric.compute()
        self.log('macro f1', f1_val_macro)
        self.log('micro f1', f1_val_micro)
        self.log('em acc', acc)
        self.macro_f1s[-1].append(f1_val_macro.item())
        self.micro_f1s[-1].append(f1_val_micro.item())
        self.test_accs[-1].append(acc.item())
        self.f1_metric_macro.reset()
        self.f1_metric_micro.reset()
        self.accuracy_metric.reset()


