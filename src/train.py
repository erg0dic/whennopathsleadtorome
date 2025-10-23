"""
Cluttr dataset preprocessing code is adapted from https://github.com/bergen/EdgeTransformer
"""
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from torch.optim import AdamW
from torch import nn
from src.model_nbf_general import entropy, NBF 
from src.model_nbf_allprobs import NBFAllProbs, NBFCluttr, NBFCluttrAllprobs
from src.model_nbf_fb import NBFdistR, NBFdistRModule, get_margin_loss_term, margin_loss
from typing import Union, List, Callable
import os
import numpy as np
import pickle
import time
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import re
import wandb
import torchmetrics
from src.utils import (
    edge_labels_to_indices,
    load_jsonl,
    load_rcc8_file_as_dict,
	query_labels_to_indices, 
	mkdirs, get_most_recent_model_str, 
	remove_not_best_models,
	save_results_models_config,
	get_temp_schedule,
	get_acc,
	get_acc_multihot,
	read_datafile,
	save_model, 
	load_model_state,
	log, 
	save_array, 
	compute_sim,
	find_unique_edge_labels,
	preprocess_graphlog_dataset,
	get_lr,
	load_rcc8_file_as_dict,
	set_seed,
	process_batch, 
	batch_multihot
)
from sklearn.metrics import confusion_matrix  
# # torch.set_deterministic_debug_mode(True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# torch.set_float32_matmul_precision('medium')

def load_hydra_conf_as_standalone(config_name='config'):
	"NOTE: path needs to be relative to the configs directory."
	# https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
	config_path = "../configs"
	with initialize(version_base=None, config_path=config_path):
		cfg = compose(config_name=config_name)
	return cfg

def get_NBF_type(config_str: str) -> Union[Callable, NBF]:
	if config_str == 'NBFAllProbs':
		return NBFAllProbs
	elif config_str == 'NBF':
		return NBF
	elif config_str == 'NBFdistR':
		return NBFdistR
	else:
		raise NotImplementedError(f"Model {config_str} not implemented.")

def get_ys_reg_term(
		model_layer: Union[NBFCluttr, NBFCluttrAllprobs, NBFdistR], 
		dist: str = 'cosine'):
	if type(model_layer) is NBFCluttr:
		rel_proto_basis = model_layer.multi_embedding # (protos, hidden)
		proto_basis = model_layer.proto_embedding; str_shape = "ph"
		sims = compute_sim(rel_proto_basis, proto_basis, 
						type=dist, einsum_str=f"nafh, {str_shape} -> nap") # (num, num)
		proto_probas = torch.softmax(sims, dim=-1)
	elif type(model_layer) is NBFCluttrAllprobs:
		proto_basis = model_layer.multi_embedding; str_shape = "npfh"
		proto_probas = torch.softmax(proto_basis, dim=-1)
		# next steps are invariant to facet index: so just unravel
		proto_probas = proto_probas.reshape(*proto_probas.shape[:-2], -1)
	elif type(model_layer) is NBFdistRModule:
		proto_basis = model_layer.r_embedding; str_shape = 'nfh'
		proto_probas = torch.softmax(proto_basis, dim=-1)
	else:
		raise AssertionError("Model type not recognized.")
	entropies = entropy(proto_probas, axis=-1)
	return entropies.sum()


class ClutrrDataset(Dataset):
	def __init__(self, dataset, reverse=False, fp_bp=False, 
			     unique_edge_labels=None,unique_query_labels=None):
		super().__init__()
		self.fp_bp = fp_bp
		self.edges = dataset['edges']
		query_labels = []
		for label_list in dataset['query_label']:
			query_labels.extend(label_list)

		#  unify unique edge and query labels
		if unique_edge_labels is None:
			unique_edge_labels = set(find_unique_edge_labels(dataset['edge_labels'])) 
			unique_query_labels = set(query_labels)
			unique_labels = list(unique_edge_labels.union(unique_query_labels))
		else:
			unique_labels = unique_edge_labels
			assert unique_edge_labels == unique_query_labels
		# assert len(unique_labels) == 20

		self.edge_labels, unique_edge_labels = edge_labels_to_indices(dataset['edge_labels'],unique_labels)
		self.query_edge = dataset['query_edge']
		self.query_label, unique_query_labels = query_labels_to_indices(dataset['query_label'],unique_labels)
		
		self.unique_edge_labels = unique_edge_labels
		self.unique_query_labels = unique_query_labels
		self.num_edge_labels = len(unique_edge_labels)
		self.num_query_labels = len(unique_query_labels)

		self.query_label = batch_multihot(self.query_label, self.num_edge_labels)

		# consider edge reversal cases
		if reverse:
			self.edges, self.edge_labels = self.get_reversed_edges()
		# for simultaneous forward and backward pass
		if self.fp_bp:
			self.rev_edges, self.rev_edge_labels = self.get_reversed_edges()

	def  get_reversed_edges(self):
		rev_edges = list(map(lambda x: reverse_edges(x), self.edges))
		rev_edge_labels = list(map(lambda x: x[::-1], self.edge_labels))
		return rev_edges, rev_edge_labels
	
	def __len__(self):
		return len(self.edges)

	def __getitem__(self,index):	
		item = {
			'edge_index': torch.LongTensor(self.edges[index]).permute(1,0),
			'edge_type': torch.LongTensor(self.edge_labels[index]),
			'target_edge_index': torch.LongTensor(self.query_edge[index]).unsqueeze(1),
			'target_edge_type': self.query_label[index],
		}
		if self.fp_bp:
			item['rev_edge_index'] = torch.LongTensor(self.rev_edges[index]).permute(1,0)
			item['rev_edge_type'] = torch.LongTensor(self.rev_edge_labels[index])
		return item
	
def remove_last_k_path(fname, k=1):
	splitf = fname.split('/')
	if k == 1:
		head, tail = splitf[:-k], splitf[-1]
	else:
		head, tail = splitf[:-k], splitf[-k:]
	return '/'.join(head), tail
	
def make_geo_transform(dataset, fp_bp=False):
	if fp_bp:
		return [HeteroData(
						fw = {'x':torch.arange(c['edge_index'].max().item()+1).unsqueeze(1) },
						bw = {'x':torch.arange(c['rev_edge_index'].max().item()+1).unsqueeze(1) },
						fw__rel__fw={
								'edge_index':c['edge_index'], 
								'edge_type':c['edge_type'], 
								'target_edge_index':c['target_edge_index'], 
								'target_edge_type':c['target_edge_type'], 
								}, 
						bw__rel__bw={
								'edge_index':c['rev_edge_index'], 
								'edge_type':c['rev_edge_type'], 
								},
						)  for c in dataset
				]
		
	else:
		return [Data(edge_index=c['edge_index'], 
			   		edge_type=c['edge_type'], 
					target_edge_index=c['target_edge_index'], 
					target_edge_type=c['target_edge_type'], 
					x=torch.arange(c['edge_index'].max().item()+1).unsqueeze(1)
					) for c in dataset
				]

def get_pickle_filename(fname, remove_not_chains=False, add_prefix=True, k=1):
	if ".." == fname[:2]:
		fname = fname[3:]
	path, file = remove_last_k_path(fname, k=k)
	assert isinstance(file, str), "File should be a string."
	pickle_path =path+"/pickles/"
	if add_prefix:
		pickle_path = '../'+pickle_path
	if not os.path.exists(pickle_path):
		os.mkdir(pickle_path)
	pfname = pickle_path + f"{file}_chains_{remove_not_chains}.pkl"
	return pfname

def reverse_edges(edge_list):
    sources = [x[0] for x in edge_list]
    sinks = [x[1] for x in edge_list]
    num_nodes = max(sources + sinks)
    reversed_nodes = list(range(num_nodes+1))[::-1]
    # map source to sink in edge_list
    new_edges = []
    for edge in edge_list:
        new_edge = reversed_nodes[edge[1]], reversed_nodes[edge[0]]
        new_edges.append(new_edge)
    return new_edges[::-1]

def get_data_loaders(fname: Union[List[str], str] = '../data/data_9b2173cf/1.2,1.3_train.csv', 
					batch_size=128, remove_not_chains=False, reverse=False, 
					fp_bp=False, dataset_type='clutrr'):
	# torch.manual_seed(42)

	# sanity test to make sure reverse isn't being used with fp_bp
	assert not(reverse and fp_bp), "reverse is incompatible with fp_bp. But got both True"
	if dataset_type != 'graphlog':
		fname_arg = fname 
	else: 
		fname_arg = fname[0]
	
	pfname = get_pickle_filename(fname_arg, remove_not_chains=remove_not_chains, k=1)
	if fp_bp:
		# pickle tag mod
		pfname = pfname.replace(".pkl", "_fp_bp.pkl")
	
	if not os.path.exists(pfname):
		# TODO preprocess graphlog file here
		
		if dataset_type in [ 'ambiguity', 'no_ambiguity']:
			data = load_rcc8_file_as_dict(fname)
			cdata = ClutrrDataset(data, reverse, fp_bp, None, None)
			pickle.dump(cdata, open(pfname, 'wb'))
			log.info(f"saving preprocessed data file at: {pfname}")
		else:
			raise NotImplementedError
	else:
		log.info(f"preprocessed data file loaded from: {pfname}")
		cdata = pickle.load(open(pfname, 'rb'))
	return make_data_loaders(cdata, batch_size, train_ratio=0.8, fp_bp=fp_bp, dataset_type=dataset_type)

def get_dataset_test(fname: str = '../data/data_9b2173cf/1.3_test.csv',
                     unique_edge_labels=None, 
					 unique_query_labels=None, 
					 remove_not_chains=False, 
					 reverse=False, 
					 fp_bp=False, 
					 dataset_type='ambiguity', 
					 batch_size=128):
	pfname = get_pickle_filename(fname, remove_not_chains=remove_not_chains)
	if fp_bp:
		# pickle tag mod
		pfname = pfname.replace(".pkl", "_fp_bp.pkl")
	if not os.path.exists(pfname):
		if dataset_type == 'clutrr':
			data = read_datafile(fname, remove_not_chains=remove_not_chains)
			cdata = ClutrrDataset(data, reverse, fp_bp, unique_edge_labels, unique_query_labels)
		elif dataset_type == 'graphlog':
			data = preprocess_graphlog_dataset(fname)
			cdata = ClutrrDataset(data, reverse, fp_bp, unique_edge_labels, unique_query_labels)
			assert cdata.unique_edge_labels == unique_edge_labels
		elif dataset_type in [ 'ambiguity', 'no_ambiguity']:
			data = load_rcc8_file_as_dict(fname)
			cdata = ClutrrDataset(data, reverse, fp_bp, unique_edge_labels, unique_query_labels)
		pickle.dump(cdata, open(pfname, 'wb'))
	else:
		cdata = pickle.load(open(pfname, 'rb'))
	# both of the following operations should preserve row order
	# dataset_geodic = make_geo_transform(cdata, fp_bp=fp_bp) # deterministic function: just containerize `cdata`
	test_dataset = make_geo_transform(cdata, fp_bp=fp_bp)
	if dataset_type == 'clutrr':
		test_dataset = Batch.from_data_list(test_dataset) # another wrapper
	return test_dataset

def make_data_loaders(cdata: Union[List[ClutrrDataset], ClutrrDataset], 
					  batch_size, train_ratio: float = 0.8, fp_bp=False, dataset_type='clutrr'):
	assert 0. < train_ratio <= 1., "acceptable domain is (0, 1]"
	if dataset_type in [ 'ambiguity', 'no_ambiguity']:
		train_dataset, val_dataset = random_split(cdata, [train_ratio, 1-train_ratio])
		unique_edge_labels = cdata.unique_edge_labels
		unique_query_labels = cdata.unique_query_labels
	else:
		raise NotImplementedError

	train_dataset, val_dataset = make_geo_transform(train_dataset,fp_bp=fp_bp), make_geo_transform(val_dataset, fp_bp=fp_bp)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
	val_loader = DataLoader(val_dataset, batch_size=batch_size if dataset_type=='graphlog' else len(val_dataset), 
						 shuffle=True, num_workers=8)

	return train_loader, val_loader, unique_edge_labels, unique_query_labels


def eval_model(model, test_dataset, fp_bp=False, use_margin_loss_multi=True, threshold=0.5, **kwargs):
	mkdirs('../results')
	model.eval()
	if use_margin_loss_multi:
		logits, _ = model(test_dataset, **kwargs)
	else:
		outs_brh, _ = model(test_dataset, **kwargs)
		logits = model.final_to_logit_mlp(outs_brh.reshape(outs_brh.shape[0], -1))
	if fp_bp:
		target_edge_type = test_dataset['fw', 'rel', 'fw'].target_edge_type
	else:
		target_edge_type = test_dataset.target_edge_type
	target_edge_type = target_edge_type.reshape(-1, logits.shape[-1])
	acc = get_acc_multihot(logits, target_edge_type, threshold=threshold)
	return acc.item(), logits, target_edge_type

def get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
						bw_only, final_linear, use_margin_loss, infer, 
						outs_as_left_arg, score_fn, num_labels, threshold=0.5, use_margin_loss_multi=True):
	
	num_data_points = 0.
	corrects = 0.
	f1_metric_macro = torchmetrics.F1Score(task='multilabel', num_labels=num_labels, 
									average='weighted').to(device)
	f1_metric_micro = torchmetrics.F1Score(task='multilabel', num_labels=num_labels, 
									average='micro').to(device)
	accuracy_metric = torchmetrics.ExactMatch(task='multilabel', num_labels=num_labels,
									).to(device)

	
	for batch in dataset_test_loader:
		acc, logits, target_edge_type = eval_model(model, batch.to(device), fp_bp=fp_bp, 
								fw_only=fw_only, bw_only=bw_only, 
								final_linear=final_linear, 
								use_margin_loss=use_margin_loss,
								infer=infer, outs_as_left_arg=outs_as_left_arg,
								score_fn=score_fn, threshold=threshold,
								use_margin_loss_multi=use_margin_loss_multi)
		if fp_bp:
			target_edge_type = batch['fw', 'rel', 'fw'].target_edge_type
		else:
			target_edge_type = batch.target_edge_type
		target_edge_type = target_edge_type.reshape(-1, batch['fw', 'rel', 'fw'].target_edge_index.shape[-1])
		target_edge_type = target_edge_type.reshape(-1, num_labels)
		preds = (logits.sigmoid() >= threshold).to(logits.dtype)
		f1_metric_macro_ = f1_metric_macro(preds, target_edge_type.reshape(-1, num_labels))
		f1_metric_micro_ = f1_metric_micro(preds, target_edge_type.reshape(-1, num_labels))
		accuracy_metric_ = accuracy_metric(preds, target_edge_type.reshape(-1, num_labels))
		# print(accuracy_metric_)

		num_data_points += target_edge_type.shape[0]
		corrects += acc*target_edge_type.shape[0]

	test_acc = corrects/num_data_points
	# test_acc = accuracy_metric.compute()
	f1_macro_full = f1_metric_macro.compute()
	f1_micro_full = f1_metric_micro.compute()
	return test_acc, f1_macro_full, f1_micro_full

def get_test_metrics(data_train_path, 
					 unique_edge_labels,
					 unique_query_labels,
					 remove_not_chains,
					 bw_only,
					 model, final_linear,
					 fp_bp, infer, fw_only,
					 use_margin_loss, 
					 outs_as_left_arg,
					 score_fn, 
					 dataset_type,
					 batch_size=None,
					 threshold=0.5,
					 use_margin_loss_multi=True):
	test_out = []
	# test
	import time
	start = time.time()
			
	test_out = {'test_d':{}, 'test_w':{}, 'test_bl':{}, 'test_opec':{}, 'test_accs': {}, 'macro_f1s':{}, 'micro_f1s':{}}

	if dataset_type == 'ambiguity':
		prefix = 'ambig'
		ratios =  [1.14, 1.17, 1.0, 1.4, 1.29, 1.43, 1.5, 1.38, 1.33, 1.12, 1.25, 0.88]
	else:
		prefix = 'no_ambig'
		ratios = [1.17,1.33,1.4,1.0,1.14,1.5,1.29,1.43,1.38,1.25,1.22,1.12,1.44,1.11,0.88,1.1]


	# chain lens
	for k in [7,8,9,10,11,12,13]:
		fname = f'../data/ambig/test_{prefix}_long_k_{k}.csv'
		test_filename = f"('long_k', {k})"
		dataset_test = get_dataset_test(f"{fname}", 
									unique_edge_labels, unique_query_labels, 
									remove_not_chains=remove_not_chains,
									fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
		dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
			
		test_acc, f1_macro, f1_micro = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
						bw_only, final_linear, use_margin_loss, infer, 
						outs_as_left_arg, score_fn, len(unique_query_labels), threshold, use_margin_loss_multi)
		test_out['test_accs'][test_filename] = test_acc
		test_out['macro_f1s'][test_filename] = f1_macro.item()
		test_out['micro_f1s'][test_filename] = f1_micro.item()
# mrnr
	for mrnr in ratios:
		fname = f'../data/ambig/test_{prefix}_long_mrnr_{mrnr}.csv'
		test_filename = f"('long_mrnr', {mrnr})"
		dataset_test = get_dataset_test(f"{fname}", 
									unique_edge_labels, unique_query_labels, 
									remove_not_chains=remove_not_chains,
									fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
		dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
			
		test_acc, f1_macro, f1_micro = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
						bw_only, final_linear, use_margin_loss, infer, 
						outs_as_left_arg, score_fn, len(unique_query_labels), threshold, use_margin_loss_multi)
			
		test_out['test_accs'][test_filename] = test_acc
		test_out['macro_f1s'][test_filename] = f1_macro.item()
		test_out['micro_f1s'][test_filename] = f1_micro.item()
	
	# mrnr
	
	for OPEC in [3,4]:
		fname = f'../data/ambig/test_{prefix}_OPEC_{OPEC}.csv'
		test_filename = f"('OPEC', {OPEC})"
		dataset_test = get_dataset_test(f"{fname}", 
									unique_edge_labels, unique_query_labels, 
									remove_not_chains=remove_not_chains,
									fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
		dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
			
		test_acc, f1_macro, f1_micro = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
						bw_only, final_linear, use_margin_loss, infer, 
						outs_as_left_arg, score_fn, len(unique_query_labels), threshold, use_margin_loss_multi)
			
		test_out['test_accs'][test_filename] = test_acc
		test_out['macro_f1s'][test_filename] = f1_macro.item()
		test_out['micro_f1s'][test_filename] = f1_micro.item()
	
	# mrnr
	if dataset_type == 'ambiguity':
		with torch.no_grad():
			for data_filename in ['test_d', 'test_w', 'test_bl', 'test_opec']:
				fname = f'../data/ambig/test_ambig_{data_filename}.csv'
				dataset_test = get_dataset_test(f"{fname}", 
											unique_edge_labels, unique_query_labels, 
											remove_not_chains=remove_not_chains,
											fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
				dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
					
				test_acc, f1_macro, f1_micro = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
								bw_only, final_linear, use_margin_loss, infer, 
								outs_as_left_arg, score_fn, len(unique_query_labels), threshold, use_margin_loss_multi)
					
				test_out[data_filename]['test_acc'] = test_acc
				test_out[data_filename]['f1_macro'] = f1_macro.item()
				test_out[data_filename]['f1_micro'] = f1_micro.item()
				print(f'{data_filename}: test_acc: {test_acc}')
				print(f'{data_filename}: f1_macro: {f1_macro.item()}')
				print(f'{data_filename}: f1_micro: {f1_micro.item()}')

			# brl and k
			for k in [4,5,6]:
				for brl in [1,2,3,4,5]:
					fname = f'../data/ambig/test_{prefix}_short_b_{brl}_k_{k}.0.csv'
					test_filename = f"('s3_short_k_branches_a', ({k}, {brl})"

					dataset_test = get_dataset_test(f"{fname}", 
												unique_edge_labels, unique_query_labels, 
												remove_not_chains=remove_not_chains,
												fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
					dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
						
					test_acc, f1_macro, f1_micro = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
									bw_only, final_linear, use_margin_loss, infer, 
									outs_as_left_arg, score_fn, len(unique_query_labels), threshold, use_margin_loss_multi)
						
					test_out['test_accs'][test_filename] = test_acc
					test_out['macro_f1s'][test_filename] = f1_macro.item()
					test_out['micro_f1s'][test_filename] = f1_micro.item()
			for brl in [6,7,8,9,10,11,12]:
				fname = f'../data/ambig/test_{prefix}_short2_b_{brl}.csv'
				test_filename = f"('s2_ood_short_k_branches_a', ({brl}))"

				dataset_test = get_dataset_test(f"{fname}", 
											unique_edge_labels, unique_query_labels, 
											remove_not_chains=remove_not_chains,
											fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
				dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
					
				test_acc, f1_macro, f1_micro = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
								bw_only, final_linear, use_margin_loss, infer, 
								outs_as_left_arg, score_fn, len(unique_query_labels), threshold, use_margin_loss_multi)
				test_out['test_accs'][test_filename] = test_acc
				test_out['macro_f1s'][test_filename] = f1_macro.item()
				test_out['micro_f1s'][test_filename] = f1_micro.item()


	elif dataset_type == 'no_ambiguity':
		with torch.no_grad():
			for data_filename in ['test_d', 'test_bl', 'test_opec']:
				fname = f'../data/ambig/test_no_ambig_{data_filename}.csv'
				dataset_test = get_dataset_test(f"{fname}", 
											unique_edge_labels, unique_query_labels, 
											remove_not_chains=remove_not_chains,
											fp_bp=fp_bp, dataset_type=dataset_type, batch_size=None)
				dataset_test_loader =  DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
					
				test_acc, f1_macro, f1_micro = get_batched_test_out(dataset_test_loader, model, fp_bp, fw_only, 
								bw_only, final_linear, use_margin_loss, infer, 
								outs_as_left_arg, score_fn, len(unique_query_labels), threshold, use_margin_loss_multi)
					
				test_out[data_filename]['test_acc'] = test_acc
				test_out[data_filename]['f1_macro'] = f1_macro.item()
				test_out[data_filename]['f1_micro'] = f1_micro.item()
				print(f'{data_filename}: {test_acc}')
				print(f'{data_filename}: f1_macro: {f1_macro.item()}')
				print(f'{data_filename}: f1_micro: {f1_micro.item()}')

	else:
		raise NotImplementedError(f'test support for {dataset_type} has not been implemented.')
	end = time.time()
	print(f'inference time taken (seconds): {end-start}')
	return test_out


def toggle_prob_dist_softening(epoch: int, temp_schedule: list, model: NBFCluttr, config: DictConfig) -> None:
	# set gumbel temperature using schedule
	if epoch == len(temp_schedule)-1:
		if config.shared:
			model.temperature = temp_schedule[epoch]
			model.BF_layers[0].temperature = temp_schedule[epoch]
			if np.allclose(temp_schedule[epoch], 1.):
				model.eval_mode = True
				model.BF_layers[0].eval_mode = True
		else:
			raise NotImplementedError("Not implemented for non-shared model layers.")

def check_if_prev_configs_have_same_params(config, exp_name):
	"""
	Checks the results directory to see if an experiment with the same config values has already been conducted. 
	Will be automatically overidden after 10 seconds with a warning.
	Just a don't-shoot-yourself-in-the-foot check.
	"""
	raise NotImplementedError

def make_sweeper_dict(config) -> str:
	sweep_config = {
		"method": "random",
		"name": "sweep",
		"metric": {"goal": "maximize", "name": "val_acc"},
		"parameters": {
			"batch_size": {"values": [8, 16, 64, 128, 400, 512]},
			"epochs": {"values": [10]},
			"lr": {"distribution": "uniform", "max": 0.01, "min": 0.0001},
			"num_layers": {"values": [6, 8, 12]},
			"hidden_dim": {"values": [18, 36, 64]},

		},
	}
	sweep_id = wandb.sweep(sweep=sweep_config, project=config.wandb.project)

	return sweep_id

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(config: DictConfig) -> None:
	# resolve the value interpolations in the config file
	OmegaConf.resolve(config)

	# create the relevant directories
	# results_dir = '/scratch/c.c2079906/results_graphlog'
	results_dir = '../results'
	models_dir = '../models'
	mkdirs([results_dir, models_dir])

	# TAG: wandb 
	if config.do_hyper_sweep:
		run = wandb.init()
	
	if config.turn_on_wandb:
	
		try:
			run = wandb.init(project=config.wandb.project, name=config.experiments.exp_name, id=config.experiments.exp_name)
		except Exception as e:
			print(e)
	
	# load all the experimental config parameters from `config`
	NBF_type, epochs, exp_name, entropy_reg = get_NBF_type(config.experiments.NBF_type), config.experiments.epochs, config.experiments.exp_name, config.experiments.entropy_reg
	final_noiseless_epochs = config.experiments.get("final_noiseless_epochs", 15)
	remove_not_chains = config.experiments.remove_not_chains
	data_train_path = config.experiments.data_train_path
	hidden_dim = config.experiments.get('hidden_dim', config.hidden_dim)
	num_layers = config.num_layers
	eval_mode = config.experiments.get('eval_mode', False)
	just_discretize = config.experiments.get('just_discretize', False)
	ys_are_probas = config.experiments.get('is_y_prob_model', False)
	do_reg = config.experiments.get('do_reg', True)
	fp_bp = config.experiments.get('fp_bp', False)
	facets = config.experiments.get('facets', 1)
	fw_only = config.experiments.get('fw_only', False)
	bw_only = config.experiments.get('bw_only', False)
	reg_prefactor = 1. if do_reg else 0.
	use_margin_loss = config.experiments.get('use_margin_loss', False)
	num_negative_samples = config.experiments.get('num_negative_samples', 10)
	margin = config.experiments.get('margin', 0.1)
	score_fn = config.experiments.get('score_fn', 'xent')
	final_linear = config.experiments.get('final_linear', False)
	outs_as_left_arg = config.experiments.get('outs_as_left_arg', True)
	infer = config.experiments.get('infer', True)
	dataset_type = config.experiments.get('dataset_type', 'clutrr')
	temperature_param = config.experiments.get('temperature', 1.)
	batch_size = config.experiments.get('batch_size', 128)
	rel_offset = config.experiments.get('rel_offset', 0)
	aggr_type = config.experiments.get('aggr_type', 'mul') 
	seed = config.experiments.get('seed', 42)
	test_only = config.experiments.get('test_only', False)
	use_margin_loss_multi = config.experiments.get('use_margin_loss_multi', True)

	if use_margin_loss_multi:
		threshold=0.25
	else:
		threshold=0.5

	# ablations
	ablate_compose = config.experiments.get('ablate_compose', False)
	ablate_probas = config.experiments.get('ablate_probas', False)
	set_seed(seed)
	 


	# fixed stuff follows here:
	lr = config.experiments.get('lr', 0.001)
	if config.set_hidden_eq_num_relations:
		hidden_dim = len(unique_query_labels)
		log.warn(f"Hidden dim {hidden_dim} is the same as number of relations")

	# TAG: wandb
	# make some hyper sweep param changes
	if config.do_hyper_sweep:
		hidden_dim = wandb.config.hidden_dim
		num_layers = wandb.config.num_layers
		epochs = wandb.config.epochs
		lr = wandb.config.lr
		# TODO take out the gumbel noise in the hypersweep?

	train_loader, val_loader, unique_edge_labels, \
	unique_query_labels = get_data_loaders(remove_not_chains=remove_not_chains, 
										         fname=data_train_path, fp_bp=fp_bp, 
												 dataset_type=dataset_type, batch_size=batch_size)
	loss = nn.BCEWithLogitsLoss()
	model = NBF_type(hidden_dim=hidden_dim, num_relations=len(unique_query_labels)+rel_offset, 
			 shared=config.shared, use_mlp_classifier=config.use_mlp_classifier, 
			 dist=config.dist, fix_ys=config.fix_ys, num_layers=num_layers,  
			 residual=False, 
			 eval_mode=eval_mode,
			 just_discretize=just_discretize, 
			 ys_are_probas=ys_are_probas,
			 facets=facets,
			 temperature=temperature_param, 
			 aggr_type=aggr_type,
			 ablate_compose=ablate_compose,
			 ablate_probas=ablate_probas,
			 use_margin_loss_multi=use_margin_loss_multi,
			 )


	model.to(device)
	opt = AdamW(model.parameters(), lr=lr)
	# add a temperature decay schedule
	temp_schedule = get_temp_schedule(epochs, final_noiseless_epochs=final_noiseless_epochs, num_steps=4, 
								      schedule=config.experiments.schedule, min_temp=0.5)

	epoch=0
	if test_only:
		config.experiments.load_from_checkpoint = True
	if config.experiments.load_from_checkpoint:
		model_str = get_most_recent_model_str(exp_name)
		if model_str:
			load_model_state(model, model_str, opt)
			match = re.search("model_epoch_[0-9]+", model_str)
			if match:
				epoch = int(match.group().split("_")[-1])
				assert epoch < epochs, f"Epochs {epochs} should be greater than the loaded epoch {epoch}."
	else:
		log.info("Training afresh. WARNING: previous checkpoints will be overwritten.")
	best_acc = 0
	best_epoch = epoch
	# model.train()
	h_entropies, accs, y_entropies = [], [], []
	all_epoch_ys = []
	
	# TAG: scheduler
	# if dataset_type == 'graphlog':
	# 	from torch.optim.lr_scheduler import StepLR
	# 	scheduler = StepLR(opt, step_size=30, gamma=0.5)
	start = time.time()
	if test_only:
		epoch = epochs
	while epoch < epochs:
		epoch_train_losses = []
		# temperature based discretization is turned off when the temp schedule "warms up" to 1
		# toggle_prob_dist_softening(epoch, temp_schedule, model, config)
		# train
		for batch in train_loader:
			batch = batch.to(device)
			# processed = process_batch(batch['fw', 'rel', 'fw'])
			# breakpoint()
			if fp_bp:
				target_edge_type = batch['fw', 'rel', 'fw'].target_edge_type
			else:
				target_edge_type = batch.target_edge_type
			opt.zero_grad()
			if use_margin_loss:
				if use_margin_loss_multi:
					outs_bfh, rs_rfh = model(batch, fw_only=fw_only, bw_only=bw_only, use_margin_loss=True, final_linear=final_linear)
					
					loss_train = get_margin_loss_term(outs_bfh, rs_rfh, target_edge_type, 
													num_negative_samples=num_negative_samples, 
													margin=margin, score_fn=score_fn, 
													outs_as_left_arg=outs_as_left_arg,
													pooling_layer=None)
				else:
					outs_brh, rs_rfh = model(batch, fw_only=fw_only, bw_only=bw_only, use_margin_loss=True, final_linear=final_linear)
					target_edge_type = target_edge_type.reshape(outs_brh.shape[0], -1)
					outs_br = model.final_to_logit_mlp(outs_brh.reshape(outs_brh.shape[0], -1))
					loss_train = loss(outs_br, target_edge_type)				

			else:
				logits, proto_proba = model(batch, fw_only=fw_only, bw_only=bw_only, final_linear=final_linear)
				loss_train = loss(logits, target_edge_type)

			epoch_train_losses.append(loss_train.item())
			# TODO: Hardcoded entropy reg (Works better than reg. at epoch 0. NEED TO TUNE?)
			if entropy_reg > 0:
				if fp_bp:
					entropy_loss = torch.tensor(0., device=device)
					entropy_loss_ys = torch.tensor(0., device=device)
				else:
					final_node_emb_entropy = entropy(proto_proba, axis=0)
					entropy_loss = final_node_emb_entropy.sum()
					entropy_loss_ys = get_ys_reg_term(model.BF_layers[0])
				
				(loss_train + reg_prefactor*entropy_reg*(entropy_loss + 0.001*entropy_loss_ys)).backward()
			else:
				loss_train.backward()

			opt.step()
		# validate
		#  TODO: add early stopping?
		with torch.no_grad():
			num_data_points = 0.
			corrects = 0.
			loss_margin_tot = 0.
			for batch in val_loader:
				batch = batch.to(device)
				if use_margin_loss:
					assert infer, "Margin loss requires that inference be made using the score function."
					outs, _ = model(batch, fw_only=fw_only, bw_only=bw_only, 
					 				use_margin_loss=True, final_linear=final_linear,
									infer=True, outs_as_left_arg=outs_as_left_arg, score_fn=score_fn)
					
					##### DEBUGGING ##### 
					if use_margin_loss_multi:
						outs_bfh, rs_rfh = model(batch, fw_only=fw_only, bw_only=bw_only, 
												use_margin_loss=True, final_linear=final_linear)
						logits = outs
					else:
						outs_brh, rs_rfh = model(batch, fw_only=fw_only, bw_only=bw_only, 
							                 use_margin_loss=True, final_linear=final_linear)
						outs_br = model.final_to_logit_mlp(outs_brh.reshape(outs_brh.shape[0], -1))
						
						logits = outs_br
					if dataset_type != 'graphlog':
						print('outs', torch.softmax(outs[0], dim=-1))

				else:
					logits, probas = model(batch, fw_only=fw_only, bw_only=bw_only, final_linear=final_linear)
				if fp_bp:
					target_edge_type = batch['fw', 'rel', 'fw'].target_edge_type
				else:
					target_edge_type = batch.target_edge_type
				# print('target', target_edge_type)
				if use_margin_loss_multi:
					target_edge_type = target_edge_type.reshape(-1, logits.shape[-1])
					loss_val = loss(logits, target_edge_type)
					loss_margin_val = get_margin_loss_term(outs_bfh, rs_rfh, target_edge_type, 
													num_negative_samples=num_negative_samples, 
													margin=margin, score_fn=score_fn, 
													outs_as_left_arg=outs_as_left_arg,
													pooling_layer=None).item()
				else:
					target_edge_type = target_edge_type.reshape(-1, logits.shape[-1])
					loss_val = loss(outs_br, target_edge_type)
					loss_margin_val = loss_val

				val_acc = get_acc_multihot(logits, target_edge_type, threshold=threshold)
				num_data_points += target_edge_type.shape[0]
				corrects += val_acc*target_edge_type.shape[0]
				loss_margin_tot += loss_margin_val
				if fp_bp:
					h_ent = torch.tensor(0., device=device)
					y_ent = torch.tensor(0., device=device)
				else:
					h_ent = entropy(probas).sum()
					y_ent = get_ys_reg_term(model.BF_layers[0])
			val_acc = corrects/num_data_points
			# save some stats
			accs.append(val_acc.item())	

		# save the best model for this epoch
			save_model(model, epoch, opt, exp_name)
		log.info(f"Epoch train {epoch} loss: {np.mean(epoch_train_losses)}")
		# log.info(f'aggr hparam: {torch.sigmoid(model.BF_layers[0].lambdaa).item()}')
		log.info(f"Epoch val {epoch} loss xent: {loss_val.item()}, acc: {val_acc.item()}, loss mar: {loss_margin_tot}")	
		# log.info(f"Epoch {epoch} val confusion mat: \n{confusion_matrix(logits.detach().cpu().argmax(dim=-1), target_edge_type.detach().cpu())}")
		log.info(f'lr is: {get_lr(opt)}')
		all_epoch_ys.append(model.BF_layers[0].multi_embedding.detach().cpu().numpy())		
		if entropy_reg > 0:
			log.info(f'reg-ent: {entropy_reg*entropy_loss_ys.item()}')
			h_entropies.append(h_ent.item())
			y_entropies.append(entropy_reg*entropy_loss_ys.item())
		

		if val_acc.item() > best_acc:
			best_acc = val_acc.item()
			best_epoch = epoch
		# if dataset_type == 'graphlog':
		# 	best_epoch = epoch
		epoch += 1
	end = time.time()
	print(f'training time taken (seconds): {end-start}')
	# TAG: wandb 
	if config.do_hyper_sweep:
		# only hypersweep on the training set as the test set is specifically for SG 
		wandb.log({"val_acc": val_acc.item()})
	# clean up
	# best_epoch=epochs-1
	remove_not_best_models(exp_name, best_epoch)
	# load the best model and change to eval mode
	load_model_state(model, f"../models/{exp_name}_model_epoch_{best_epoch}.pth", opt)
	# remove model again
	# remove_not_best_models(exp_name, -1)
	if config.shared:
		if not just_discretize:
			model.eval_mode = True
			model.BF_layers[0].eval_mode = True
	# call dataset-agnostic test function 
	test_accs = get_test_metrics(data_train_path, unique_edge_labels, unique_query_labels,
					             remove_not_chains, bw_only, model, final_linear,
					 			 fp_bp, infer, fw_only, use_margin_loss, outs_as_left_arg,
								 score_fn, dataset_type, batch_size, threshold, use_margin_loss_multi)
	# save results
	results_dict = dict()
	results_dict['test_accs'] = test_accs
	results_dict['accs'] = accs
	results_dict['h_entropies'] = h_entropies
	results_dict['y_entropies'] = y_entropies
	save_results_models_config(config, exp_name, results_dir, 
							   [model, best_epoch, opt], results_dict)

	save_array(all_epoch_ys, results_dir, exp_name, "all_epoch_ys")
	

if __name__ == '__main__':
	
	config = OmegaConf.load('../configs/config.yaml')
	if config.do_hyper_sweep:
		# import os
		# os.environ['WANDB_MODE'] = 'offline'
		sweep_id = make_sweeper_dict(config)
	if config.do_hyper_sweep:
		wandb.agent(sweep_id, function=run, count=config.num_hyper_runs)
	else:
		run()
