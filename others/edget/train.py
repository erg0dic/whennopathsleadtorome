from ast import Assert
import os
import argparse
import re
import random
import numpy as np
import test
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from pytorch_lightning import seed_everything
import time


from model import EdgeTransformer
from rat import RAT
import json
from src.train import ClutrrDataset, get_pickle_filename
from src.utils import log, read_datafile, preprocess_graphlog_dataset, load_rcc8_file_as_dict, load_amb_data_as_dict
import pickle

torch.set_float32_matmul_precision('medium')

# Hyperparameters to tune
# lr
# batch size
# num layers (6, 8) 
# num heads (4, 8)
# dim (200, 400)

parser = argparse.ArgumentParser()
parser.add_argument('--model_type',type=str, default="edge_transformer")
parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--epochs',type=int, default=40)
parser.add_argument('--batch_size',type=int, default=32)
parser.add_argument('--num_message_rounds',type=int, default=2)
parser.add_argument('--dropout',type=float, default=0.2)
parser.add_argument('--dim',type=int, default=256)
parser.add_argument('--num_heads',type=int, default=32)
parser.add_argument('--max_grad_norm',type=float,default=1.0)
parser.add_argument('--share_layers', dest='share_layers', action='store_true')
parser.add_argument('--no_share_layers', dest='share_layers', action='store_false')
parser.set_defaults(share_layers=True)
parser.add_argument('--data_path',type=str,default='data_9b2173cf')
parser.add_argument('--lesion_values', action='store_true')
parser.add_argument('--lesion_scores',  action='store_true')
parser.add_argument('--update_relations', type=str, default='True') #this is for relation transformer
parser.add_argument('--flat_attention', action='store_true') 
parser.add_argument('--zero_init', dest='zero_init', action='store_true')  #initialization strategy for relation aware transformer
parser.add_argument('--random_init', dest='zero_init', action='store_false')
parser.set_defaults(zero_init=True)
parser.add_argument('--optimizer',type=str,default="Adam")
parser.add_argument('--scheduler',type=str,default="linear_warmup")
parser.add_argument('--num_warmup_steps',type=int,default=100)
parser.add_argument('--ff_factor',type=int,default=4)
parser.add_argument('--log_file',type=str,default='logs/clutrr_log_file.csv')
parser.add_argument('--precision',type=int,default=32,choices=[16,32])
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--dataset_type',type=str, default='clutrr')
parser.add_argument('--input_rep',type=str, default='multiedge')

parser.add_argument('--exp_name',type=str,default="edget")

cl_args = parser.parse_args()
if str(cl_args.update_relations) == 'False':
	cl_args.update_relations = False
else:
	cl_args.update_relations = True



def train():

	train_loader, validation_loader, test_loaders, test_filenames = load_files(cl_args.dataset_type)
	
	optimizer_args = {'lr':cl_args.lr}
	num_training_steps = cl_args.epochs*len(train_loader)
	scheduler_args = {'num_warmup_steps':cl_args.num_warmup_steps,'num_training_steps':num_training_steps}
	cl_args.optimizer_args = optimizer_args
	cl_args.scheduler_args = scheduler_args
	

	trainer = pl.Trainer(
			# gpus=1,
			max_epochs=cl_args.epochs,
			gradient_clip_val=cl_args.max_grad_norm,
			# progress_bar_refresh_rate=1,
			precision=cl_args.precision
		)

	if cl_args.model_type=='edge_transformer':
		model = EdgeTransformer(cl_args)
	elif cl_args.model_type=='rat':
		model = RAT(cl_args)
	################### PERSONAL MOD ###################
	path = f'{cl_args.exp_name}.pth'
	if not os.path.exists(path):
		start = time.time()
		trainer.fit(model,train_loader,validation_loader)
		end = time.time()
		print(f"Training Time taken: {end-start}")

		torch.save(model.state_dict(), path)
	else:
		model.load_state_dict(torch.load(path))
	################### PERSONAL MOD ###################
	test_acc_schema = {'test_accs': {}, 'macro_f1s': {}, 'micro_f1s': {}}
	start = time.time()
	if cl_args.dataset_type not in  ['ambiguity', 'no_ambiguity']:
		test_names_rcc8 = [(pl,brl) for pl in [2,3,4,5,6,7,8,9] for brl in [1,2,3,4,5,6]] 
		for i in range(len(test_loaders)):
			test_loader = test_loaders[i]
			test_filename = test_filenames[i]
			if cl_args.dataset_type == 'clutrr':
				chain_length = int(re.search("[0-9]+_test", test_filename).group().split("_")[0])
				print(test_filename)
				# add filename to the model loss dict
				model.test_accs.append([chain_length])
			elif cl_args.dataset_type == 'graphlog':
				model.test_accs.append([0])
			elif cl_args.dataset_type in ['rcc8', 'interval']:
				model.test_accs.append([test_names_rcc8[i]])
			trainer.test(model,dataloaders=test_loader)

	################### PERSONAL MOD ###################
	# custom save some metrics in the format that is consistent with other models for later comparisons
		if cl_args.dataset_type in ['rcc8', 'interval']:
			test_acc_schema = {"test_accs": {}}
			new_test_accs = model.test_accs
			for test_acc in new_test_accs:
				config = test_acc[0]
				acc = np.array(test_acc[1:]).mean()
				test_acc_schema["test_accs"][str(config)] = acc
		elif cl_args.dataset_type not in ['ambiguity', 'no_ambiguity']:
			test_acc_schema = {"test_accs": [0 for _ in range(2, 10+1)]}
			if len(model.test_accs[0]) > 2:
				new_test_accs = []
				for l in model.test_accs:
					new_test_accs.append([l[0], np.array(l[1:]).mean()])
			else:
				new_test_accs = model.test_accs
			for (chain_length, acc) in new_test_accs:
				test_acc_schema["test_accs"][chain_length-2] = acc

	else:

		if cl_args.dataset_type == 'no_ambiguity':
			ratios = [1.17,1.33,1.4,1.0,1.14,1.5,1.29,1.43,1.38,1.25,1.22,1.12,1.44,1.11,0.88,1.1]
		elif cl_args.dataset_type == 'ambiguity':
			ratios =  [1.14, 1.17, 1.0, 1.4, 1.29, 1.43, 1.5, 1.38, 1.33, 1.12, 1.25, 0.88]
		else:
			raise AssertionError(f"Unknown dataset type: {cl_args.dataset_type}")
		
		# test for the reasoning depth on the long chain data
		test_loader_counter = 0

		for k in [7,8,9,10,11,12,13]:
			test_loader = test_loaders[test_loader_counter]
			print(test_filenames[test_loader_counter])
			test_filename = f"('long_k', {k})"
			model.test_accs.append([test_filename])
			model.macro_f1s.append([test_filename])
			model.micro_f1s.append([test_filename])
			test_loader_counter += 1
			
			# load the test metric here and save it
			trainer.test(model,dataloaders=test_loader)

		for mrnr in ratios:
			test_loader = test_loaders[test_loader_counter]
			print(test_filenames[test_loader_counter])
			test_filename = f"('long_mrnr', {mrnr})"
			model.test_accs.append([test_filename])
			model.macro_f1s.append([test_filename])
			model.micro_f1s.append([test_filename])
			test_loader_counter += 1
			trainer.test(model,dataloaders=test_loader)

		for OPEC in [3,4]:
			test_loader = test_loaders[test_loader_counter]
			print(test_filenames[test_loader_counter])
			test_filename = f"('OPEC', {OPEC})"
			model.test_accs.append([test_filename])
			model.macro_f1s.append([test_filename])
			model.micro_f1s.append([test_filename])
			test_loader_counter += 1
			trainer.test(model,dataloaders=test_loader)
			

		if cl_args.dataset_type == 'ambiguity':
			# test wrt num branches and path length
			for k in [4,5,6]:
				for brl in [1,2,3,4,5]:
					test_loader = test_loaders[test_loader_counter]
					print(test_filenames[test_loader_counter])
					test_filename = f"('s3_short_k_branches_a', ({k}, {brl})"
					model.test_accs.append([test_filename])
					model.macro_f1s.append([test_filename])
					model.micro_f1s.append([test_filename])
					test_loader_counter += 1
					trainer.test(model,dataloaders=test_loader)
			for brl in [6,7,8,9,10,11,12]:
				test_loader = test_loaders[test_loader_counter]
				print(test_filenames[test_loader_counter])
				test_filename = f"('s2_ood_short_k_branches_a', ({brl}))"
				model.test_accs.append([test_filename])
				model.macro_f1s.append([test_filename])
				model.micro_f1s.append([test_filename])
				test_loader_counter += 1
				trainer.test(model,dataloaders=test_loader)

		if cl_args.dataset_type == 'ambiguity':
			
			for data_filename in ['test_d', 'test_w', 'test_bl', 'test_opec']:
				test_loader = test_loaders[test_loader_counter]
				print(test_filenames[test_loader_counter])
				test_filename = data_filename
				model.test_accs.append([test_filename])
				model.macro_f1s.append([test_filename])
				model.micro_f1s.append([test_filename])
				trainer.test(model, dataloaders=test_loader)
				test_loader_counter += 1
		else:
			for data_filename in ['test_d', 'test_bl', 'test_opec']:
				test_loader = test_loaders[test_loader_counter]
				
				print(test_filenames[test_loader_counter])
				test_filename = data_filename
				model.test_accs.append([test_filename])
				model.macro_f1s.append([test_filename])
				model.micro_f1s.append([test_filename])
				trainer.test(model,dataloaders=test_loader)
				test_loader_counter += 1
			
		new_test_accs = model.test_accs
		new_macro_f1s = model.macro_f1s
		new_micro_f1s = model.micro_f1s
		for test_acc, micro, macro in zip(new_test_accs, new_micro_f1s, new_macro_f1s):
			config = test_acc[0]
			

			acc = test_acc[1]
			micro = micro[1]
			macro = macro[1]

			test_acc_schema['test_accs'][config] = acc
			test_acc_schema['macro_f1s'][config] = macro
			test_acc_schema['micro_f1s'][config] = micro


	exp_dir = f"../../results/edge_t_{cl_args.exp_name}"
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
		# save model.test_accs as a json file
	end = time.time()
	print(f"Inference Time taken: {end-start}")
	json.dump(test_acc_schema, open(os.path.join(exp_dir, 'results.json'), 'w'))
	################### PERSONAL MOD ###################

##Load data##
def load_dataset(data_filename: str, unique_edge_labels: list = None, 
				 unique_query_labels: list = None, dataset_type='clutrr') -> ClutrrDataset:
	pfname = get_pickle_filename(data_filename)
	
	if not os.path.exists(pfname):
		if dataset_type == 'clutrr':
			data = read_datafile(data_filename)
			data = ClutrrDataset(data, reverse=False, fp_bp=False, unique_edge_labels=unique_edge_labels, unique_query_labels=unique_query_labels)
			pickle.dump(data, open(pfname, 'wb'))
		elif dataset_type == 'graphlog':
			data = preprocess_graphlog_dataset(data_filename)
			data = ClutrrDataset(data, False, False, unique_edge_labels, unique_query_labels)
			pickle.dump(data, open(pfname, 'wb'))
		elif dataset_type in ['rcc8', 'interval', 'ambiguity', 'no_ambiguity']:
			data = load_rcc8_file_as_dict(data_filename)
			data = ClutrrDataset(data, False, False, unique_edge_labels, unique_query_labels)
			pickle.dump(data, open(pfname, 'wb'))
		else:
			raise ValueError(f"Unknown dataset type: {dataset_type}")
	else:
		log.info(f"preprocessed data file loaded from: {pfname}")
		data = pickle.load(open(pfname, 'rb'))
		# assert len(data.unique_edge_labels) == 20
	return data

def batch_edges(edges,edge_labels, num_edge_types):
	batch_size = len(edges)
	lens = torch.tensor(list(map(lambda x: torch.max(x)+1,edges)))
	max_len = max(lens)

	mask = torch.arange(max_len)[None, :] >= lens[:, None]

	batch = []
	for i in range(batch_size):
		s = torch.zeros(max_len,max_len).long()
		edge = edges[i]
		lab = edge_labels[i]
		s[edge[:,0],edge[:,1]] = lab
		batch.append(s)
	batch = torch.stack(batch)
	return batch, mask

from typing import List, Tuple, Callable
def batch_edges_multi(
	edges: List[torch.Tensor],
	edge_labels: List[torch.Tensor],
	num_edge_types: int | None = None):

	B = len(edges)
	lens = torch.tensor(list(map(lambda x: torch.max(x)+1,edges)))
	max_len = max(lens)
	mask = torch.arange(max_len)[None, :] >= lens[:, None]

	batched = torch.zeros(
		(B, num_edge_types, max_len, max_len), dtype=torch.float
	)
	# print('edge types:', num_edge_types)
	for i, (e, lab) in enumerate(zip(edges, edge_labels)):
		# Advanced indexing: shapes all (Eᵢ,)
		# print(e[:, 0])
		batched[i, lab, e[:, 0], e[:, 1]] = 1.
		# print(batched[i])
		# breakpoint()
	return batched, mask

def collate(data, batch_edges_fn,num_edge_types=None):
	batch_size = len(data)
	edges = [d['edge_index'].permute(1,0) for d in data]
	edge_labels = [d['edge_type'] for d in data]
	query_edge = [d['target_edge_index'].squeeze(1) for d in data]
	query_label = [d['target_edge_type'] for d in data]
	
	batched_edges, mask = batch_edges_fn(edges,edge_labels, num_edge_types=num_edge_types)
	batched_query_edges = torch.stack(query_edge)
	batched_query_edges = torch.cat((torch.arange(batch_size).unsqueeze(1),batched_query_edges),dim=1)
	
	
	if isinstance(query_label[0], torch.Tensor):
		batched_query_labels = torch.stack(query_label)
	else:
		batched_query_labels = torch.tensor(query_label)
	# print('batched_query_labels:', batched_query_labels.shape)

	batched = {}
	batched['batched_graphs']=batched_edges
	batched['target_edge_index'] = batched_query_edges
	batched['target_edge_type'] = batched_query_labels
	batched['masks'] = mask

	return batched
	# padded_edges = []
	# max_len = max(torch.tensor(list(map(lambda x: torch.max(x)+1,edges))))
	# for edge in edges:
	# 	new_edge = edge.tolist()
	# 	if len(edge) < max_len:
	# 		new_edge.extend([[-1,-1]]*(max_len-len(edge)))
	# 	padded_edges.append(new_edge)
	# batched['edge_index'] = torch.as_tensor(padded_edges, dtype=torch.long)

from functools import partial

def load_files(dataset_type='clutrr'):
	if dataset_type == 'clutrr':
		train_filename = get_filenames('train', dataset_type)[0]
		test_filenames = get_filenames('test', dataset_type)
	elif dataset_type == 'graphlog':
		train_filename = get_filenames('train', dataset_type)[0]
		val_filename = get_filenames('valid', dataset_type)[0]
	
		test_filenames = get_filenames('test', dataset_type)
	elif dataset_type in ['rcc8', 'interval']:
		train_filename = f'../../data/rcc8/train_{dataset_type}.csv'
		test_filenames = []
		for pl in [2,3,4,5,6,7,8,9]:
			for brl in [1,2,3,4,5,6]: 
				fname = f'../../data/rcc8/test_{dataset_type}_k_{pl}_b_{brl}.csv'
				test_filenames.append(fname)
	elif dataset_type in ['ambiguity', 'no_ambiguity']:
		if dataset_type == 'ambiguity':
			train_filename = f'../../data/ambig/train_ambig.csv'
			prefix = 'ambig'
			ratios =  [1.14, 1.17, 1.0, 1.4, 1.29, 1.43, 1.5, 1.38, 1.33, 1.12, 1.25, 0.88]
		else:
			train_filename = f'../../data/ambig/train_no_ambig.csv'
			prefix = 'no_ambig'
			ratios = [1.17,1.33,1.4,1.0,1.14,1.5,1.29,1.43,1.38,1.25,1.22,1.12,1.44,1.11,0.88,1.1]



		test_filenames = []
		# chain lens
		for k in [7,8,9,10,11,12,13]:
			fname = f'../../data/ambig/test_{prefix}_long_k_{k}.csv'
			test_filenames.append(fname)
		# mrnr
		for mrnr in ratios:
			fname = f'../../data/ambig/test_{prefix}_long_mrnr_{mrnr}.csv'
			test_filenames.append(fname)
		
		for OPEC in [3,4]:
			fname = f'../../data/ambig/test_{prefix}_OPEC_{OPEC}.csv'
			test_filenames.append(fname)

		if dataset_type == 'ambiguity':
			# brl and k
			for k in [4,5,6]:
				for brl in [1,2,3,4,5]:
					fname = f'../../data/ambig/test_{prefix}_short_b_{brl}_k_{k}.0.csv'
					test_filenames.append(fname)
			for brl in [6,7,8,9,10,11,12]:
				fname = f'../../data/ambig/test_{prefix}_short2_b_{brl}.csv'
				test_filenames.append(fname)

		if dataset_type == 'ambiguity':
			
			for data_filename in ['test_d', 'test_w', 'test_bl', 'test_opec']:
				fname = f'../../data/ambig/test_{prefix}_{data_filename}.csv'
				test_filenames.append(fname)
		else:
			for data_filename in ['test_d', 'test_bl', 'test_opec']:
				fname = f'../../data/ambig/test_{prefix}_{data_filename}.csv'
				test_filenames.append(fname)


	data_params = {'batch_size': cl_args.batch_size,
				'shuffle': False,
				'drop_last':False,
				'num_workers':8
				}

	test_params = {'batch_size': cl_args.batch_size if dataset_type not in ['rcc8', 'interval'] else 1,
				'shuffle': False,
				'drop_last':False,
				'num_workers':8
				}
	if dataset_type != 'graphlog':
		training_data = load_dataset(train_filename, dataset_type=dataset_type)

		unique_edge_labels = training_data.unique_edge_labels
		unique_query_labels = training_data.unique_query_labels
		cl_args.edge_types = training_data.num_edge_labels+1
		cl_args.target_size = training_data.num_query_labels
		
		training_len = int(0.8*len(training_data))
		validation_len = len(training_data) - training_len
		training_set, validation_set = torch.utils.data.random_split(training_data, [training_len, validation_len])

		if cl_args.input_rep == 'multiedge':
			collate_fn: Callable = partial(collate, batch_edges_fn=batch_edges_multi, num_edge_types=training_data.num_edge_labels+1)
		else:
			collate_fn: Callable = partial(collate, batch_edges_fn=batch_edges)
		
		training_loader = DataLoader(training_set, **data_params,collate_fn=collate_fn)
		validation_loader = DataLoader(validation_set, **data_params,collate_fn=collate_fn)
		
	else:
		training_data = load_dataset(train_filename, dataset_type=dataset_type)
		
		unique_edge_labels = training_data.unique_edge_labels
		unique_query_labels = training_data.unique_query_labels
		cl_args.edge_types = training_data.num_edge_labels+1
		cl_args.target_size = training_data.num_query_labels

		val_data = load_dataset(val_filename, unique_edge_labels, unique_query_labels, dataset_type)

		training_loader = DataLoader(training_data, **data_params,collate_fn=collate)
		validation_loader = DataLoader(val_data, **data_params,collate_fn=collate)
		
	test_loaders = []
	for test_filename in test_filenames:
		test_data = load_dataset(test_filename, unique_edge_labels, unique_query_labels, dataset_type)
		test_loader = DataLoader(test_data, **test_params, collate_fn=collate_fn)
		test_loaders.append(test_loader)

	return training_loader,validation_loader,test_loaders,test_filenames


def get_filenames(data_file_name, dataset_type='clutrr'):
	data_dir = os.path.join('../../data/',cl_args.data_path)
	files = []
	for file in os.listdir(data_dir):
		check = file.endswith(data_file_name+'.csv') if dataset_type != 'graphlog' else file.endswith(data_file_name+'.jsonl')
		if check:
			fname = os.path.join(data_dir, file)
			files.append(fname)
	return files

def set_random_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	seed_everything(seed=seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


if __name__=="__main__":
	set_random_seed(cl_args.seed)
	train()