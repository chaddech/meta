import numpy as np
import pickle
import os
from collections import defaultdict
import torch
import torch.nn.functional as F
from scipy.stats import entropy


resnet152_output_correct = torch.load('/media/chad/delft/meta/imagenet_intermediates/resnet152/valid/output/correct/test_sorted_outputs.torch')
resnet152_output_incorrect = torch.load('/media/chad/delft/meta/imagenet_intermediates/resnet152/valid/output/incorrect/test_sorted_outputs.torch')

vgg16_output_correct = torch.load('/media/chad/delft/meta/imagenet_intermediates/vgg16_bn/valid/output/correct/test_sorted_outputs.torch')
vgg16_output_incorrect = torch.load('/media/chad/delft/meta/imagenet_intermediates/vgg16_bn/valid/output/incorrect/test_sorted_outputs.torch')

resnet18_output_correct = torch.load('/media/chad/delft/meta/imagenet_intermediates/resnet18/valid/output/correct/test_sorted_outputs.torch')
resnet18_output_incorrect = torch.load('/media/chad/delft/meta/imagenet_intermediates/resnet18/valid/output/incorrect/test_sorted_outputs.torch')

densenet161_output_correct = torch.load('/media/chad/delft/meta/imagenet_intermediates/densenet161/valid/output/correct/test_sorted_outputs.torch')
densenet161_output_incorrect = torch.load('/media/chad/delft/meta/imagenet_intermediates/densenet161/valid/output/incorrect/test_sorted_outputs.torch')

alexnet_output_correct = torch.load('/media/chad/delft/meta/imagenet_intermediates/alexnet/valid/output/correct/test_sorted_outputs.torch')
alexnet_output_incorrect = torch.load('/media/chad/delft/meta/imagenet_intermediates/alexnet/valid/output/incorrect/test_sorted_outputs.torch')

outputs_array = np.zeros((50000,5))


right_answers = {}
right_answers_array = np.zeros((50000))
for i in resnet152_output_incorrect:
	right_answers[i[1].item()] = i[3].cpu().item()
	right_answers_array[i[1].item()] = i[3].cpu().item()
for i in resnet152_output_correct:
	right_answers[i[1].item()] = i[3].cpu().item()
	right_answers_array[i[1].item()] = i[3].cpu().item()

resnet152_outputs = {}
for i in resnet152_output_correct:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	resnet152_outputs[identification] = base_output
	outputs_array[identification][3] = base_output 

for i in resnet152_output_incorrect:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	resnet152_outputs[identification] = base_output
	outputs_array[identification][3] = base_output 

resnet18_full_outputs = {}

resnet18_outputs = {}
for i in resnet18_output_correct:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	resnet18_outputs[identification] = base_output
	outputs_array[identification][2] = base_output 
	resnet18_full_outputs[identification] = i[0]
for i in resnet18_output_incorrect:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	resnet18_outputs[identification] = base_output
	outputs_array[identification][2] = base_output 
	resnet18_full_outputs[identification] = i[0]

densenet161_outputs = {}
for i in densenet161_output_correct:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	densenet161_outputs[identification] = base_output
	outputs_array[identification][4] = base_output 

for i in densenet161_output_incorrect:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	densenet161_outputs[identification] = base_output
	outputs_array[identification][4] = base_output 

vgg16_outputs = {}
for i in vgg16_output_correct:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	vgg16_outputs[identification] = base_output
	outputs_array[identification][1] = base_output 

for i in vgg16_output_incorrect:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	vgg16_outputs[identification] = base_output
	outputs_array[identification][1] = base_output 

alexnet_outputs = {}
for i in alexnet_output_correct:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	alexnet_outputs[identification] = base_output
	outputs_array[identification][0] = base_output 

for i in alexnet_output_incorrect:
	identification = i[1].item()
	base_output = np.argmax(np.exp(i[0])).item()

	alexnet_outputs[identification] = base_output
	outputs_array[identification][0] = base_output 


loaded_results = []
outer_folder= '/media/chad/nara/meta/imagenet/condensed_results/'
for folder in os.listdir(outer_folder):
	for results_file in os.listdir(outer_folder+folder+'/'):
		these_results = open(outer_folder+folder+'/'+results_file, "rb")
		loaded_results.append(pickle.load(these_results))


any_base_correct = defaultdict(int)
results_info = []
for net in loaded_results:
	counter = 0
	for a in net[2:4]:
		for item in a:
			counter += 1

			any_base_correct[item] += 1
	results_info.append((net[0], counter))

all_indices = set()
for net in loaded_results:
	for a in net[2:]:
		for item in a:
			all_indices.add(item)


all_results = np.zeros((50000,4))

base_correct_meta_correct = defaultdict(int)
for net in loaded_results:
	counter = 0
	for item in net[2]:
		base_correct_meta_correct[item] += 1
		all_results[item][0] += 1

base_correct_meta_incorrect = defaultdict(int)
for net in loaded_results:
	counter = 0
	for item in net[3]:
		base_correct_meta_incorrect[item] += 1
		all_results[item][1] += 1

base_incorrect_meta_correct = defaultdict(int)
for net in loaded_results:
	counter = 0
	for item in net[4]:
		base_incorrect_meta_correct[item] += 1
		all_results[item][2] += 1


base_incorrect_meta_incorrect = defaultdict(int)
for net in loaded_results:
	counter = 0
	for item in net[5]:
		base_incorrect_meta_incorrect[item] += 1
		all_results[item][3] += 1

layer_verdicts_table = np.zeros((50000, len(loaded_results)))
names = []
#if meta network looking at layer says correct, 1; if meta network says incorrect, 0
for i in range(len(loaded_results)):
	names.append(loaded_results[i][0])
	for index in loaded_results[i][2]:
		layer_verdicts_table[index][i] = 1

	for index in loaded_results[i][3]:
		layer_verdicts_table[index][i] = 0

	for index in loaded_results[i][4]:
		layer_verdicts_table[index][i] = 0

	for index in loaded_results[i][5]:
		layer_verdicts_table[index][i] = 1

new_order = [2,5,11,0,7,10,1,8,4,9,3,6]
layer_names = [names[i] for i in new_order]

layer_verdicts_table = layer_verdicts_table[:,np.asarray(new_order)]
layer_verdicts_table = layer_verdicts_table.astype(int)

majority_or_default_reject_base_classifiers = np.zeros((50000, 5))


for i in range(50000):
	majority_or_default_reject_base_classifiers[i][0] = np.bincount(layer_verdicts_table[i][:3]).argmax()

	majority_or_default_reject_base_classifiers[i][1] = np.bincount(layer_verdicts_table[i][3:6]).argmax()

	if layer_verdicts_table[i][6] != layer_verdicts_table[i][7]:
		majority_or_default_reject_base_classifiers[i][2] = 0
	else:
		majority_or_default_reject_base_classifiers[i][2] = layer_verdicts_table[i][6]

	if layer_verdicts_table[i][8] != layer_verdicts_table[i][9]:
		majority_or_default_reject_base_classifiers[i][3] = 0
	else:
		majority_or_default_reject_base_classifiers[i][3] = layer_verdicts_table[i][8]

	if layer_verdicts_table[i][10] != layer_verdicts_table[i][11]:
		majority_or_default_reject_base_classifiers[i][4] = 0
	else:
		majority_or_default_reject_base_classifiers[i][4] = layer_verdicts_table[i][10]


majority_or_default_accept_base_classifiers = np.zeros((50000, 5))

for i in range(50000):
	majority_or_default_accept_base_classifiers[i][0] = np.bincount(layer_verdicts_table[i][:3]).argmax()

	majority_or_default_accept_base_classifiers[i][1] = np.bincount(layer_verdicts_table[i][3:6]).argmax()

	if layer_verdicts_table[i][6] != layer_verdicts_table[i][7]:
		majority_or_default_accept_base_classifiers[i][2] = 1
	else:
		majority_or_default_accept_base_classifiers[i][2] = layer_verdicts_table[i][6]

	if layer_verdicts_table[i][8] != layer_verdicts_table[i][9]:
		majority_or_default_accept_base_classifiers[i][3] = 1
	else:
		majority_or_default_accept_base_classifiers[i][3] = layer_verdicts_table[i][8]

	if layer_verdicts_table[i][10] != layer_verdicts_table[i][11]:
		majority_or_default_accept_base_classifiers[i][4] = 1
	else:
		majority_or_default_accept_base_classifiers[i][4] = layer_verdicts_table[i][10]

outputs_array = outputs_array.astype(int)

nonzero_indices = list(right_answers.keys())
nonzero_indices.sort()

only_nonzero_outputs_array = outputs_array[nonzero_indices]
only_nonzero_majority_or_default_reject_base_classifiers = majority_or_default_reject_base_classifiers[nonzero_indices]
only_nonzero_majority_or_default_accept_base_classifiers = majority_or_default_accept_base_classifiers[nonzero_indices]
only_nonzero_right_answers = right_answers_array[nonzero_indices]
only_nonzero_right_answers = only_nonzero_right_answers.astype(int)

simple_majority_vote_of_base_classifiers = np.zeros((25000))
for i in range(25000):
	simple_majority_vote_of_base_classifiers[i] = np.bincount(only_nonzero_outputs_array[i]).argmax()

simple_majority_vote_correct = only_nonzero_right_answers == simple_majority_vote_of_base_classifiers
num_simple_majority_vote_correct = simple_majority_vote_correct.sum()
print("number correct of simple majority ensemble: " + str(num_simple_majority_vote_correct))

verdict_of_majority_or_default_reject_base_classifiers = np.zeros((25000))
for i in range(25000):
	votes = []
	for x in range(5):
		if only_nonzero_majority_or_default_reject_base_classifiers[i][x] == 1:
			votes.append(only_nonzero_outputs_array[i][x])
	if len(votes) == 0:
		verdict_of_majority_or_default_reject_base_classifiers[i] = simple_majority_vote_of_base_classifiers[i]
	else:
		votes = np.asarray(votes)
		counts = np.bincount(votes)
		argmax_counts = counts.argmax()
		num_max = counts.max()
		where_max = np.where(counts == num_max)[0]
		how_many_same_max = len(where_max)
		if how_many_same_max == 1:
			verdict_of_majority_or_default_reject_base_classifiers[i] = argmax_counts
		else:
			if simple_majority_vote_of_base_classifiers[i] in where_max:
				verdict_of_majority_or_default_reject_base_classifiers[i] = simple_majority_vote_of_base_classifiers[i]
			else:
				#else pick one at random ?????
				verdict_of_majority_or_default_reject_base_classifiers[i] = argmax_counts

verdict_of_majority_or_default_reject_base_classifiers_vote_correct = only_nonzero_right_answers == verdict_of_majority_or_default_reject_base_classifiers
num_verdict_of_majority_or_default_reject_base_classifiers_vote_correct = verdict_of_majority_or_default_reject_base_classifiers_vote_correct.sum()
print("number correct of majority or default reject ensemble: " + str(num_verdict_of_majority_or_default_reject_base_classifiers_vote_correct))


verdict_of_majority_or_default_accept_base_classifiers = np.zeros((25000))

for i in range(25000):
	votes = []
	for x in range(5):
		if only_nonzero_majority_or_default_accept_base_classifiers[i][x] == 1:
			votes.append(only_nonzero_outputs_array[i][x])
	if len(votes) == 0:
		verdict_of_majority_or_default_accept_base_classifiers[i] = simple_majority_vote_of_base_classifiers[i]
	else:
		votes = np.asarray(votes)
		counts = np.bincount(votes)
		argmax_counts = counts.argmax()
		num_max = counts.max()
		where_max = np.where(counts == num_max)[0]
		how_many_same_max = len(where_max)
		if how_many_same_max == 1:
			verdict_of_majority_or_default_accept_base_classifiers[i] = argmax_counts
		else:
			if simple_majority_vote_of_base_classifiers[i] in where_max:
				verdict_of_majority_or_default_accept_base_classifiers[i] = simple_majority_vote_of_base_classifiers[i]
			else:
				#else pick one at random ?????
				verdict_of_majority_or_default_accept_base_classifiers[i] = argmax_counts


keep_best_only_nonzero_majority_or_default_accept_base_classifiers = only_nonzero_majority_or_default_accept_base_classifiers
keep_best_only_nonzero_majority_or_default_accept_base_classifiers[:,3] = 1

verdict_of_majority_or_default_accept_base_classifiers_vote_correct = only_nonzero_right_answers == verdict_of_majority_or_default_accept_base_classifiers
num_verdict_of_majority_or_default_accept_base_classifiers_vote_correct = verdict_of_majority_or_default_accept_base_classifiers_vote_correct.sum()
print("number correct of majority or default accept ensemble: " + str(num_verdict_of_majority_or_default_accept_base_classifiers_vote_correct))

verdict_of_keep_best_majority_or_default_accept_base_classifiers = np.zeros((25000))

for i in range(25000):
	votes = []
	for x in range(5):
		if keep_best_only_nonzero_majority_or_default_accept_base_classifiers[i][x] == 1:
			votes.append(only_nonzero_outputs_array[i][x])
	if len(votes) == 0:
		verdict_of_keep_best_majority_or_default_accept_base_classifiers[i] = simple_majority_vote_of_base_classifiers[i]
	else:
		votes = np.asarray(votes)
		counts = np.bincount(votes)
		argmax_counts = counts.argmax()
		num_max = counts.max()
		where_max = np.where(counts == num_max)[0]
		how_many_same_max = len(where_max)
		if how_many_same_max == 1:
			verdict_of_keep_best_majority_or_default_accept_base_classifiers[i] = argmax_counts
		else:
			if simple_majority_vote_of_base_classifiers[i] in where_max:
				verdict_of_keep_best_majority_or_default_accept_base_classifiers[i] = simple_majority_vote_of_base_classifiers[i]
			else:
				#else pick one at random ?????
				if only_nonzero_outputs_array[i,3] in where_max:
					verdict_of_keep_best_majority_or_default_accept_base_classifiers[i] = only_nonzero_outputs_array[i,3]
				else:

					verdict_of_keep_best_majority_or_default_accept_base_classifiers[i] = argmax_counts

verdict_of_keep_best_majority_or_default_accept_base_classifiers_vote_correct = only_nonzero_right_answers == verdict_of_keep_best_majority_or_default_accept_base_classifiers
num_verdict_of_keep_best_majority_or_default_accept_base_classifiers_vote_correct = verdict_of_keep_best_majority_or_default_accept_base_classifiers_vote_correct.sum()
print("number correct of majority or default accept and keep best ensemble: " + str(num_verdict_of_keep_best_majority_or_default_accept_base_classifiers_vote_correct))


"""
majority_or_reject_multiplied_nonzero_outputs = np.multiply(only_nonzero_majority_or_default_reject_base_classifiers, only_nonzero_outputs_array)
majority_or_reject_multiplied_nonzero_outputs = majority_or_reject_multiplied_nonzero_outputs.astype(int)
for i in range(25000):
	verdicts = np.bincount(majority_or_reject_multiplied_nonzero_outputs[i])
	if verdicts.argmax()
	verdict_of_majority_or_default_reject_base_classifiers[i] = 



verdict_of_majority_or_default_accept_base_classifiers = np.zeros((25000))
majority_or_accept_multiplied_nonzero_outputs = np.multiply(only_nonzero_majority_or_default_accept_base_classifiers, only_nonzero_outputs_array)
majority_or_accept_multiplied_nonzero_outputs = majority_or_accept_multiplied_nonzero_outputs.astype(int)

for i in range(25000):
	verdict_of_majority_or_default_accept_base_classifiers[i] = np.bincount(majority_or_accept_multiplied_nonzero_outputs[i]).argmax()

verdict_of_majority_or_default_accept_base_classifiers_vote_correct = only_nonzero_right_answers == verdict_of_majority_or_default_accept_base_classifiers
num_verdict_of_majority_or_default_accept_base_classifiers_vote_correct = verdict_of_majority_or_default_accept_base_classifiers_vote_correct.sum()
print("number correct of majority or default accept ensemble: " + str(num_verdict_of_majority_or_default_accept_base_classifiers_vote_correct))
"""

this_layer = loaded_results[1]
means = [[] for i in range(4)]
stds = [[] for i in range(4)]
entropies = [[] for i in range(4)]
maxes = [[] for i in range(4)]

outputs = [[] for i in range(4)]
for x in range(2,6,1):
	for i in this_layer[x]:
		outputs[x-2].append(F.softmax(resnet18_full_outputs[i]))

for x in range(4):
	for i in outputs[x]:
		means[x].append(i.mean().numpy())
		stds[x].append(i.std().numpy())
		maxes[x].append(i.max().numpy())
		entropies[x].append(entropy(i.numpy()))


import IPython
IPython.embed()

correct_base_meta_accurate = []
correct_base_meta_inaccurate = []
incorrect_base_meta_accurate = []
incorrect_base_meta_inaccurate = []
