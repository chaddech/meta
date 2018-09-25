import torch
import numpy as np
import os

valid_correct_tensors = []
valid_error_tensors = []

train_correct_tensors = []
train_error_tensors = []


valid_file_names = []
inter_layers = ['output', 'last_conv_layer','pre_bn_layer']
for inter_layer in inter_layers:

	valid_correct_dir = '/media/chad/nara/cifar/wide/valid/'+inter_layer+'/correct/'
	valid_error_dir = '/media/chad/nara/cifar/wide/valid/'+inter_layer+'/incorrect/'

	valid_correct_files = os.listdir(valid_correct_dir)
	valid_error_files = os.listdir(valid_error_dir)

	valid_correct_files.sort()
	valid_error_files.sort()

	for i in valid_correct_files:
		t = torch.load(valid_correct_dir+i)
		valid_file_names.append(i)
		for x in t:
			valid_correct_tensors.append(x)


	for i in valid_error_files:
		t = torch.load(valid_error_dir+i)
		for x in t:
			valid_error_tensors.append(x)

	torch.save(valid_correct_tensors, '/media/chad/nara/cifar/wide/valid/'+inter_layer+'/correct_sorted_outputs.torch')
	torch.save(valid_error_tensors, '/media/chad/nara/cifar/wide/valid/'+inter_layer+'/incorrect_sorted_outputs.torch')




	train_correct_dir = '/media/chad/nara/cifar/wide/train/'+inter_layer+'/correct/'
	train_error_dir = '/media/chad/nara/cifar/wide/train/'+inter_layer+'/incorrect/'

	train_correct_files = os.listdir(train_correct_dir)
	train_error_files = os.listdir(train_error_dir)

	train_correct_files.sort()
	train_error_files.sort()

	for i in train_correct_files:
		t = torch.load(train_correct_dir+i)
		for x in t:
			train_correct_tensors.append(x)


	for i in train_error_files:
		t = torch.load(train_error_dir+i)
		for x in t:
			train_error_tensors.append(x)

	torch.save(train_correct_tensors, '/media/chad/nara/cifar/wide/train/'+inter_layer+'/correct_sorted_outputs.torch')
	torch.save(train_error_tensors, '/media/chad/nara/cifar/wide/train/'+inter_layer+'/incorrect_sorted_outputs.torch')
