import torch
import numpy as np
import os

PATH = '/media/chad/nara/'

valid_correct_tensors = []
valid_error_tensors = []

train_correct_tensors = []
train_error_tensors = []

inter_layer = 'output'
valid_correct_dir = PATH + 'cifar/wide/valid/'+inter_layer+'/correct/'
valid_error_dir = PATH + 'cifar/wide/valid/'+inter_layer+'/incorrect/'

valid_correct_files = os.listdir(valid_correct_dir)
valid_error_files = os.listdir(valid_error_dir)

for i in valid_correct_files:
	t = torch.load(valid_correct_dir+i)
	for x in t:
		valid_correct_tensors.append(x)


for i in valid_error_files:
	t = torch.load(valid_error_dir+i)
	for x in t:
		valid_error_tensors.append(x)

torch.save(valid_correct_tensors, PATH + 'cifar/wide/valid/'+inter_layer+'/correct_outputs.torch')
torch.save(valid_error_tensors, PATH + 'cifar/wide/valid/'+inter_layer+'/incorrect_outputs.torch')




train_correct_dir = PATH + 'cifar/wide/train/'+inter_layer+'/correct/'
train_error_dir = PATH + 'cifar/wide/train/'+inter_layer+'/incorrect/'

train_correct_files = os.listdir(train_correct_dir)
train_error_files = os.listdir(train_error_dir)

for i in train_correct_files:
	t = torch.load(train_correct_dir+i)
	for x in t:
		train_correct_tensors.append(x)


for i in train_error_files:
	t = torch.load(train_error_dir+i)
	for x in t:
		train_error_tensors.append(x)

torch.save(train_correct_tensors, PATH + 'cifar/wide/train/'+inter_layer+'/correct_outputs.torch')
torch.save(train_error_tensors, PATH + 'cifar/wide/train/'+inter_layer+'/incorrect_outputs.torch')
