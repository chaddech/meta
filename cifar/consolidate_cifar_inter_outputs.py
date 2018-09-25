import torch
import os

valid_file_names = []
for inter_layer in next(os.walk(os.getcwd() + '/wide/valid/'))[1]:
    print(inter_layer)
    valid_correct_tensors = []
    valid_error_tensors = []

    train_correct_tensors = []
    train_error_tensors = []

    valid_correct_dir = os.getcwd() + '/wide/valid/' + inter_layer + '/correct/'
    valid_error_dir = os.getcwd() + '/wide/valid/' + inter_layer + '/incorrect/'

    if not os.path.exists(valid_correct_dir):
        os.makedirs(valid_correct_dir)
    if not os.path.exists(valid_error_dir):
        os.makedirs(valid_error_dir)

    valid_correct_files = os.listdir(valid_correct_dir)
    valid_error_files = os.listdir(valid_error_dir)

    valid_correct_files.sort()
    valid_error_files.sort()

    for i in valid_correct_files:
        t = torch.load(valid_correct_dir + i)
        valid_file_names.append(i)
        for x in t:
            valid_correct_tensors.append(x)

    for i in valid_error_files:
        t = torch.load(valid_error_dir + i)
        for x in t:
            valid_error_tensors.append(x)

    torch.save(valid_correct_tensors, os.getcwd() + '/wide/valid/' + inter_layer + '/correct_outputs.torch')
    torch.save(valid_error_tensors, os.getcwd() + '/wide/valid/' + inter_layer + '/incorrect_outputs.torch')

    train_correct_dir = os.getcwd() + '/wide/train/' + inter_layer + '/'
    train_error_dir = os.getcwd() + '/wide/train/' + inter_layer + '/'

    if not os.path.exists(train_correct_dir):
        os.makedirs(train_correct_dir)
    if not os.path.exists(train_error_dir):
        os.makedirs(train_error_dir)

    train_correct_files = os.listdir(train_correct_dir)
    train_error_files = os.listdir(train_error_dir)

    train_correct_files.sort()
    train_error_files.sort()
    
    for i in train_correct_files:
        t = torch.load(train_correct_dir + i)
        for x in t:
            train_correct_tensors.append(x)

    for i in train_error_files:
        t = torch.load(train_error_dir + i)
        for x in t:
            train_error_tensors.append(x)

    torch.save(train_correct_tensors, os.getcwd() + '/wide/train/' + inter_layer + '/correct_outputs.torch')
    torch.save(train_error_tensors, os.getcwd() + '/wide/train/' + inter_layer + '/incorrect_outputs.torch')

