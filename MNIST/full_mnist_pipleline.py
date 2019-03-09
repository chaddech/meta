
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data.dataloader import default_collate
import sys
import datetime
import os
from shutil import copyfile
from tensorboard_logger import configure, log_value
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


base_train_idx = np.load('meta_metamnist/base_train.npy')
base_validation_idx = np.load('meta_metamnist/base_valid.npy')
meta_train_idx = np.load('meta_metamnist/meta_train.npy')
meta_validation_idx = np.load('meta_metamnist/meta_valid.npy')


CUDA_DEVICE = 'cuda:1'
torch.cuda.set_device(1)

#CUDA_DEVICE = 'cuda:'+sys.argv[1]
#torch.cuda.set_device(int(sys.argv[1]))
fc1_output = []
output_folder = ""

#saved_base_model = 'rotatedmnist25/rotatedmnist25percenttrain_1_epoch_75.6percentacc.torch'
#saved_meta_model = 'rotatedmnist25/meta_model25_best_diff_adj_geo_acc_valid_epoch_11.pth'

#make a folder to house record of training
name_and_time = str(sys.argv[0]) + str(datetime.datetime.now())
results_folder = '/media/chad/nara/meta/mnist/meta_full_pipeline/results/'+name_and_time+'/'
os.mkdir(results_folder)
copyfile(sys.argv[0], results_folder + sys.argv[0])
accuracies_file_name = results_folder+sys.argv[0]+'_accuracies_record.txt'
accuracies_file = open(accuracies_file_name, "w+")

configure(results_folder+sys.argv[0]+'tblogfile')

class IntermediateLayersInMemoryDataset(Dataset):
    def __init__(self, error_files, correct_files, percentage = 1.0, one_class = False, transform=None):


        num_errors = 0
        num_correct = 0
        self.correct_running_count = 0
        self.incorrect_running_count = 0
        self.X_data = []
        correct_data = []
        error_data = []
        self.num_filters = 0
        self.isConv = False
        self.dim_size = 0

        if not one_class:
            for i in range(len(correct_files)):
                self.X_data.append([])
            assert len(correct_files) == len(error_files)

        elif one_class == 'correct':
            for i in range(len(correct_files)):
                self.X_data.append([])

        elif one_class == 'error':
            for i in range(len(error_files)):
                self.X_data.append([])
        else:
            raise Exception('one_class must be False, correct, or error')

#       for i in range(len(correct_files)):
#           self.X_data.append(torch.load(correct_files[i]))
        if len(correct_files) > 0:
            loaded = torch.load(correct_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_correct = np.random.choice(all_indices, size=int(percentage * len(loaded)), replace = False)

        for layer_index in range(len(correct_files)):
            loaded = torch.load(correct_files[layer_index])
            for item_idx in selected_indices_correct:
                self.X_data[layer_index].append(loaded[item_idx])

        num_correct = len(self.X_data[0])
        
        if len(error_files) > 0:

            loaded = torch.load(error_files[0])
            all_indices = list(range(len(loaded)))
            selected_indices_incorrect = np.random.choice(all_indices, size=int(percentage * len(loaded)), replace = False)

        for layer_index in range(len(error_files)):
            loaded = torch.load(error_files[layer_index])
            for item_idx in selected_indices_incorrect:
                self.X_data[layer_index].append(loaded[item_idx])

        num_errors = len(self.X_data[0]) - num_correct

        if len(self.X_data[0][0].shape) > 2:
            self.isConv = True
            self.num_filters = self.X_data[0][0].shape[0]
        else:
            self.dim_size = self.X_data[0][0].shape[0]


        self.correct_len = num_correct
        self.error_len = num_errors
        self.total_len = self.error_len + self.correct_len

        self.y_data = np.zeros((self.total_len))
        self.y_data[0:self.correct_len] = 1

        print('number of errors')
        print(num_errors)
        print('size of dataset')
        print(num_errors+num_correct)
        print('percentage of errors')
        print(num_errors/(0.0+num_errors+num_correct))


    def __len__(self):
        return self.total_len


    def __getitem__(self, idx):
        Xs_to_return = []

        for layer in range(len(self.X_data)):

            Xs_to_return.append(self.X_data[layer][idx].float().to(CUDA_DEVICE))

        #Xs_to_return = (Xs_to_return[0], Xs_to_return[1], Xs_to_return[2], Xs_to_return[3])
        Xs_to_return = (Xs_to_return[0])

        if self.y_data[idx] == 1:
            self.correct_running_count += 1
        else:
            self.incorrect_running_count += 1

        return (Xs_to_return, torch.tensor(self.y_data[idx]).long())

    def get_correct_len(self):
        return self.correct_len

    def get_error_len(self):
        return self.error_len

    def get_y_data(self):
        return self.y_data

    def get_num_layers(self): 
        return self.X_data.shape()[0]

    def isConvolutional(self):
        return self.isConv

    def get_num_filters(self):
        return self.num_filters

    def get_size(self):
        return self.dim_size

class BaseMNIST(nn.Module):
    def __init__(self, fc1_size = 4096):
        super(BaseMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class FCMetaNet(nn.Module):

    def __init__(self, first_layer_size):
        super(FCMetaNet, self).__init__()
        self.fc1 = nn.Linear(first_layer_size, 4096) #input dimension both output and fc layer output
        self.bn1 = nn.BatchNorm1d(4096)

        self.fc2 = nn.Linear(4096, 7000)
        self.bn2 = nn.BatchNorm1d(7000)
        self.fc3 = nn.Linear(7000, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.fc4 = nn.Linear(2048, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc6 = nn.Linear(512, 64)
        self.bn6 = nn.BatchNorm1d(64)

        self.fc7 = nn.Linear(64, 2)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.bn1(x)

        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)

        x = F.dropout(x, training=self.training)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)

        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = F.dropout(x, training=self.training)

        x = F.relu(self.fc5(x))
        x = self.bn5(x)
        #x = F.dropout(x, training=self.training)

        x = F.relu(self.fc6(x))
        x = self.bn6(x)

        x = self.fc7(x)

        return F.log_softmax(x, dim = 1)



"""
def test_meta_model(args, base_model, meta_model, train_idx, device, train_dataset, base_optimizer, meta_optimizer, epoch):
    base_model.eval()
    meta_model.eval()
    base_model.fc1.register_forward_hook(fc1_hook)
    num_meta_majority_correct = 0
    num_base_majority_correct = 0
    num_no_base_correct = 0
    batch_id_progress = 0
    num_no_base_correct_meta_correct = 0
    num_no_base_correct_meta_incorrect = 0
    num_a_base_correct_meta_says_none = 0
    number_of_metas_when_false_positive = 0
    num_a_base_correct = 0
    for meta_mini_batch in range(int(len(train_idx)/args.meta_batch_size)):

        mini_batch = []
        for original_x in range(args.meta_batch_size):
            this_index = train_idx[batch_id_progress]
            batch_id_progress += 1
            for i in range(args.rotate_batch_size):
                mini_batch.append(train_dataset.__getitem__(this_index))

        collated = default_collate(mini_batch)
        data, target = collated[0].to(device), collated[1].to(device)
        #optimizer.zero_grad()
        output = base_model(data)
        loss = F.nll_loss(output, target)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        #correct += pred.eq(target.view_as(pred)).sum().item()


        meta_output = meta_model(fc1_output[0])
        meta_prob = torch.exp(meta_output)
        meta_prob_np = meta_prob.detach().cpu().numpy()
        pred_np = pred.cpu().numpy()
        where_meta_thinks_correct = np.where(meta_prob_np[:,1] > 0.5)
        pred_np = np.reshape(pred_np, (20))
        meta_filtered_pred = pred_np[where_meta_thinks_correct]
        meta_count = np.bincount(meta_filtered_pred)
        
        where_base_correct = np.where(pred_np == target[0])

        if len(where_meta_thinks_correct[0]) != 0:
            meta_prediction = np.argmax(meta_count)
            if meta_prediction == target[0]:
                num_meta_majority_correct += 1
            if len(where_base_correct[0]) == 0:
                num_no_base_correct_meta_incorrect += 1
                #number_of_metas_when_false_positive +=
        else:
            if len(where_base_correct[0]) == 0:
                num_no_base_correct_meta_correct += 1
            else:
                num_a_base_correct_meta_says_none += 1
        
        if len(where_base_correct[0]) == 0:
            num_no_base_correct += 1
        else:
            num_a_base_correct += 1

        base_count = np.bincount(pred_np)
        base_pred = np.argmax(base_count)

        if base_pred == target[0]:
            num_base_majority_correct += 1
        fc1_output.clear()
    
    import IPython
    IPython.embed()
"""


def test_meta_model(model,device,error_test_loader, correct_test_loader,optimizer,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    accuracies = []
    with torch.no_grad():

        test_loss = 0
        correct = 0
        correct_acc = 0
        error_acc = 0

        for batch_idx, (data, target) in enumerate(correct_test_loader):
            
            #only need to put tensors in position 0 onto device (?)
            data[0] = data[0].to(device)

            target = target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


        test_loss /= len(correct_test_loader.dataset)
        print('\nTest set: Average loss on correctly classified examples: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(correct_test_loader.dataset),
            100. * correct / len(correct_test_loader.dataset)))
        correct_acc = 100. * correct / len(correct_test_loader.dataset)

        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(error_test_loader):
            
            #only need to put tensors in position 0 onto device (?)
            data[0] = data[0].to(device)

            target = target.to(device)
            output = model(data)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


        test_loss /= len(error_test_loader.dataset)
        print('\nTest set: Average loss on incorrectly classified examples: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(error_test_loader.dataset),
            100. * correct / len(error_test_loader.dataset)))
        error_acc = 100. * correct / len(error_test_loader.dataset)
    return (correct_acc, error_acc)


class Layer(object):
    def __init__(self, name, layer_pos, model_layer):
        self.name = name
        self.layer_pos = layer_pos
        self.hook_list = []
        self.model_layer = model_layer

    def hook_fn(self, model, input, output):
        if self.layer_pos == 'input':
            self.hook_list.append(input[0].cpu())
        elif self.layer_pos == 'output':
            self.hook_list.append(output.data.cpu())
        else:
            print("ERROR")

hooked_layers = [Layer('conv1', 'output', lambda model: model.conv1),
                 Layer('conv2', 'output', lambda model: model.conv2),
                 Layer('fc1_input', 'input', lambda model: model.fc1),
                 Layer('fc1_output', 'output', lambda model: model.fc1)]

conv1_output = []
conv2_output = []
fc1_output = []

def conv1_hook(self, input, output):
    conv1_output.append(output.data)

def conv2_hook(self, input, output):
    conv2_output.append(input[0])

def fc1_hook(self, input, output):
    fc1_output.append(output.data)



def train_base_model(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_base_model(args, model, device, test_loader, epoch, mode, set_size):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, set_size,
        100. * correct / set_size))
    accuracy = 100.0 * correct / set_size
    accuracies_file.write(mode+ " " + str(epoch) + " " + str(accuracy) + '\n')
    return accuracy

def train_meta(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        #only need to put tensors in position 0 onto device (?)

        data[0] = data[0].to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    print('percentage correct:')
    correct_percent  = 100.0*correct/len(train_loader.dataset)
    print(correct_percent)
    return correct_percent


def makeIntermediates(args, model, device, test_loader, mode):
    model.eval()
    test_losses = np.zeros(11)
    correct_array = np.zeros(11)
    mod_test_loss = 0
    mod_correct = 0
    list_of_output_lists = []

    error_output_lists = []
    correct_output_lists = []

    list_of_correct_conv1_outputs = []
    list_of_incorrect_conv1_outputs = []
    list_of_correct_conv2_outputs = []
    list_of_incorrect_conv2_outputs = []
    list_of_correct_fc1_outputs = []
    list_of_incorrect_fc1_outputs = []

    model.conv1.register_forward_hook(conv1_hook)
    model.fc1.register_forward_hook(conv2_hook)
    model.fc1.register_forward_hook(fc1_hook)




    with torch.no_grad():
        outputs = []
        correct_outputs = []
        incorrect_outputs = []
        correct_conv1_outputs = []
        incorrect_conv1_outputs = []
        correct_conv2_outputs = []
        incorrect_conv2_outputs = []
        correct_fc1_outputs = []
        incorrect_fc1_outputs = []



        test_loss = 0
        correct = 0

        for data, target in test_loader:

            data = data.float()

            data, target = data.to(device), target.to(device)

            output = model(data)

            #output=full_output[0]
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            for i in output:
                outputs.append(torch.exp(i))
            where_wrong = np.where(np.not_equal(pred.cpu(),target.cpu().view_as(pred)))[0]
            incorrect_outs = output[where_wrong]

            incorrect_conv1_outs = conv1_output[0][where_wrong]

            incorrect_conv2_outs = conv2_output[0][where_wrong]

            incorrect_fc1_outs = fc1_output[0][where_wrong]



            where_correct = np.where(np.equal(pred.cpu(),target.cpu().view_as(pred)))[0]
            correct_outs = output[where_correct]

            correct_conv1_outs = conv1_output[0][where_correct]
            correct_conv2_outs = conv2_output[0][where_correct]
            correct_fc1_outs = fc1_output[0][where_correct]


            for i in incorrect_outs:
                incorrect_outputs.append(torch.exp(i))

            for i in incorrect_conv1_outs:
                incorrect_conv1_outputs.append(i)

            for i in incorrect_conv2_outs:
                incorrect_conv2_outputs.append(i)

            for i in incorrect_fc1_outs:
                incorrect_fc1_outputs.append(i)



            for i in correct_outs:
                correct_outputs.append(torch.exp(i))

            for i in correct_conv1_outs:
                correct_conv1_outputs.append(i)

            for i in correct_conv2_outs:
                correct_conv2_outputs.append(i)

            for i in correct_fc1_outs:
                correct_fc1_outputs.append(i)


            correct += pred.eq(target.view_as(pred)).sum().item()



            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, 10000,
                100. * correct / 10000.00))
            """
            list_of_output_lists.append(outputs)
            error_output_lists.append(incorrect_outputs)
            correct_output_lists.append(correct_outputs)
            list_of_correct_conv1_outputs.append(correct_conv1_outputs)
            list_of_incorrect_conv1_outputs.append(incorrect_conv1_outputs)
            list_of_correct_conv2_outputs.append(correct_conv2_outputs)
            list_of_incorrect_conv2_outputs.append(incorrect_conv2_outputs)
            list_of_correct_fc1_outputs.append(correct_fc1_outputs)
            list_of_incorrect_fc1_outputs.append(incorrect_fc1_outputs)
            """
            conv1_output.clear()
            conv2_output.clear()
            fc1_output.clear()

        #torch.save(correct_fc1_outputs, 'fc1_inter_correct_meta_val_15percent_data.torch')
        #torch.save(incorrect_fc1_outputs, 'fc1_inter_incorrect_meta_val_15percent_data.torch')
        
        torch.save(correct_fc1_outputs, results_folder + 'fc1_inter_correct_meta_' + mode +'.torch')
        torch.save(incorrect_fc1_outputs, results_folder + 'fc1_inter_incorrect_meta_' + mode +'.torch')
        #torch.save(correct_conv2_outputs, results_folder + 'conv2_inter_correct_meta_' + mode +'.torch')
        #torch.save(incorrect_conv2_outputs, results_folder + 'conv2_inter_incorrect_meta_' + mode +'.torch')
        #torch.save(correct_conv1_outputs, results_folder + 'conv1_inter_correct_meta_' + mode +'.torch')
        #torch.save(incorrect_conv1_outputs, results_folder + 'conv1_inter_incorrect_meta_' + mode +'.torch')
        #torch.save(correct_outputs, results_folder + 'correct_outputs_meta_' + mode +'.torch')
        #torch.save(incorrect_outputs, results_folder + 'incorrect_outputs_meta_' + mode +'.torch')
        """
        torch.save(correct_fc1_outputs, 'rotatedmnist25/fc1_inter_correct_meta_val_25percent_data.torch')
        torch.save(incorrect_fc1_outputs, 'rotatedmnist25/fc1_inter_incorrect_meta_val_25percent_data.torch')
        torch.save(correct_conv2_outputs, 'rotatedmnist25/conv2_inter_correct_meta_val_25percent_data.torch')
        torch.save(incorrect_conv2_outputs, 'rotatedmnist25/conv2_inter_incorrect_meta_val_25percent_data.torch')
        torch.save(correct_conv1_outputs, 'rotatedmnist25/conv1_inter_correct_meta_val_25percent_data.torch')
        torch.save(incorrect_conv1_outputs, 'rotatedmnist25/conv1_inter_incorrect_meta_val_25percent_data.torch')
        torch.save(correct_outputs, 'rotatedmnist25/correct_outputs_meta_val_25percent_data.torch')
        torch.save(incorrect_outputs, 'rotatedmnist25/incorrect_outputs_meta_val_25percent_data.torch')
        """
        #import IPython
        #IPython.embed()

def make_and_train_base_model(args, train_dataset, device):




    train_sampler = SubsetRandomSampler(base_train_idx)

    validation_sampler = SubsetRandomSampler(base_validation_idx)




    base_train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    base_validation_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=args.batch_size, sampler=validation_sampler)

    """
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([transforms.RandomRotation(30),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    """


    #make a base model
    base_model = BaseMNIST(fc1_size=args.base_model_fc1_size).to(device)
    if args.load_base_model_from_saved_state:
        base_saved_state = torch.load(args.load_base_model_from_saved_state)
        base_model.load_state_dict(base_saved_state)



    #choose optimizers for meta and base networks
    base_optimizer = optim.SGD(base_model.parameters(), lr=args.lr, momentum=args.momentum)

    #train the base model 
    for epoch in range(args.base_train_num_epochs):
        train_base_model(args, base_model, device, base_train_loader, base_optimizer, epoch)
        test_base_model(args, base_model, device, base_train_loader, epoch, "train", len(base_train_idx))
        test_base_model(args, base_model, device, base_validation_loader, epoch, "valid", len(base_validation_idx))

    torch.save(base_model.state_dict(), results_folder + "base_model.torch")

    return base_model

def make_and_train_meta_model(args, device, train_set_percentage):
    

    outputs_correct_train_file = results_folder + 'correct_outputs_meta_train.torch'
    outputs_incorrect_train_file = results_folder + 'incorrect_outputs_meta_train.torch'
    conv2_correct_train_file = results_folder + 'conv2_inter_correct_meta_train.torch'
    conv2_correct_train_file = results_folder + 'conv2_inter_incorrect_meta_train.torch'
    """
    results_folder + 'conv1_inter_correct_meta_train.torch'
    results_folder + 'conv1_inter_incorrect_meta_train.torch'
    results_folder + 'correct_outputs_meta_train.torch'
    results_folder + 'incorrect_outputs_meta_train.torch'
    results_folder + 'conv1_inter_correct_meta_train.torch'
    results_folder + 'conv1_inter_incorrect_meta_train.torch'
    results_folder + 'correct_outputs_meta_train.torch'
    results_folder + 'incorrect_outputs_meta_train.torch'
    """

    fc1_correct_train_file = results_folder + 'fc1_inter_correct_meta_train.torch'
    fc1_incorrect_train_file = results_folder + 'fc1_inter_incorrect_meta_train.torch'

    fc1_correct_valid_file = results_folder + 'fc1_inter_correct_meta_valid.torch'
    fc1_incorrect_valid_file = results_folder + 'fc1_inter_incorrect_meta_valid.torch'
    conv2_correct_valid_file = results_folder + 'conv2_inter_correct_meta_valid.torch'
    conv2_correct_valid_file = results_folder + 'conv2_inter_incorrect_meta_valid.torch'

    fc1_correct_test_file = results_folder + 'fc1_inter_correct_meta_test.torch'
    fc1_incorrect_test_file = results_folder + 'fc1_inter_incorrect_meta_test.torch'
    conv2_correct_test_file = results_folder + 'conv2_inter_correct_meta_test.torch'
    conv2_correct_test_file = results_folder + 'conv2_inter_incorrect_meta_test.torch'

    outputs_correct_valid_file = results_folder + 'correct_outputs_meta_valid.torch'
    outputs_incorrect_valid_file = results_folder + 'incorrect_outputs_meta_valid.torch'

    outputs_correct_test_file = results_folder + 'correct_outputs_meta_test.torch'
    outputs_incorrect_test_file = results_folder + 'incorrect_outputs_meta_test.torch'

    train_correct_files = [fc1_correct_train_file]
    train_error_files = [fc1_incorrect_train_file]

    valid_correct_files = [fc1_correct_valid_file]
    valid_error_files = [fc1_incorrect_valid_file]

    test_correct_files = [fc1_correct_test_file]
    test_error_files = [fc1_incorrect_test_file]

    """
    train_error_files = (outputs_error_train_file, pre_bn_layer_error_train_file, last_conv_error_train_file, pen_conv_error_train_file)
    train_correct_files = (outputs_correct_train_file, pre_bn_layer_correct_train_file, last_conv_correct_train_file, pen_conv_correct_train_file)

    valid_correct_files = (outputs_correct_valid_file, pre_bn_layer_correct_valid_file, last_conv_correct_valid_file, pen_conv_correct_valid_file)
    valid_error_files = (outputs_error_valid_file, pre_bn_layer_error_valid_file, last_conv_error_valid_file, pen_conv_error_valid_file)
    """

    empty_files = ()

    train_dataset = IntermediateLayersInMemoryDataset(train_error_files, train_correct_files, percentage = train_set_percentage)
    valid_error_dataset = IntermediateLayersInMemoryDataset(valid_error_files, empty_files, one_class='error' )
    valid_correct_dataset = IntermediateLayersInMemoryDataset(empty_files, valid_correct_files, one_class='correct' )

    test_error_dataset = IntermediateLayersInMemoryDataset(test_error_files, empty_files, one_class='error' )
    test_correct_dataset = IntermediateLayersInMemoryDataset(empty_files, test_correct_files, one_class='correct' )

    #make weights for balancing training samples


    correct_count = train_dataset.get_correct_len()
    error_count = train_dataset.get_error_len()
    total_count = correct_count + error_count

    y_vals = train_dataset.get_y_data()

    correct_weight = float(total_count)/correct_count
    error_weight = float(total_count)/error_count

    weights = np.zeros((total_count))

    for i in range(len(y_vals)):
        if y_vals[i] == 0:
            weights[i] = error_weight
        else:
            weights[i] = correct_weight


    
    error_range = list(range(correct_count,total_count))
    correct_range = list(range(correct_count))
    total_range = list(range(total_count))

    
    train_weights = torch.DoubleTensor(weights)
    train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,total_count)

    #import IPython
    #IPython.embed()
    
    #length_of_dataset = len(train_dataset)
    #indices = list(range(length_of_dataset))
    #train_split = int(length_of_dataset*.8)

    #train_indices = np.random.choice(indices, size = train_split, replace = False)
    #validation_indices = list(set(indices)- set(train_indices))

    #train_sampler = SubsetRandomSampler(train_indices)
    #validation_sampler = SubsetRandomSampler(validation_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,  shuffle = False,
        sampler = train_weighted_sampler)
    error_validation_loader = torch.utils.data.DataLoader(valid_error_dataset, batch_size = 32,  shuffle = False)
    correct_validation_loader = torch.utils.data.DataLoader(valid_correct_dataset, batch_size = 32,  shuffle = False)
    error_test_loader = torch.utils.data.DataLoader(test_error_dataset, batch_size = 32,  shuffle = False)
    correct_test_loader = torch.utils.data.DataLoader(test_correct_dataset, batch_size = 32,  shuffle = False)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if train_dataset.isConvolutional():
        num_filters_in_input = train_dataset.get_num_filters()
        model = ConvNet(num_filters_in_input).to(device)
    else: 
        size_of_first_layer = train_dataset.get_size()
        meta_model=FCMetaNet(size_of_first_layer).to(device)
    
    #make a meta model
    if args.load_meta_model_from_saved_state:
        meta_saved_state = torch.load(args.load_meta_model_from_saved_state)
        meta_model.load_state_dict(meta_saved_state)


    meta_optimizer = optim.Adam(meta_model.parameters(),lr=.00001)
    scheduler = ReduceLROnPlateau(meta_optimizer, 'max', verbose=True)

    best_error_valid_value = 0
    best_correct_valid_value = 0
    best_total_valid_value = 0
    best_total_geo_valid_value = 0
    best_total_diff_adj_geo_acc = 0
    old_diff_adj_geo_acc_file_name_created = False
    old_correct_acc_file_name_created = False
    best_train_acc = 0
    best_total_diff_adj_geo_acc_correct = 0
    best_total_diff_adj_geo_acc_error = 0

    for epoch in range (1,args.meta_train_num_epochs+1):

        train_acc = train_meta(meta_model, device,train_loader,meta_optimizer,epoch)

        correct_acc, error_acc = test_meta_model(meta_model, device,error_validation_loader, correct_validation_loader, meta_optimizer,epoch)
        total_acc = error_acc + correct_acc
        total_geo_acc = np.sqrt(error_acc * correct_acc)
        total_diff_adj_geo_acc = total_geo_acc - np.abs(error_acc-correct_acc)


        #accuracies_file = open(accuracies_file_name, "a+")
        accuracies_file.write(str(epoch) + " " + str(train_acc) + " " + " " + str(correct_acc) + " " + str(error_acc)+ " " + str(total_acc) + " " +  str(total_geo_acc) + " " + str(total_diff_adj_geo_acc)+"\n")
        #accuracies_file.close()

        if train_acc > best_train_acc:
            best_train_acc = train_acc

        if total_acc > best_total_valid_value:
            best_total_valid_value = total_acc
            if epoch > 1:
                os.remove(old_total_acc_file_name)

            old_total_acc_file_name = results_folder+'_best_total_acc_valid_epoch_'+str(epoch)+'.pth'
            torch.save(meta_model.state_dict(), old_total_acc_file_name)


        if total_diff_adj_geo_acc > best_total_diff_adj_geo_acc:
            best_total_diff_adj_geo_acc = total_diff_adj_geo_acc
            best_total_diff_adj_geo_acc_correct = correct_acc
            best_total_diff_adj_geo_acc_error = error_acc

            if epoch > 1 and old_diff_adj_geo_acc_file_name_created == True:
                os.remove(old_diff_adj_geo_acc_file_name)

            old_diff_adj_geo_acc_file_name_created = True
            old_diff_adj_geo_acc_file_name = results_folder+'_best_diff_adj_geo_acc_valid_epoch_'+str(epoch)+'.pth'
            torch.save(meta_model.state_dict(), old_diff_adj_geo_acc_file_name)

        
        print("Geo dif adj valid mean acc: " + str(total_diff_adj_geo_acc))

        scheduler.step(total_diff_adj_geo_acc)


    if old_diff_adj_geo_acc_file_name_created == True:

        meta_saved_state = torch.load(old_diff_adj_geo_acc_file_name)
        meta_model.load_state_dict(meta_saved_state)

    #return meta_model, best_total_diff_adj_geo_acc

    test_correct_acc, test_error_acc = test_meta_model(meta_model, device,error_test_loader, correct_test_loader, meta_optimizer,epoch)

    return best_total_diff_adj_geo_acc_correct, best_total_diff_adj_geo_acc_error, test_correct_acc, test_error_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='meta MNIST pipeline')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=10027, metavar='S',
                        help='random seed (default: 10027)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--meta_batch_size', type=int, default=1, metavar='MBS',
                        help='size of batches to the meta classifier')
    parser.add_argument('--base_model_fc1_size', type=int, default=128, metavar='fc1size',
                        help='size of batches to the meta classifier')
    parser.add_argument('--base_train_num_epochs', type=int, default=1, metavar='basetrainepochs',
                        help='size of batches to the meta classifier')
    parser.add_argument('--meta_train_num_epochs', type=int, default=50, metavar='metatrainepochs',
                        help='size of batches to the meta classifier')
    parser.add_argument('--load_base_model_from_saved_state', default="")
    parser.add_argument('--load_meta_model_from_saved_state', default="")
    

    meta_sizes_accuracies_file_name = results_folder+sys.argv[0]+'meta_acc_by_meta_size_accuracies_record.txt'
    meta_sizes_accuracies_file = open(meta_sizes_accuracies_file_name, "w+")


    args = parser.parse_args()




    kwargs = {'num_workers': 1, 'pin_memory': True} 
    device = torch.device(CUDA_DEVICE)

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    test_dataset = datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    
    meta_test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=args.batch_size)


    """
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([transforms.RandomRotation(30),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    """

    for x in [40]:
        meta_sizes_accuracies_file.write('base network epochs trained:\n')
        meta_sizes_accuracies_file.write(str(x)+'\n')
        args.base_train_num_epochs = x

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        base_model = make_and_train_base_model(args, train_dataset, device)
        base_acc = test_base_model(args, base_model, device, meta_test_loader, x, "TEST", len(test_dataset))
        meta_sizes_accuracies_file.write("base model accuracy: " + str(base_acc)+'\n')

        meta_train_sampler = SubsetRandomSampler(meta_train_idx)

        meta_validation_sampler = SubsetRandomSampler(meta_validation_idx)


        meta_train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=args.batch_size, sampler=meta_train_sampler)
        meta_validation_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=args.batch_size, sampler=meta_validation_sampler)

        makeIntermediates(args, base_model, device, meta_train_loader, "train")
        makeIntermediates(args, base_model, device, meta_validation_loader, "valid")


        makeIntermediates(args, base_model, device, meta_test_loader, "test")


        #meta_model, best_diff_adj_geo_acc = make_and_train_meta_model(args, device)


        for i in [10]:
            print(i)
            p = i*.1
            correct, error, test_correct, test_error = make_and_train_meta_model(args, device, train_set_percentage=p)
            meta_sizes_accuracies_file.write(str(i) + " " + str(correct) + " " + str(error)+" " +str(test_correct)+" " +str(test_error)+'\n')

    accuracies_file.close()
    meta_sizes_accuracies_file.close()

if __name__ == '__main__':
    main()