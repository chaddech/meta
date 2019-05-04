import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from shutil import copyfile
import datetime
from tensorboard_logger import configure, log_value
from train_cifar10_meta_classifier import ImageNetInterMediateLayersInMemoryDataset, Net, test
import IPython

MODEL_NAME = sys.argv[1]
LAYER_NAME = sys.argv[2]
MODEL_SAVED_STATE_PATH = sys.argv[4]
CUDA = sys.argv[3]

#os.environ["CUDA_VISIBLE_DEVICES"]="2"
CUDA_DEVICE = 'cuda:'+sys.argv[3]
torch.cuda.set_device(int(sys.argv[3]))


#make a folder to house results
base_path = '/home/seungwookhan/cifar10_results/'
results_folder = base_path +str(sys.argv[0])+ MODEL_NAME + LAYER_NAME + str(datetime.datetime.now())+'/'
os.mkdir(results_folder)
copyfile(sys.argv[0], results_folder + sys.argv[0])
accuracies_file_name = results_folder+sys.argv[0]+MODEL_NAME+LAYER_NAME+'_accuracies_record.txt'
final_results_file_name = results_folder + 'final_best_results_record_vgg19_bn_cifar10.txt'
accuracies_file = open(accuracies_file_name, "w+")
accuracies_file.close()

configure(results_folder+sys.argv[0]+'tblogfile')

# ** CHANGE DIRECTORY INFORMATION
single_layer_correct_test_file = '/home/seungwookhan/cifar10_intermediates/' + MODEL_NAME + '/test/' + LAYER_NAME + '/correct/sorted_outputs.torch'
single_layer_incorrect_test_file = '/home/seungwookhan/cifar10_intermediates/' + MODEL_NAME + '/test/' + LAYER_NAME + '/incorrect/sorted_outputs.torch'

def main():
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(33)
    np.random.seed(10024)

    device = torch.device(CUDA_DEVICE if use_cuda else "cpu")

    test_error_files = [single_layer_incorrect_test_file]
    test_correct_files = [single_layer_correct_test_file]

    empty_files = ()

    test_error_dataset = ImageNetInterMediateLayersInMemoryDataset(test_error_files, empty_files, one_class='error' )
    test_correct_dataset = ImageNetInterMediateLayersInMemoryDataset(empty_files, test_correct_files, one_class='correct' )

    error_test_loader = torch.utils.data.DataLoader(test_error_dataset, batch_size = 124,  shuffle = False)
    correct_test_loader = torch.utils.data.DataLoader(test_correct_dataset, batch_size = 124,  shuffle = False)

    size_of_first_layer = test_correct_dataset.get_size()
    model=Net(size_of_first_layer).to(device)

    model_saved_state = torch.load(MODEL_SAVED_STATE_PATH)
    model.load_state_dict(model_saved_state)

    optimizer = optim.Adam(model.parameters(),lr=.00001)
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True)
    best_error_valid_value = 0
    best_correct_valid_value = 0
    best_total_valid_value = 0
    best_total_geo_valid_value = 0
    best_total_diff_adj_geo_acc = 0
    old_diff_adj_geo_acc_file_name_created = False
    old_correct_acc_file_name_created = False
    best_train_acc = 0

    correct_acc, error_acc = test(model, device, error_test_loader, correct_test_loader, optimizer, 0)
    
    total_acc = error_acc + correct_acc
    total_geo_acc = np.sqrt(error_acc * correct_acc)
    total_diff_adj_geo_acc = total_geo_acc - np.abs(error_acc-correct_acc)

    accuracies_file = open(accuracies_file_name, "a+")
    accuracies_file.write(str(epoch) + " " + str(correct_acc) + " " + str(error_acc)+ " " + str(total_acc) + " " +  str(total_geo_acc) + " " + str(total_diff_adj_geo_acc)+"\n")
    accuracies_file.close()


if __name__ == '__main__':
    main()
