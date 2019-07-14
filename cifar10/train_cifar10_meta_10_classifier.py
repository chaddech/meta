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
from cifar10_datasets import ImageNetTargetGroundInterMediateLayersInMemoryDataset, ImageNetTargetPredictInterMediateLayersInMemoryDataset
from meta_models import Net, Net10, ConvNet
import IPython

MODEL_NAME = sys.argv[1]
LAYER_NAME = sys.argv[2]
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
single_layer_correct_train_file = '/home/seungwookhan/cifar10_intermediates/' + MODEL_NAME + '/train/' + LAYER_NAME + '/correct/sorted_outputs.torch'
single_layer_incorrect_train_file = '/home/seungwookhan/cifar10_intermediates/' + MODEL_NAME + '/train/' + LAYER_NAME + '/incorrect/sorted_outputs.torch'

single_layer_correct_valid_file = '/home/seungwookhan/cifar10_intermediates/' + MODEL_NAME + '/valid/' + LAYER_NAME + '/correct/sorted_outputs.torch'
single_layer_incorrect_valid_file = '/home/seungwookhan/cifar10_intermediates/' + MODEL_NAME + '/valid/' + LAYER_NAME + '/incorrect/sorted_outputs.torch'

num_epochs = 50

def train(model, device, train_loader, optimizer, epoch):
	model.train()
	correct = 0

	for batch_idx, (data, target) in enumerate(train_loader):
		#only need to put tensors in position 0 onto device (?)
		#CHECK DATA 
		#import IPython
		#IPython.embed()

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





def test(model,device,error_test_loader, correct_test_loader,optimizer,epoch):
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

def main():

	use_cuda = torch.cuda.is_available()

	torch.manual_seed(33)
	np.random.seed(10024)

	device = torch.device(CUDA_DEVICE if use_cuda else "cpu")


	train_error_files = [single_layer_incorrect_train_file]
	train_correct_files = [single_layer_correct_train_file]

	valid_correct_files = [single_layer_correct_valid_file]
	valid_error_files = [single_layer_incorrect_valid_file]
	

	empty_files = ()

	train_dataset = ImageNetTargetGroundInterMediateLayersInMemoryDataset(train_error_files, train_correct_files, layer_name=LAYER_NAME)
	valid_error_dataset = ImageNetTargetGroundInterMediateLayersInMemoryDataset(valid_error_files, empty_files, one_class='error', layer_name=LAYER_NAME)
	valid_correct_dataset = ImageNetTargetGroundInterMediateLayersInMemoryDataset(empty_files, valid_correct_files, one_class='correct', layer_name=LAYER_NAME)


	#make weights for balancing training samples
	correct_count = train_dataset.get_correct_len()
	error_count = train_dataset.get_error_len()
	total_count = correct_count + error_count

	x_vals = train_dataset.get_x_data()

	correct_weight = float(total_count)/correct_count
	error_weight = float(total_count)/error_count

	weights = np.zeros((total_count))

	for i in range(len(x_vals)):
		if x_vals[i][2] != x_vals[i][3]:
			weights[i] = error_weight
		else:
			weights[i] = correct_weight

	
	error_range = list(range(correct_count,total_count))
	correct_range = list(range(correct_count))
	total_range = list(range(total_count))

	
	train_weights = torch.DoubleTensor(weights)
	train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights,total_count)


	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,  shuffle = False,
		sampler = train_weighted_sampler)
	error_validation_loader = torch.utils.data.DataLoader(valid_error_dataset, batch_size = 124,  shuffle = False)
	correct_validation_loader = torch.utils.data.DataLoader(valid_correct_dataset, batch_size = 124,  shuffle = False)

	if train_dataset.isConvolutional():
		num_filters_in_input = train_dataset.get_num_filters()
		model = ConvNet(num_filters_in_input).to(device)
	else: 
		size_of_first_layer = train_dataset.get_size()
		model=Net(size_of_first_layer).to(device)


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

	for epoch in range (1,num_epochs+1):

		train_acc = train(model, device,train_loader,optimizer,epoch)

		correct_acc, error_acc = test(model, device,error_validation_loader, correct_validation_loader, optimizer,epoch)
		total_acc = error_acc + correct_acc
		total_geo_acc = np.sqrt(error_acc * correct_acc)
		total_diff_adj_geo_acc = total_geo_acc - np.abs(error_acc-correct_acc)


		accuracies_file = open(accuracies_file_name, "a+")
		accuracies_file.write(str(epoch) + " " + str(train_acc) + " " + " " + str(correct_acc) + " " + str(error_acc)+ " " + str(total_acc) + " " +  str(total_geo_acc) + " " + str(total_diff_adj_geo_acc)+"\n")
		accuracies_file.close()

		if train_acc > best_train_acc:
			best_train_acc = train_acc

		if correct_acc > best_correct_valid_value:
			best_correct_valid_value = correct_acc
			if epoch > 1 and old_correct_acc_file_name_created == True:
				os.remove(old_correct_acc_file_name)

			old_correct_acc_file_name = results_folder+'_best_correct_acc_valid_epoch_'+str(epoch)+'.pth'
			torch.save(model.state_dict(), old_correct_acc_file_name)
			old_correct_acc_file_name_created = True

		if error_acc > best_error_valid_value:
			best_error_valid_value = error_acc
			if epoch > 1:
				os.remove(old_error_acc_file_name)

			old_error_acc_file_name = results_folder+'_best_error_acc_valid_epoch_'+str(epoch)+'.pth'
			torch.save(model.state_dict(), old_error_acc_file_name)

		if total_acc > best_total_valid_value:
			best_total_valid_value = total_acc
			if epoch > 1:
				os.remove(old_total_acc_file_name)

			old_total_acc_file_name = results_folder+'_best_total_acc_valid_epoch_'+str(epoch)+'.pth'
			torch.save(model.state_dict(), old_total_acc_file_name)

		if total_geo_acc > best_total_geo_valid_value:
			best_total_geo_valid_value = total_geo_acc
			if epoch > 1:
				os.remove(old_geo_acc_file_name)

			old_geo_acc_file_name = results_folder+'_best_geo_acc_valid_epoch_'+str(epoch)+'.pth'
			torch.save(model.state_dict(), old_geo_acc_file_name)

		if total_diff_adj_geo_acc > best_total_diff_adj_geo_acc:
			best_total_diff_adj_geo_acc = total_diff_adj_geo_acc
			#if epoch > 1 and old_diff_adj_geo_acc_file_name_created == True:
			#	os.remove(old_diff_adj_geo_acc_file_name)

			old_diff_adj_geo_acc_file_name_created = True
			old_diff_adj_geo_acc_file_name = results_folder+'_best_diff_adj_geo_acc_valid_epoch_'+str(epoch)+'.pth'
			torch.save(model.state_dict(), old_diff_adj_geo_acc_file_name)

		
		print("Geo valid mean acc: " + str(total_geo_acc))
		print("Geo dif adj valid mean acc: " + str(total_diff_adj_geo_acc))

		scheduler.step(total_diff_adj_geo_acc)

	final_results_file = open(final_results_file_name, "a+")
	final_results_file.write(accuracies_file_name +" " + sys.argv[2] + "\n")
	final_results_file.write("final" + str(num_epochs) + " " + str(best_train_acc) +  " " + str(best_correct_valid_value) + " " + str(best_error_valid_value)+ " " + str(best_total_valid_value/2.0) + " " + str(best_total_geo_valid_value) + " " + str(best_total_diff_adj_geo_acc)+"\n\n")



if __name__ == '__main__':
	main()
