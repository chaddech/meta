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
import IPython

#os.environ["CUDA_VISIBLE_DEVICES"]="2"
CUDA_DEVICE = 'cuda:'+sys.argv[3]
torch.cuda.set_device(int(sys.argv[3]))


#make a folder to house results
base_path = '/home/seungwookhan/cifar10_results/'
results_folder = base_path +str(sys.argv[0])+ sys.argv[1] + sys.argv[2] + str(datetime.datetime.now())+'/'
os.mkdir(results_folder)
copyfile(sys.argv[0], results_folder + sys.argv[0])
accuracies_file_name = results_folder+sys.argv[0]+sys.argv[1]+sys.argv[2]+'_accuracies_record.txt'
final_results_file_name = results_folder + 'final_best_results_record_vgg19_bn_cifar10.txt'
accuracies_file = open(accuracies_file_name, "w+")
accuracies_file.close()

configure(results_folder+sys.argv[0]+'tblogfile')

# ** CHANGE DIRECTORY INFORMATION
single_layer_correct_train_file = '/home/seungwookhan/cifar10_intermediates/' + sys.argv[1] + '/train/' + sys.argv[2] + '/correct/sorted_outputs.torch'
single_layer_incorrect_train_file = '/home/seungwookhan/cifar10_intermediates/' + sys.argv[1] + '/train/' + sys.argv[2] + '/incorrect/sorted_outputs.torch'

single_layer_correct_valid_file = '/home/seungwookhan/cifar10_intermediates/' + sys.argv[1] + '/valid/' + sys.argv[2] + '/correct/sorted_outputs.torch'
single_layer_incorrect_valid_file = '/home/seungwookhan/cifar10_intermediates/' + sys.argv[1] + '/valid/' + sys.argv[2] + '/incorrect/sorted_outputs.torch'



num_epochs = 75

class ImageNetInterMediateLayersInMemoryDataset(Dataset):
	def __init__(self, error_files, correct_files, one_class = False, transform=None):


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


		for layer_index in range(len(correct_files)):
			loaded = torch.load(correct_files[layer_index])
			for item_idx in range(len(loaded)):
				self.X_data[layer_index].append(loaded[item_idx])

		num_correct = len(self.X_data[0])


		for layer_index in range(len(error_files)):
			loaded = torch.load(error_files[layer_index])
			for item_idx in range(len(loaded)):
				self.X_data[layer_index].append(loaded[item_idx])

		num_errors = len(self.X_data[0]) - num_correct

		if len(self.X_data[0][0][0].shape) > 2:
			self.isConv = True
			self.num_filters = self.X_data[0][0][0].shape[0]
		else:
			self.dim_size = self.X_data[0][0][0].shape[0]

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
			Xs_to_return.append(self.X_data[layer][idx][0].float().to(CUDA_DEVICE))
		#Xs_to_return = (Xs_to_return[0], Xs_to_return[1], Xs_to_return[2], Xs_to_return[3])
		Xs_to_return = (Xs_to_return[0],self.X_data[layer][idx][1])

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
		return self.X_data.shape()[0][0]

	def isConvolutional(self):
		return self.isConv

	def get_num_filters(self):
		return self.num_filters

	def get_size(self):
		return self.dim_size





class Net(nn.Module):

	def __init__(self, first_layer_size):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(first_layer_size, 1024) #input dimension both output and fc layer output
		self.bn1 = nn.BatchNorm1d(1024)

		self.fc2 = nn.Linear(1024, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.fc3 = nn.Linear(1024, 1024)
		self.bn3 = nn.BatchNorm1d(1024)

		

		self.fc4 = nn.Linear(1024, 512)
		self.bn4 = nn.BatchNorm1d(512)

		self.fc5 = nn.Linear(512, 512)
		self.bn5 = nn.BatchNorm1d(512)

		self.fc6 = nn.Linear(512, 64)
		self.bn6 = nn.BatchNorm1d(64)

		self.fc7 = nn.Linear(64, 2)


	def forward(self, x):
		x=x[0]
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






class ConvNet(nn.Module):

	def __init__(self, num_initial_filters):
		super(ConvNet, self).__init__()
		print('Num initial filters: ' + str(num_initial_filters))
		self.conv1 = nn.Conv2d(num_initial_filters, 256, kernel_size=3)
		self.convbn1 = nn.BatchNorm2d(256)

		self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
		self.convbn2 = nn.BatchNorm2d(256)

		self.AdaptMaxPool = nn.AdaptiveMaxPool2d((12,12))
		self.conv3 = nn.Conv2d(256, 96, kernel_size=3)
		self.convbn3 = nn.BatchNorm2d(96)

		self.conv4 = nn.Conv2d(96, 96, kernel_size=3)
		self.convbn4 = nn.BatchNorm2d(96)

		self.conv5 = nn.Conv2d(96, 96, kernel_size=3)
		self.convbn5 = nn.BatchNorm2d(96)

		self.fc1 = nn.Linear(3456,2048) #input dimension both output and fc layer output
		self.bn1 = nn.BatchNorm1d(2048)

		self.fc2 = nn.Linear(2048, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.fc3 = nn.Linear(1024, 1024)
		self.bn3 = nn.BatchNorm1d(1024)

		

		self.fc4 = nn.Linear(1024, 512)
		self.bn4 = nn.BatchNorm1d(512)

		self.fc5 = nn.Linear(512, 512)
		self.bn5 = nn.BatchNorm1d(512)

		self.fc6 = nn.Linear(512, 64)
		self.bn6 = nn.BatchNorm1d(64)

		self.fc7 = nn.Linear(64, 2)

	def forward(self, Xs):


		convs = Xs
		convs = F.relu(self.conv1(convs))
		convs = self.convbn1(convs)

		#convs = F.dropout2d(convs, training = self.training)
		convs = F.relu(self.conv2(convs))
		convs = self.convbn2(convs)

		convs = F.dropout2d(convs, training = self.training)
		convs = self.AdaptMaxPool(convs)
		convs = F.relu(self.conv3(convs))
		convs = self.convbn3(convs)

		convs = F.relu(self.conv4(convs))
		convs = self.convbn4(convs)

		convs = F.relu(self.conv5(convs))
		convs = self.convbn5(convs)

		#convs = F.max_pool2d(convs, 2)
		#convs = F.dropout2d(convs, training = self.training)

		x = convs.view(-1, 3456)

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
		#x = F.dropout(x, training=self.training)

		x = F.relu(self.fc5(x))
		x = self.bn5(x)
		#x = F.dropout(x, training=self.training)

		x = F.relu(self.fc6(x))
		x = self.bn6(x)

		x = self.fc7(x)



		return F.log_softmax(x, dim = 1)



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

	train_dataset = ImageNetInterMediateLayersInMemoryDataset(train_error_files, train_correct_files)
	valid_error_dataset = ImageNetInterMediateLayersInMemoryDataset(valid_error_files, empty_files, one_class='error' )
	valid_correct_dataset = ImageNetInterMediateLayersInMemoryDataset(empty_files, valid_correct_files, one_class='correct' )


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
