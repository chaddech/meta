import torch
import torch.nn as nn
import torch.nn.functional as F

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

class Net10(nn.Module):

	def __init__(self, first_layer_size):
		super(Net1, self).__init__()
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

		self.fc7 = nn.Linear(64, 10)


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