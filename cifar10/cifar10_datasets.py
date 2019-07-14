import torch
from torch.utils.data import Dataset

def process_layer_data(data, layer_name):
	processed_data = None

	if 'conv' in layer_name:
		processed_data = data.reshape(data.shape[0] * data.shape[1] * data.shape[2])
	else:
		processed_data = data
	
	return processed_data

class ImageNetInterMediateLayersInMemoryDataset(Dataset):
    def __init__(self, error_files, correct_files, one_class = False, transform=None, layer_name=None):


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
        self.layer_name = layer_name

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
        
        
        data_shape = self.X_data[0][0][0].shape
        if len(data_shape) > 1:
            self.dim_size = data_shape[0] * data_shape[1] * data_shape[2]
        else:
            self.dim_size = data_shape[0]

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
        global LAYER_NAME
        Xs_to_return = []

        for layer in range(len(self.X_data)):
            data = self.X_data[layer][idx][0].float()
            processed_data = process_layer_data(data, self.layer_name)
            Xs_to_return.append(processed_data)
            
        #Xs_to_return = (Xs_to_return[0], Xs_to_return[1], Xs_to_return[2], Xs_to_return[3])
        Xs_to_return = (Xs_to_return[0],self.X_data[layer][idx][1])

        if self.y_data[idx] == 1:
            self.correct_running_count += 1
        else:
            self.incorrect_running_count += 1

        return (Xs_to_return, self.y_data[idx].long())

    def get_correct_len(self):
        return self.correct_len

    def get_error_len(self):
        return self.error_len

    def get_x_data(self):
        return self.X_data[0]

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

class ImageNetTargetGroundInterMediateLayersInMemoryDataset(Dataset):
    def __init__(self, error_files, correct_files, one_class = False, transform=None, layer_name=None):


        num_errors = 0
        num_correct = 0
        self.correct_running_count = 0
        self.incorrect_running_count = 0
        self.X_data = []
        self.y_data = []
        correct_data = []
        error_data = []
        self.num_filters = 0
        self.isConv = False
        self.dim_size = 0
        self.layer_name = layer_name

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
                self.y_data.append(loaded[item_idx][3])

        num_correct = len(self.X_data[0])


        for layer_index in range(len(error_files)):
            loaded = torch.load(error_files[layer_index])
            for item_idx in range(len(loaded)):
                self.X_data[layer_index].append(loaded[item_idx])
                self.y_data.append(loaded[item_idx][3])

        num_errors = len(self.X_data[0]) - num_correct
        
        
        data_shape = self.X_data[0][0][0].shape
        if len(data_shape) > 1:
            self.dim_size = data_shape[0] * data_shape[1] * data_shape[2]
        else:
            self.dim_size = data_shape[0]

        self.correct_len = num_correct
        self.error_len = num_errors
        self.total_len = self.error_len + self.correct_len
        
        print('number of errors')
        print(num_errors)
        print('size of dataset')
        print(num_errors+num_correct)
        print('percentage of errors')
        print(num_errors/(0.0+num_errors+num_correct))


    def __len__(self):
        return self.total_len


    def __getitem__(self, idx):
        global LAYER_NAME
        Xs_to_return = []

        for layer in range(len(self.X_data)):
            data = self.X_data[layer][idx][0].float()
            processed_data = process_layer_data(data, self.layer_name)
            Xs_to_return.append(processed_data)
            
        #Xs_to_return = (Xs_to_return[0], Xs_to_return[1], Xs_to_return[2], Xs_to_return[3])
        Xs_to_return = (Xs_to_return[0],self.X_data[layer][idx][1])

        if self.y_data[idx] == 1:
            self.correct_running_count += 1
        else:
            self.incorrect_running_count += 1

        return (Xs_to_return, self.y_data[idx].long())

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

class ImageNetTargetPredictInterMediateLayersInMemoryDataset(Dataset):
    def __init__(self, error_files, correct_files, one_class = False, transform=None, layer_name=None):


        num_errors = 0
        num_correct = 0
        self.correct_running_count = 0
        self.incorrect_running_count = 0
        self.X_data = []
        self.y_data = []
        correct_data = []
        error_data = []
        self.num_filters = 0
        self.isConv = False
        self.dim_size = 0
        self.layer_name = layer_name

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
                self.y_data.append(loaded[item_idx][2])

        num_correct = len(self.X_data[0])


        for layer_index in range(len(error_files)):
            loaded = torch.load(error_files[layer_index])
            for item_idx in range(len(loaded)):
                self.X_data[layer_index].append(loaded[item_idx])
                self.y_data.append(loaded[item_idx][2])

        num_errors = len(self.X_data[0]) - num_correct
        
        
        data_shape = self.X_data[0][0][0].shape
        if len(data_shape) > 1:
            self.dim_size = data_shape[0] * data_shape[1] * data_shape[2]
        else:
            self.dim_size = data_shape[0]

        self.correct_len = num_correct
        self.error_len = num_errors
        self.total_len = self.error_len + self.correct_len
        
        print('number of errors')
        print(num_errors)
        print('size of dataset')
        print(num_errors+num_correct)
        print('percentage of errors')
        print(num_errors/(0.0+num_errors+num_correct))


    def __len__(self):
        return self.total_len


    def __getitem__(self, idx):
        global LAYER_NAME
        Xs_to_return = []

        for layer in range(len(self.X_data)):
            data = self.X_data[layer][idx][0].float()
            processed_data = process_layer_data(data, self.layer_name)
            Xs_to_return.append(processed_data)
            
        #Xs_to_return = (Xs_to_return[0], Xs_to_return[1], Xs_to_return[2], Xs_to_return[3])
        Xs_to_return = (Xs_to_return[0],self.X_data[layer][idx][1])

        if self.y_data[idx] == 1:
            self.correct_running_count += 1
        else:
            self.incorrect_running_count += 1

        return (Xs_to_return, self.y_data[idx].long())

    def get_correct_len(self):
        return self.correct_len

    def get_error_len(self):
        return self.error_len

    def get_x_data(self):
        return self.X_data[0]
        
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