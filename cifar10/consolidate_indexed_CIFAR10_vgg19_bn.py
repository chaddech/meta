import torch
import numpy as np
import os


# ** CHANGE BASE PATH AND INTER LAYERS TO MATCH DIRECTORY STRUCTURE AND LAYERS SAVED
base_path = '/home/seungwookhan/cifar_imtermediates/'
inter_layers = ['conv_0_layer_layer/', 'conv_21_layer_layer/', 'conv_47_layer_layer/', 'conv_47_post_maxpool_layer_layer/'
				'fc_0_layer_layer/', 'fc_3_layer_layer/', 'indices/', 'output/']


for inter_layer in inter_layers:
	for which_set in ['train/', 'valid/', 'test/']:
		for in_co in ['correct/', 'incorrect/']:
			tensors = []
			target_dir = base_path + which_set+inter_layer+ in_co
			print(target_dir)
			files = os.listdir(target_dir)

			files.sort()


			for i in files:
				if 'sorted' not in i:
					t = torch.load(target_dir+'/'+i)
					for x in t:
						tensors.append(x)


			torch.save(tensors, base_path + which_set+inter_layer+in_co+'sorted_outputs.torch')

			#np.save(base_path + which_set+inter_layer+in_co+'_sorted_outputs.np', np_indices)



