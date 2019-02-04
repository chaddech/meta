import torch
import numpy as np
import os


base_path = '/media/chad/paro/vgg11_bn/'
inter_layers = ['output/']
for inter_layer in inter_layers:
	for which_set in ['test/','train/','valid/']:
		for in_co in ['correct', 'incorrect']:
			tensors = []
			target_dir = base_path + which_set+inter_layer+ in_co + '/'
			print(target_dir)
			files = os.listdir(target_dir)

			files.sort()

			for i in files:
				t = torch.load(target_dir+'/'+i)
				for x in t:

					tensors.append(x[1])
			indices = []
			for i in tensors:
				indices.append(i.numpy())
			np_indices = np.asarray(indices)
			#torch.save(tensors, base_path + which_set+inter_layer+in_co+'_sorted_outputs_indices')
			np.save(base_path + which_set+inter_layer+in_co+'_sorted_outputs.np', np_indices)



