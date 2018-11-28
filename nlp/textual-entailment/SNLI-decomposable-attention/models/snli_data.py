import torch
import numpy as np
import h5py

class snli_data(object):
    '''
        class to handle training data
    '''

    def __init__(self, fname, max_length, meta=False):

        if max_length < 0:
            max_length = 9999

        f = h5py.File(fname, 'r')
        self.source = torch.from_numpy(np.array(f['source'])) - 1
        self.target = torch.from_numpy(np.array(f['target'])) - 1
        self.label = torch.from_numpy(np.array(f['label'])) - 1
        self.label_size = torch.from_numpy(np.array(f['label_size']))
        self.source_l = torch.from_numpy(np.array(f['source_l']))
        self.target_l = torch.from_numpy(np.array(f['target_l'])) # max target length each batch
        # idx in torch style; indicate the start index of each batch (starting
        # with 1)
        self.batch_idx = torch.from_numpy(np.array(f['batch_idx'])) - 1
        self.batch_l = torch.from_numpy(np.array(f['batch_l']))

        self.batches = []   # batches

        self.length = self.batch_l.size(0)  # number of batches

        self.size = 0   # number of sentences

        n = len(self.source)

        if fname == '../decomp-attn/data/entail-train.hdf5':
            if meta:
                # todo: use meta to get dev file too
                meta_train_indices = np.load('meta_train_indices.npy')
                self.source = self.source[meta_train_indices]
                self.target = self.target[meta_train_indices]
                self.label = self.label[meta_train_indices]
                self.length = 2595
            else:
                import random
                meta_train_indices = random.sample(range(len(self.source)), 80000)
                np.save('meta_train_indices.npy', meta_train_indices)
                train_indices = list(set(range(n)) - set(meta_train_indices))
                self.source = self.source[train_indices]
                self.target = self.target[train_indices]
                self.label = self.label[train_indices]
                # print(len(self.source_l))
            # elif fname == '../decomp-attn/data/entail-val.hdf5':
            #     meta_train_indices = np.load('meta_train_indices.npy')
            #     print(len(self.source_l))
            #     # np.save('meta_train_source.npy', self.source[meta_train_indices])
            #     # np.save('meta_train_target.npy', self.target[meta_train_indices])
            #     # np.save('meta_train_label.npy', self.label[meta_train_indices])
            #     # import IPython
            #     # IPython.embed()
            # else:
            #     print(len(self.source_l))
            #     assert fname == '../decomp-attn/data/entail-test.hdf5'
            #     # np.save('meta_val_source.npy', )
            #     # np.save('meta_val_source.npy', )
            #     # np.save('meta_val_source.npy', )
        for i in range(self.length):
            if self.source_l[i] <= max_length and self.target_l[i] <= max_length:
              batch = (self.source[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.source_l[i]],
                       self.target[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]][:, :self.target_l[i]],
                       self.label[self.batch_idx[i] : self.batch_idx[i] + self.batch_l[i]],
                       )
              self.batches.append(batch)
              self.size += self.batch_l[i]

class w2v(object):

  def __init__(self, fname):
    f = h5py.File(fname, 'r')
    self.word_vecs = torch.from_numpy(np.array(f['word_vecs']))

