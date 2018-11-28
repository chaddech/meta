import argparse
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler
import IPython
import logging
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
import sys
from models.baseline_snli import encoder
from models.baseline_snli import atten
import argparse
from models.snli_data import snli_data
from models.snli_data import w2v
from random import shuffle


import sys

sys.path.insert(0, './models/')
from baseline_snli import atten

# used for logging to TensorBoard
best_prec1 = 0
CUDA_DEVICE = 'cuda:0'
torch.cuda.set_device(0)


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

hooked_layers = [Layer('post_f_final_relu', 'output', lambda model: model.mlp_f[5]),
                 Layer('post_g_final_relu', 'output', lambda model: model.mlp_g[5]),
                 Layer('post_h_final_relu', 'output', lambda model: model.mlp_h[5])]


def train(args):
    if args.max_length < 0:
        args.max_length = 9999

    # initialize the logger
    # create logger
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # file handler
    fh = logging.FileHandler(args.log_dir + args.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    torch.cuda.set_device(args.gpu_id)

    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    # load train/dev/test data
    # train data
    logger.info('loading data...')
    train_data = snli_data(args.train_file, args.max_length, meta=args.dev_file)
    train_batches = train_data.batches
    train_lbl_size = 3
    dev_data = snli_data(args.test_file, args.max_length)
    # todo: use a better dev_data (from train or val. not test)
    dev_batches = dev_data.batches
    # test_data = snli_data(args.test_file, args.max_length)
    # test_batches = test_data.batches
    logger.info('train size # sent ' + str(train_data.size))
    logger.info('dev size # sent ' + str(dev_data.size))

    # get input embeddings
    logger.info('loading input embeddings...')
    word_vecs = w2v(args.w2v_file).word_vecs

    best_dev = []   # (epoch, dev_acc)

    logger.info('loading input embeddings...')
    word_vecs = w2v(args.w2v_file).word_vecs

    best_dev = []  # (epoch, dev_acc)

    # build the model
    input_encoder = encoder(word_vecs.size(0), args.embedding_size, args.hidden_size, args.para_init)
    input_encoder.embedding.weight.data.copy_(word_vecs)
    input_encoder.embedding.weight.requires_grad = False
    inter_atten = atten(args.hidden_size, train_lbl_size, args.para_init)

    input_encoder.cuda()
    inter_atten.cuda()

    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters())
    para2 = inter_atten.parameters()

    if args.optimizer == 'Adagrad':
        input_optimizer = optim.Adagrad(para1, lr=args.lr, weight_decay=args.weight_decay)
        inter_atten_optimizer = optim.Adagrad(para2, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adadelta':
        input_optimizer = optim.Adadelta(para1, lr=args.lr)
        inter_atten_optimizer = optim.Adadelta(para2, lr=args.lr)
    else:
        logger.info('No Optimizer.')
        sys.exit()

    criterion = nn.NLLLoss(size_average=True)

    saved_state = torch.load('./data/runs/420_10_10_epoch10/log54_epoch-189_dev-acc-0.833_input-encoder.pt')

    input_encoder.load_state_dict(saved_state)

    saved_state = torch.load('./data/runs/420_10_10_epoch10/log54_epoch-189_dev-acc-0.833_inter-atten.pt')

    inter_atten.load_state_dict(saved_state)

    prec1 = generate_intermediate_outputs(train_data, (input_encoder,inter_atten), criterion, 0)


def generate_intermediate_outputs(val_loader, model, criterion, epoch):
    input_encoder, inter_atten = model
    # model = inter_atten

    list_of_output_lists = []

    error_output_lists = []
    correct_output_lists = []

    lists_of_target_lists = []
    # make hooks to capture intermediate layer outputs
    for layer in hooked_layers:
        model_layer = layer.model_layer(inter_atten)
        model_layer.register_forward_hook(layer.hook_fn)
    #
    # input_encoder.load_state_dict(torch.load(best_model_fname + '_input-encoder.pt'))
    # inter_atten.load_state_dict(torch.load(best_model_fname + '_inter-atten.pt'))

    input_encoder.eval()
    inter_atten.eval()

    correct = 0.
    total = 0.
    test_batches = val_loader.batches
    for i in range(len(test_batches)):
        test_src_batch, test_tgt_batch, test_lbl_batch = test_batches[i]

        test_src_batch = Variable(test_src_batch.cuda())
        test_tgt_batch = Variable(test_tgt_batch.cuda())
        test_lbl_batch = Variable(test_lbl_batch.cuda())

        test_src_linear, test_tgt_linear=input_encoder(
            test_src_batch, test_tgt_batch)
        log_prob=inter_atten(test_src_linear, test_tgt_linear)

        _, predict=log_prob.data.max(dim=1)
        total += test_lbl_batch.data.size()[0]
        correct += torch.sum(predict == test_lbl_batch.data).item()


        output = predict
        target = test_lbl_batch

        # IPython.embed()
        which_correct, this_batch_correct, this_batch_incorrect = get_acc_info(output.data, target)

        which_correct = which_correct.cpu().numpy()
        which_correct = which_correct.reshape((len(target)))

        where_which_correct = np.where(which_correct == 1)[0]

        where_which_incorrect = np.where(which_correct == 0)[0]

        correct_outputs = output[where_which_correct].cpu()
        incorrect_outputs = output[where_which_incorrect].cpu()

        correct_layer_hooks = []
        incorrect_layer_hooks = []
        for layer in hooked_layers:
            correct_layer_hooks.append(layer.hook_list[0][where_which_correct])
            incorrect_layer_hooks.append(layer.hook_list[0][where_which_incorrect])

        # correct_outputs = list(zip(correct_outputs, indices[where_which_correct]))
        # incorrect_outputs = list(zip(incorrect_outputs, indices[where_which_incorrect]))
        save_tensor(correct_outputs, os.getcwd() + '/snli/valid/output/correct/correct_output' + str(i) + '.torch')
        save_tensor(incorrect_outputs,
                    os.getcwd() + '/snli/valid/output/incorrect/incorrect_output' + str(i) + '.torch')

        for j, layer in enumerate(hooked_layers):
            save_tensor(correct_layer_hooks[j], os.getcwd() + '/snli/valid/' + layer.name + '_layer/correct/correct_inter_output' + str(i) + '.torch')
            save_tensor(incorrect_layer_hooks[j], os.getcwd() + '/snli/valid/' + layer.name + '_layer/incorrect/incorrect_inter_output' + str(i) + '.torch')
        # clear hook lists
        for layer in hooked_layers:
            layer.hook_list[:] = []

    test_acc = correct / total
    print('test-acc %.3f' % (test_acc))
    return test_acc


def save_tensor(tensor, path):
    directory = path[:path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(tensor, path)


def get_acc_info(output, target):
    batch_size = target.size(0)
    correctly_classified = []
    incorrectly_classified = []

    for i in range(batch_size):
        if output[i] == target[i]:
            correctly_classified.append(output[i])
        else:
            incorrectly_classified.append(output[i])

    return output == target, correctly_classified, incorrectly_classified


if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--train_file', help='training data file (hdf5)',
                        type=str, default='../decomp-attn/data/entail-train.hdf5')

    parser.add_argument('--dev_file', help='development data file (hdf5)',
                        type=str, default='../decomp-attn/data/entail-val.hdf5')

    parser.add_argument('--test_file', help='test data file (hdf5)',
                        type=str, default='../decomp-attn/data/entail-test.hdf5')

    parser.add_argument('--w2v_file', help='pretrained word vectors file (hdf5)',
                        type=str, default='../decomp-attn/data/glove.hdf5')

    parser.add_argument('--log_dir', help='log file directory',
                        type=str, default='./data')

    parser.add_argument('--log_fname', help='log file name',
                        type=str, default='log54.log')

    parser.add_argument('--gpu_id', help='GPU device id',
                        type=int, default=0)

    parser.add_argument('--embedding_size', help='word embedding size',
                        type=int, default=300)

    parser.add_argument('--epoch', help='training epoch',
                        type=int, default=250)

    parser.add_argument('--dev_interval', help='interval for development',
                        type=int, default=1)

    parser.add_argument('--optimizer', help='optimizer',
                        type=str, default='Adagrad')

    parser.add_argument('--Adagrad_init', help='initial accumulating values for gradients',
                        type=float, default=0.)

    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.05)

    parser.add_argument('--hidden_size', help='hidden layer size',
                        type=int, default=300)

    parser.add_argument('--max_length', help='maximum length of training sentences,\
                        -1 means no length limit',
                        type=int, default=10)

    parser.add_argument('--display_interval', help='interval of display',
                        type=int, default=1000)

    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
                               to have the norm equal to max_grad_norm',
                        type=float, default=5)

    parser.add_argument('--para_init', help='parameter initialization gaussian',
                        type=float, default=0.01)

    parser.add_argument('--weight_decay', help='l2 regularization',
                        type=float, default=5e-5)

    parser.add_argument('--model_path', help='path of model file (not include the name suffix',
                        type=str, default='./data/runs/420_10_10_epoch10/')

    args=parser.parse_args()
    # args.max_lenght = 10   # args can be set manually like this
    train(args)

else:
    pass