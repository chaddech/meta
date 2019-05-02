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
#import torchvision.datasets as datasets
from cifar_dataset_w_idx import cifar10
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler
#from fewer_max_vgg import *
import sys
#sys.path.insert(0, '../WideResNet-pytorch/')
import torchvision.models as models
from datafolder import ImageFolder
# used for logging to TensorBoard
from tensorboard_logger import configure, log_value
from new_smaller_vgg import vgg19_bn

# **************************************************
# PLACES WHERE CHANGES ARE NEEDED ARE MARKED WITH **
# **************************************************

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# ** MODE SHOULD BE EITHER TRAIN, VALID, OR TEST
mode = 'valid'

# ** CHANGE IF DIFFERENT MODEL NAME
model_name = 'vgg19_bn'


# ** CHANGE BASE PATH 
base_path = "~/cifar10_intermediates/"
parser = argparse.ArgumentParser(description='vgg_cifar_10_get_intermediates')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')


parser.add_argument('-a', '--arch', default='vgg19_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg19_bn)')


parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='vgg19bn', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', type=int, default=10027, metavar='S',
                    help='random seed (default: 10027)')
parser.add_argument('--cuda', type=int, default=-1,
                    help='CUDA device to use')
parser.add_argument('--mode', default='train',
                    help='Mode: train, valid, or test')
parser.set_defaults(augment=False)


best_prec1 = 0

# ** SET CUDA DEVICE
DEVICE = None
# CUDA_DEVICE = 'cuda:2'
# torch.cuda.set_device(2)



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


# Lists to hold the intermediate outputs captured by hooks
# ** UPDATE BASED ON WHICH LAYERS YOU WANT TO SAVE
hooked_layers = [Layer('conv_0_layer', 'output', lambda model: list(model.children())[0][0]),
                 Layer('conv_21_layer', 'output', lambda model: list(model.children())[0][21]),
                 Layer('conv_47_layer', 'output', lambda model: list(model.children())[0][47]),
                 Layer('conv_47_post_maxpool_layer', 'output', lambda model: list(model.children())[0][50]),
                 Layer('fc_0_layer', 'output', lambda model: list(model.children())[-1][0]),
                 Layer('fc_3_layer', 'output', lambda model: list(model.children())[-1][3])]


def main():
    global device, args, best_prec1, mode
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Setting mode
    mode = args.mode

    # Setting CUDA device
    DEVICE = torch.device(args.cuda)

    #IGNORE
    if args.tensorboard:
        configure("/media/chad/delft/imagenet_intermediates/" + model_name+ "/" + mode + "/tensorboard/%s" % args.name)

    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                (4,4,4,4),mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])

    # TODO: Change transform to: https://github.com/kuangliu/pytorch-cifar/issues/19 ?
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])
    kwargs = {'num_workers': 1, 'pin_memory': True}

    if mode == 'valid':
        valid_indices = np.load('cifar10_valid_indices.npy')
    elif mode == 'test':
        valid_indices = np.load('cifar10_test_indices.npy')

    valid_sampler = SubsetRandomSampler(valid_indices)

    # ** POSSIBLY CHANGE WHERE DATASET GOES
    train_loader = torch.utils.data.DataLoader(
        cifar10('../../cifar10_data', train=True, download=True,
                         transform=transform_train),
        batch_size=args.batch_size, shuffle = True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        cifar10('../../cifar10_data', train=False, transform=transform_test),
        batch_size=args.batch_size, sampler = valid_sampler, **kwargs)

    # create model
    model = vgg19_bn(num_classes=10)

    # ** CHANGE LOCATION OF LOADED MODEL
    saved_state = torch.load('./cifar10_vgg19_bn_model_best91340.pth.tar')
    model.load_state_dict(saved_state['state_dict'])
    # get the number of model parameters

    #device = torch.device(CUDA_DEVICE)
    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.to(DEVICE)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # generate intermediate outputs
    if mode == 'train':
        prec1 = generate_intermediate_outputs(train_loader, model, criterion, 0)
    elif mode == 'valid' or mode == 'test':
        prec1 = generate_intermediate_outputs(val_loader, model, criterion, 0)



def generate_intermediate_outputs(val_loader, model, criterion, epoch):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    list_of_output_lists = []

    error_output_lists = []
    correct_output_lists = []

    lists_of_target_lists = []
    # make hooks to capture intermediate layer outputs
    for layer in hooked_layers:
        model_layer = layer.model_layer(model)
        model_layer.register_forward_hook(layer.hook_fn)

    model.eval()
    list_of_correct_indices = []
    list_of_incorrect_indices = []

    end = time.time()
    batch_counter = 0
    for i, (indices, paths, input, target) in enumerate(val_loader):


        target = target.cuda(non_blockin=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        soutput = F.softmax(output)

        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        which_correct, this_batch_correct, this_batch_incorrect, predictions = get_acc_info(output.data, target, topk=(1,))

        which_correct = which_correct.cpu().numpy()
        which_correct = which_correct.reshape((len(target)))

        where_which_correct = np.where(which_correct == 1)[0]

        where_which_incorrect = np.where(which_correct == 0)[0]

        correct_outputs = output[where_which_correct].cpu()
        incorrect_outputs = output[where_which_incorrect].cpu()


        correct_layer_hooks = []
        incorrect_layer_hooks = []

        predictions = predictions.cpu().numpy()
        predictions = predictions.reshape((len(target)))

        correct_predictions = predictions[where_which_correct]
        incorrect_predictions = predictions[where_which_incorrect]

        for layer in hooked_layers:
            correct_layer_output = list(zip(layer.hook_list[0][where_which_correct], indices[where_which_correct], predictions[where_which_correct], target[where_which_correct]))
            incorrect_layer_output = list(zip(layer.hook_list[0][where_which_incorrect], indices[where_which_incorrect], predictions[where_which_incorrect], target[where_which_incorrect]))

            correct_layer_hooks.append(correct_layer_output)
            incorrect_layer_hooks.append(incorrect_layer_output)


        correct_outputs = list(zip(correct_outputs, indices[where_which_correct], correct_predictions, target[where_which_correct]))
        incorrect_outputs = list(zip(incorrect_outputs, indices[where_which_incorrect], incorrect_predictions, target[where_which_incorrect]))

        paths = np.asarray(paths)

        correct_indices_and_paths = list(zip(indices[where_which_correct], target[where_which_correct], predictions[where_which_correct], paths[where_which_correct]))
        incorrect_indices_and_paths = list(zip(indices[where_which_incorrect], target[where_which_incorrect], predictions[where_which_incorrect], paths[where_which_incorrect]))
 

        # ** CHECK PATHS -- SHOULD BE OK BASED ON BASE PATH
        save_tensor(correct_outputs, base_path + '/' + model_name+ '/' + mode + '/output/correct/correct_output' + str(i) + '.torch')
        save_tensor(incorrect_outputs,
                    base_path + '/' + model_name+ '/' + mode + '/output/incorrect/incorrect_output' + str(i) + '.torch')
        save_tensor(correct_indices_and_paths, base_path+'/' + model_name+ '/' + mode + '/indices/correct/indices_and_paths' + str(i) + '.torch')
        save_tensor(incorrect_indices_and_paths,
                    base_path + '/' + model_name+ '/' + mode + '/indices/incorrect/indices_and_paths' + str(i) + '.torch')
        for x in indices[where_which_correct]:
            list_of_correct_indices.append(x)

        for x in indices[where_which_incorrect]:
            list_of_incorrect_indices.append(x)

        # ** CHECK PATHS -- SHOULD BE OK BASED ON BASE PATH

        for j, layer in enumerate(hooked_layers):
            save_tensor(correct_layer_hooks[j], base_path + '/' + model_name+ '/' + mode + '/' + layer.name + '_layer/correct/correct_inter_output' + str(i) + '.torch')
            save_tensor(incorrect_layer_hooks[j], base_path + '/' + model_name+ '/' + mode + '/' + layer.name + '_layer/incorrect/incorrect_inter_output' + str(i) + '.torch')
        # clear hook lists
        for layer in hooked_layers:
            layer.hook_list[:] = []

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    # ** CHECK PATHS -- SHOULD BE OK BASED ON BASE PATH    
    np.save(base_path + '/' + model_name+ '/' + mode + '/indices/correct/all_correct_indices.npy', list_of_correct_indices)
    np.save(base_path + '/' + model_name+ '/' + mode + '/indices/incorrect/all_incorrect_indices.npy', list_of_incorrect_indices)

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


def save_tensor(tensor, path):
    directory = path[:path.rfind('/')]
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(tensor, path)


def get_acc_info(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    soutput = nn.functional.softmax(torch.autograd.Variable(output))

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correctly_classified = []
    incorrectly_classified = []

    for i in range(batch_size):
        if correct[:, i].sum() == 1:
            correctly_classified.append(soutput[i])
        else:
            incorrectly_classified.append(soutput[i])

    return correct, correctly_classified, incorrectly_classified, pred


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "../WideResNet-pytorch/runs/%s/" % args.name
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../WideResNet-pytorch/runs/%s/' % args.name + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
