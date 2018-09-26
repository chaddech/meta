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
import cifar_dataset_w_idx as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler
import IPython

import sys

sys.path.insert(0, '../WideResNet-pytorch/')
from wideresnet import WideResNet

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
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
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-4-drop50', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)

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


# Lists to hold the intermediate outputs captured by hooks
hooked_layers = [Layer('pre_final_fc', 'input', lambda model: model.fc),
                 Layer('post_penultimate_conv', 'output', lambda model: list(model.block3.children())[0][3].conv1),
                 Layer('post_last_conv', 'output', lambda model: list(model.block3.children())[0][3].conv2),
                 Layer('pre_bn', 'input', lambda model: model.bn1),
                 Layer('post_bn', 'output', lambda model: model.bn1),
                 Layer('pre_block3_bn1', 'input', lambda model: list(model.block3.children())[0][3].bn1),
                 Layer('pre_block2_bn1', 'input', lambda model: list(model.block2.children())[0][3].bn1),
                 Layer('post_block1_conv1', 'output', lambda model: list(model.block1.children())[0][3].conv1),
                 Layer('post_conv1', 'output', lambda model: model.conv1)]


def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard:
        configure("../WideResNet-pytorch/runs/%s" % args.name)

    # Data loading code
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100')

    train_indices = np.load('underlying_valid_indices_ie_meta_train_indices.npy')
    valid_indices = np.load('meta_val_indices.npy')

    train_dataset = datasets.CIFAR100('../data', train_indices, train=True, transform=transform_test, download=True)
    valid_dataset = datasets.CIFAR100('../data', valid_indices, train=True, transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # create model
    model = WideResNet(args.layers, 100,
                       args.widen_factor, dropRate=args.droprate)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    device = torch.device(CUDA_DEVICE)
    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    saved_state = torch.load('../WideResNet-pytorch/runs/model_60.pth.tar')

    model.load_state_dict(saved_state['state_dict'])

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov=args.nesterov,
                                weight_decay=args.weight_decay)

    # generate intermediate outputs
    prec1 = generate_intermediate_outputs(val_loader, model, criterion, 0)


def generate_intermediate_outputs(val_loader, model, criterion, epoch):
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

    end = time.time()
    for i, (indices, input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
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

        which_correct, this_batch_correct, this_batch_incorrect = get_acc_info(output.data, target, topk=(1,))

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

        correct_outputs = list(zip(correct_outputs, indices[where_which_correct]))
        incorrect_outputs = list(zip(incorrect_outputs, indices[where_which_incorrect]))
        save_tensor(correct_outputs, os.getcwd() + '/wide/valid/output/correct/correct_output' + str(i) + '.torch')
        save_tensor(incorrect_outputs,
                    os.getcwd() + '/wide/valid/output/incorrect/incorrect_output' + str(i) + '.torch')

        for j, layer in enumerate(hooked_layers):
            save_tensor(correct_layer_hooks[j], os.getcwd() + '/wide/valid/' + layer.name + '_layer/correct/correct_inter_output' + str(i) + '.torch')
            save_tensor(incorrect_layer_hooks[j], os.getcwd() + '/wide/valid/' + layer.name + '_layer/incorrect/incorrect_inter_output' + str(i) + '.torch')
        # clear hook lists
        for layer in hooked_layers:
            layer.hook_list[:] = []

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

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

    return correct, correctly_classified, incorrectly_classified


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
