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
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.data.sampler import SequentialSampler

from wideresnet import WideResNet

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

PATH = 'C:/Users/Roop Pal/workspace/meta' #'/media/chad/nara/'

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

#Lists to hold the intermediate outputs captured by hooks
final_fc_pre_hook_list = []
penultimate_conv_hook_list = []
last_conv_hook_list = []
pre_bn_list = []
post_bn_list = []


def main():
    global args, best_prec1
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.augment:
        transform_train = transforms.Compose([
        	transforms.ToTensor(),
        	transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
        						(4,4,4,4),mode='reflect').squeeze()),
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
    assert(args.dataset == 'cifar10' or args.dataset == 'cifar100')


    valid_indices = np.load('underlying_train_indices.npy')
    train_indices = np.load('underlying_valid_indices_ie_meta_train_indices.npy')

    full_dataset = datasets.CIFAR100('../data', train = True, transform = transform_test, download=True)

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(full_dataset, valid_indices)



    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = False)

    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = args.batch_size, shuffle = False)




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


    saved_state = torch.load('runs/WideResNet-28-10-run2/model_best.pth.tar')

    model.load_state_dict(saved_state['state_dict'])

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum, nesterov = args.nesterov,
                                weight_decay=args.weight_decay)


    # generate intermediate outputs
    prec1 = generate_intermediate_outputs(val_loader, model, criterion, 0)



def penultimate_conv_layer_hook(model, input, output):
    penultimate_conv_hook_list.append(output.data.cpu())

def last_conv_layer_hook(model, input, output):
    last_conv_hook_list.append(output.data.cpu())

def pre_bn_hook(model, input, output):
    pre_bn_list.append(input[0].cpu())

def post_bn_hook(model, input, output):
    post_bn_list.append(output.data.cpu())

def fc_input_hook(model, input, output):
    final_fc_pre_hook_list.append(input[0].cpu())

def generate_intermediate_outputs(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()


    list_of_output_lists = []

    error_output_lists = []
    correct_output_lists = []

    lists_of_target_lists = []

    #make hooks to capture intermediate layer outputs
    hook1 = model.fc.register_forward_hook(fc_input_hook)
    hook2 = model.bn1.register_forward_hook(post_bn_hook)
    hook3 = model.bn1.register_forward_hook(pre_bn_hook)
    hook4 = list(model.block3.children())[0][5].conv2.register_forward_hook(last_conv_layer_hook)
    hook5 = list(model.block3.children())[0][5].conv1.register_forward_hook(penultimate_conv_layer_hook)

    model.eval()


    end = time.time()
    for i, (input, target) in enumerate(val_loader):
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

        which_correct, this_batch_correct, this_batch_incorrect = get_acc_info(output.data,target,topk = (1,))

        which_correct = which_correct.cpu().numpy()
        which_correct = which_correct.reshape((len(target)))

        where_which_correct = np.where(which_correct == 1)[0]

        where_which_incorrect = np.where(which_correct == 0)[0]

        correct_outputs = output[where_which_correct].cpu()
        incorrect_outputs = output[where_which_incorrect].cpu()

        correct_penultimate_conv_hooks = penultimate_conv_hook_list[0][where_which_correct]
        incorrect_penultimate_conv_hooks = penultimate_conv_hook_list[0][where_which_incorrect]

        correct_last_conv_hooks = last_conv_hook_list[0][where_which_correct]
        incorrect_last_conv_hooks = last_conv_hook_list[0][where_which_incorrect]

        correct_pre_bn_hooks = pre_bn_list[0][where_which_correct]
        incorrect_pre_bn_hooks = pre_bn_list[0][where_which_incorrect]

        correct_post_bn_hooks = post_bn_list[0][where_which_correct]
        incorrect_post_bn_hooks = post_bn_list[0][where_which_incorrect]

        correct_final_fc_pre_hooks = final_fc_pre_hook_list[0][where_which_correct]
        incorrect_final_fc_pre_hooks = final_fc_pre_hook_list[0][where_which_incorrect]

        torch.save(correct_outputs, PATH + 'cifar/wide/valid/output/correct/correct_output'+str(i)+'.torch')
        torch.save(incorrect_outputs, PATH + 'cifar/wide/valid/output/incorrect/incorrect_output'+str(i)+'.torch')

        torch.save(correct_penultimate_conv_hooks, PATH + 'cifar/wide/valid/penultimate_conv_layer/correct/correct_inter_output'+str(i)+'.torch')
        torch.save(incorrect_penultimate_conv_hooks, PATH + 'cifar/wide/valid/penultimate_conv_layer/incorrect/incorrect_inter_output'+str(i)+'.torch')

        torch.save(correct_last_conv_hooks, PATH + 'cifar/wide/valid/last_conv_layer/correct/correct_inter_output'+str(i)+'.torch')
        torch.save(incorrect_last_conv_hooks, PATH + 'cifar/wide/valid/last_conv_layer/incorrect/incorrect_inter_output'+str(i)+'.torch')

        torch.save(correct_pre_bn_hooks, PATH + 'cifar/wide/valid/pre_bn_layer/correct/correct_inter_output'+str(i)+'.torch')
        torch.save(incorrect_pre_bn_hooks, PATH + 'cifar/wide/valid/pre_bn_layer/incorrect/incorrect_inter_output'+str(i)+'.torch')

        torch.save(correct_post_bn_hooks, PATH + 'cifar/wide/valid/post_bn_layer/correct/correct_inter_output'+str(i)+'.torch')
        torch.save(incorrect_post_bn_hooks, PATH + 'cifar/wide/valid/post_bn_layer/incorrect/incorrect_inter_output'+str(i)+'.torch')

        torch.save(correct_final_fc_pre_hooks, PATH + 'cifar/wide/valid/pre_final_fc_layer/correct/correct_inter_output'+str(i)+'.torch')
        torch.save(incorrect_final_fc_pre_hooks, PATH + 'cifar/wide/valid/pre_final_fc_layer/incorrect/incorrect_inter_output'+str(i)+'.torch')

        #clear hook lists
        penultimate_conv_hook_list[:] = []

        final_fc_pre_hook_list[:] = []
        last_conv_hook_list[:] = []
        pre_bn_list[:] = []
        post_bn_list[:] = []




        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))



    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg

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
        if correct[:,i].sum()==1:
            correctly_classified.append(soutput[i])
        else:
            incorrectly_classified.append(soutput[i])

    return(correct, correctly_classified,incorrectly_classified)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%(args.name) + 'model_best.pth.tar')

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
    lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
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