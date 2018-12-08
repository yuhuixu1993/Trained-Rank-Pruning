
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from decompose import VH_decompose_model, channel_decompose, \
    EnergyThreshold, ValueThreshold, LinearRate


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[81, 122],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#decouple options
parser.add_argument('--nuclear-weight', type=float, default=None, help='The weight for nuclear norm regularization')
parser.add_argument('--retrain', dest='retrain',help='wether retrain from a decoupled model, only valid when evaluation is on', action='store_true')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device_id = int(args.gpu_id)
use_cuda = torch.cuda.is_available() and device_id >= 0

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

DEBUG = False # debug option for singular value

f_decouple = channel_decompose# set decouple method

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    # model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda(torch.device('cuda:1'))
    model = model.cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

	if DEBUG:
		# print(model)
		show_low_rank(model, input_size=[32, 32])

	print(' Start decomposition:')
	look_up_table = get_look_up_table(model)
	#print(model)
        thresholds = [0.85]#+0.03*x for x in range(10)]
        T = np.array(thresholds)
        cr = np.zeros(T.shape)
        acc = np.zeros(T.shape)
        
        model_path = 'net.pth'
        torch.save(model, model_path)

        for i, t in enumerate(thresholds):
            test_model = torch.load(model_path)        

            cr[i] = show_low_rank(test_model, input_size=[32, 32], criterion=EnergyThreshold(t))
	    test_model = f_decouple(test_model, look_up_table,
                    criterion=EnergyThreshold(t),train=False)
	    #print(model)
	    print(' Done! test decoupled model')
	    test_loss, test_acc = test(testloader, test_model, criterion, start_epoch, use_cuda)
            print(' Test Loss :  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
            acc[i] = test_acc
	    
            if args.retrain:
            # retrain model
                print(' Retrain decoupled model')
                finetune_epoch = 4
                args.schedule = [int(finetune_epoch/2), finetune_epoch]
                best_acc = 0.0
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
                global state
                init_lr = args.lr
                state['lr'] = init_lr

                for epoch in range(finetune_epoch):
                    adjust_learning_rate(optimizer, epoch)
                    train_loss, train_acc = train(trainloader, test_model, criterion, optimizer, epoch, use_cuda)
                    test_loss, test_acc = test(testloader, test_model, criterion, epoch, use_cuda)
                    best_acc = max(test_acc, best_acc)

                    # append logger file
                    # logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

                #logger.close()
                #logger.plot()
		#savefig(os.path.join(args.checkpoint, 'log.eps'))

	        acc[i] = best_acc

        torch.save(OrderedDict([('acc',acc),('cr', cr)]), 'test.pth')
        print(cr)
        print(acc)

        return

    # Train and val

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def get_look_up_table(model):
    count = 0
    look_up_table = []
    First_conv = True
    for name, m in model.named_modules():
        #TODO: change the if condition here to select different kernel to decouple
        if isinstance(m, nn.Conv2d) and m.kernel_size != (1,1) and count > 0:
            if First_conv:
                First_conv = False
            else:
                look_up_table.append(name)

        count += 1

    return look_up_table


def show_low_rank(model, input_size=None, criterion=None):
    
    look_up_table = get_look_up_table(model)
    # threshold = 0.9
    # left = 0.25
    redundancy = OrderedDict()
    comp_rate = OrderedDict()

    if input_size is not None:
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
	elif isinstance(input_size, list):
            pass
        else:
            raise Exception

    origin_FLOPs = 0.
    decouple_FLOPs = 0.

    for name, m in model.named_modules():

        if not isinstance(m, nn.Conv2d):
            continue

        p = m.weight.data
        dim = p.size()
        FLOPs = dim[0]*dim[1]*dim[2]*dim[3]

        if name in look_up_table and m.stride == (1,1):

            NC = p.view(dim[0], -1)
            N, sigma, C = torch.svd(NC, some=True)
            #VH = p.permute(1,2,0,3).contiguous().view(dim[1]*dim[2],-1)
            #
            #U, sigma, V =torch.svd(VH, some=True)
            if DEBUG:
                print('sigma in module %s :' % name)
                print(sigma)
            if criterion is not None:
                item_num = criterion(sigma)
                print(item_num)
                # lambda_ = (sigma**2).data
                # sum_e = lambda_.sum()
                # invalid = float(((sigma**2).data < threshold).sum())
                # total = sigma.size(0)
                # for i in xrange(total):
                #     if lambda_[:(i+1)].sum()/sum_e > threshold:
                #         item_num = i + 1
                #         break
                #new_FLOPs = dim[1]*item_num*dim[2]+dim[0]*item_num*dim[3]
                new_FLOPs = dim[1]*dim[2]*dim[3]*item_num + item_num*dim[0]
                rate = float(new_FLOPs)/FLOPs
                # redundancy[name] = ('%.2f' % (float(item_num)/total*100) ) + '%'
                comp_rate[name] = ('%.2f' % (rate) )

        elif criterion is not None:
            new_FLOPs = FLOPs

        if input_size is not None and criterion is not None:
            if 'downsample' not in name:
                output_h = input_size[0]/m.stride[0]
                output_w = input_size[1]/m.stride[1]
            else:
                output_h = input_size[0]
                output_w = input_size[1]
            origin_FLOPs += FLOPs*output_h*output_w
            decouple_FLOPs += new_FLOPs*output_h*output_w
            input_size = [output_h, output_w]

    if criterion is not None:
        r = origin_FLOPs / decouple_FLOPs
        # print(redundancy)
        # print('\n')
        print(comp_rate)
        print('\n')
        print('comp rate:')
        print(r)

        return r
             

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    look_up_table = get_look_up_table(model)
    #print(look_up_table)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
	
        if args.nuclear_weight is not None:
            f_decouple(model,look_up_table,criterion=None,train=True,lambda_=state['nuclear_weight'], truncate=0.1)
            #nuclear_regularize(model, look_up_table, state, verbose=False)
	
        count = 0
        """
            for name, m in model.named_modules():
            if name in look_up_table and isinstance(m, nn.Conv2d):
                param = m.weight
                dim = param.size()
                # print(dim)
                VH = param.permute(1,2,0,3).contiguous().view(dim[1]*dim[2], -1)
    
                U, sigma, V = torch.svd(VH, some=True)
                if count == 5:
                    print(sigma)
                count += 1
        """

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
