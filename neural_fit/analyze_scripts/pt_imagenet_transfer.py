import argparse
import copy
import torch
import pdb
import torch.backends.cudnn as cudnn
import os
import sys
import json
import numpy as np
import pickle
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import time
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./brainscore_mask'))
from bs_fit_utils import get_dc_model, load_set_func

IMAGENET_FOLDER = os.environ.get(
        'IMAGENET_FOLDER',
        '/data5/chengxuz/Dataset/imagenet_raw/')
MODEL_SAVE_FOLDER = os.environ.get(
        'MODEL_SAVE_FOLDER',
        os.path.join(
            '/mnt/fs4/chengxuz/v4it_temp_results/.result_caching/',
            'pt_model_imagenet_transfer'))


def get_pt_imagenet_transfer_parser():
    parser = argparse.ArgumentParser(
            description='ImageNet transfer for pytorch models')
    parser.add_argument(
            '--set_func', type=str, 
            default=None,
            action='store')
    parser.add_argument(
            '--gpu', default='0', type=str, action='store')
    parser.add_argument(
            '--setting_name', type=str, 
            default=None,
            action='store')
    parser.add_argument(
            '--batch_size', type=int, 
            default=256,
            action='store')
    parser.add_argument(
            '--workers', type=int, 
            default=30,
            action='store')
    parser.add_argument(
            '--val_workers', type=int, 
            default=10,
            action='store')
    parser.add_argument(
            '--init_lr', type=float, 
            default=0.01,
            action='store')
    parser.add_argument(
            '--linear_type', type=str, 
            default='default',
            action='store')
    return parser

        
class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features for resnet"""
    def __init__(self, num_labels=1000):
        super(RegLog, self).__init__()
        self.linear = nn.Linear(512 * 7 * 7, num_labels)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)

        
class RegLogMid(nn.Module):
    """Creates logistic regression on top of frozen features for resnet"""
    def __init__(self, num_labels=1000, mid_units=1000):
        super(RegLogMid, self).__init__()
        self.linear_mid = nn.Linear(512 * 7 * 7, mid_units)
        self.linear_final = nn.Linear(mid_units, num_labels)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear_final(self.linear_mid(x))


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


def lr_decay_boundary(optimizer, epoch, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 
        if epoch >= 40:
            lr *= 0.1
        if epoch >= 100:
            lr *= 0.1
        if epoch >= 140:
            lr *= 0.1

        param_group['lr'] = lr


def lr_decay_boundary_faster(optimizer, epoch, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 
        if epoch >= 30:
            lr *= 0.1
        if epoch >= 60:
            lr *= 0.1
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ImageNetTransfer(object):
    def __init__(self, args):
        self.args = args

    def __build_dc_model(self):
        args = self.args
        assert getattr(args, 'load_from_ckpt', None) is not None, \
                "Must specify ckpt to load from"
        dc_model = get_dc_model(
                args.load_from_ckpt, verbose=False)
        self._model = dc_model.features
        self.add_preprocess = dc_model.sobel

    def __build_la_cmc_model(self):
        args = self.args
        sys.path.append(os.path.expanduser('~/RotLocalAggregation'))
        from src.models.resnet import ResNetLabV1
        model = ResNetLabV1(skip_final_layer=True, before_pool=True)
        model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model = model.cuda()
            cudnn.benchmark = True

        checkpoint = torch.load(args.load_from_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])

        # freeze the layers
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self._model = model.module.l_to_ab
        self.add_preprocess = lambda x: x[:, :1, :, :]

    def build_model(self):
        pt_model_type = getattr(self.args, 'pt_model', None)
        if pt_model_type == 'deepcluster':
            self.__build_dc_model()
        elif pt_model_type in ['la_cmc']:
            self.__build_la_cmc_model()
        else:
            raise NotImplementedError
        self._criterion = nn.CrossEntropyLoss().cuda()
        if self.args.linear_type == 'default':
            self._linear_pred = RegLog().cuda()
        elif self.args.linear_type == 'mid':
            self._linear_pred = RegLogMid().cuda()
        else:
            raise NotImplementedError('Linear type not supported!')

        self._optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, self._linear_pred.parameters()),
            self.args.init_lr,
            momentum=0.9,
            weight_decay=10**(-4),
        )

    def build_data_provider(self):
        args = self.args
        # data loading code
        traindir = os.path.join(IMAGENET_FOLDER, 'train')
        valdir = os.path.join(IMAGENET_FOLDER, 'val')

        pt_model_type = getattr(self.args, 'pt_model', None)
        if pt_model_type == 'deepcluster':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            transformations_val = [transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize]
            transformations_train = [transforms.Resize(256),
                                     transforms.CenterCrop(256),
                                     transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize]
        elif pt_model_type in ['la_cmc']:
            LAB_MEAN = [
                    (0 + 100) / 2, 
                    (-86.183 + 98.233) / 2, 
                    (-107.857 + 94.478) / 2]
            LAB_STD = [
                    (100 - 0) / 2, 
                    (86.183 + 98.233) / 2, 
                    (107.857 + 94.478) / 2]
            normalize = transforms.Normalize(mean=LAB_MEAN, std=LAB_STD)
            sys.path.append(os.path.expanduser('~/RotLocalAggregation/'))
            from src.datasets.imagenet import RGB2Lab
            transformations_val = [transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   RGB2Lab(),
                                   transforms.ToTensor(),
                                   normalize]
            transformations_train = [transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                     transforms.RandomHorizontalFlip(),
                                     RGB2Lab(),
                                     transforms.ToTensor(),
                                     normalize]
        else:
            raise NotImplementedError

        train_dataset = datasets.ImageFolder(
            traindir,
            transform=transforms.Compose(transformations_train)
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transform=transforms.Compose(transformations_val)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.val_workers)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def save_model(self):
        path = os.path.join(
            self.save_folder,
            'checkpoints',
            'checkpoint_' + str(self._epoch) + '.pth.tar',
        )
        torch.save({
            'epoch': self._epoch,
            'state_dict': self._linear_pred.state_dict(),
            'optimizer' : self._optimizer.state_dict()
        }, path)

    def validate(self):
        args = self.args
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self._model.eval()
        softmax = nn.Softmax(dim=1).cuda()
        end = time.time()
        for i, (input_tensor, target) in enumerate(self.val_loader):
            target = target.cuda()
            input_var = torch.autograd.Variable(
                    input_tensor.cuda(), volatile=True)
            input_var = self.add_preprocess(input_var)
            target_var = torch.autograd.Variable(target, volatile=True)

            output = self._linear_pred(self._model(input_var))

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], input_tensor.size(0))
            top5.update(prec5[0], input_tensor.size(0))
            loss = self._criterion(output, target_var)
            losses.update(loss.item(), input_tensor.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      .format(i, len(self.val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))
        self.save_model()
        return top1.avg, top5.avg, losses.avg

    def train(self):
        args = self.args
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # freeze also batch norm layers
        self._model.eval()

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            #adjust learning rate
            if self.args.linear_type == 'default':
                lr_decay_boundary(
                        self._optimizer, 
                        self._epoch, 
                        args.init_lr)
            else:
                lr_decay_boundary_faster(
                        self._optimizer, 
                        self._epoch, 
                        args.init_lr)

            target = target.cuda()
            input_var = torch.autograd.Variable(input.cuda())
            input_var = self.add_preprocess(input_var)
            target_var = torch.autograd.Variable(target)
            # compute output

            output = self._model(input_var)
            output = self._linear_pred(output)
            loss = self._criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                      .format(
                          self._epoch, i, len(self.train_loader), 
                          batch_time=batch_time,
                          data_time=data_time, 
                          loss=losses, top1=top1, top5=top5))

    def run(self):
        save_folder = os.path.join(MODEL_SAVE_FOLDER, self.args.identifier)
        self.save_folder = save_folder
        os.system('mkdir -p ' + save_folder)
        os.system('mkdir -p ' + os.path.join(save_folder, 'checkpoints'))

        self.build_model()
        self.build_data_provider()

        loss_log = Logger(os.path.join(save_folder, 'loss_log'))
        prec1_log = Logger(os.path.join(save_folder, 'prec1'))
        prec5_log = Logger(os.path.join(save_folder, 'prec5'))

        num_epochs = 160
        if self.args.linear_type == 'mid':
            num_epochs = 70

        for self._epoch in range(num_epochs):
            end = time.time()

            # train for one epoch
            self.train()

            # evaluate on validation set
            prec1, prec5, loss = self.validate()

            loss_log.log(loss)
            prec1_log.log(prec1)
            prec5_log.log(prec5)


def main():
    parser = get_pt_imagenet_transfer_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.set_func, "Must specify set_func"
    args = load_set_func(args)

    transfer_cls = ImageNetTransfer(args)
    transfer_cls.run()


if __name__ == '__main__':
    main()
    pass
