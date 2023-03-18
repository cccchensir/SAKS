from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from ScanObjectNN import ScanObjectNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream,Poly1FocalLoss
import sklearn.metrics as metrics
from importlib import import_module

import warnings
from thop import profile
warnings.filterwarnings('ignore')

TRAIN_NAME = __file__.split('.')[0]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='model_cls', metavar='N',
                        help='Model to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--Tmax', type=int, default=250, metavar='N',
                        help='Max iteration number of scheduler. ')
    parser.add_argument('--use_sgd', type=int, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=int, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')

    parser.add_argument('--gpu_idx', type=int, default=[0, 1], nargs='+', help='set < 0 to use CPU')
    parser.add_argument('--proj', type=str, default='modelsV1', help='directory of this experiment')

    args = parser.parse_args()

    return args


def _init_(args):
    if args.name == '':
        args.name = TRAIN_NAME
    if not os.path.exists(args.proj):
        os.makedirs(args.proj)
    if not os.path.exists(args.proj + '/' + args.name):
        os.makedirs(args.proj + '/' + args.name)
    if not os.path.exists(args.proj + '/' + args.name + '/' + 'models'):
        os.makedirs(args.proj + '/' + args.name + '/' + 'models')
    os.system('cp {}.py {}/{}/{}.py.backup'.format(TRAIN_NAME, args.proj, args.name, TRAIN_NAME))
    os.system('cp {}.py {}/{}/{}.py.backup'.format(args.model, args.proj, args.name, args.model))


def train(args, io):
    device = torch.device('cpu' if len(args.gpu_idx) == 0 else 'cuda:{}'.format(args.gpu_idx[0]))
    MODEL = import_module(args.model)

    # colored console output
    green = lambda x: '\033[92m' + x + '\033[0m'
    blue = lambda x: '\033[94m' + x + '\033[0m'

    torch.manual_seed(args.seed)
    if len(args.gpu_idx) < 0:
        io.cprint('Using CPU')
    else:
        io.cprint('Using GPU: {}'.format(args.gpu_idx))
        torch.cuda.manual_seed(args.seed)

    # Load data
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points), num_workers=6,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=6,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    # Try to load models
    io.cprint('Using model: {}'.format(args.model))
    model = MODEL.Net(args,15).to(device)
    model = nn.DataParallel(model, device_ids=args.gpu_idx)
    print(str(model))
    para = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model parameter number: {}".format(para))

    # model = nn.DataParallel(model)
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    parameters = [{'params': (p for name, p in model.named_parameters() if 'bias' not in name), 'weight_decay': 0.05},
                  {'params': (p for name, p in model.named_parameters() if 'bias' in name)}]

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(parameters, lr=args.lr * 100, momentum=args.momentum)
    else:
        print("Use Adamw")
        opt = optim.AdamW(parameters, lr=args.lr)

    scheduler = CosineAnnealingLR(opt, args.Tmax, eta_min=0.0001)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        if epoch < args.Tmax:
            scheduler.step()
        elif epoch == args.Tmax:
            for group in opt.param_groups:
                group['lr'] = 0.0001

        learning_rate = opt.param_groups[0]['lr']
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits, corr_loss = model(data)
            loss = criterion(logits, label)
            loss1 = loss + 30 * torch.mean(corr_loss)
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
            opt.step()
            model.zero_grad()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint('EPOCH #{}  lr = {}'.format(epoch, learning_rate))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                logits, _ = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss * 1.0 / count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '%s/%s/models/model.t7' % (args.proj, args.name))
            io.cprint('Current best saved in: {}'.format(
                '********** %s/%s/models/model.t7 **********' % (args.proj, args.name)))


def test(args, io):
    MODEL = import_module(args.model)
    device = torch.device('cpu' if len(args.gpu_idx) == 0 else 'cuda:{}'.format(args.gpu_idx[0]))

    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points), num_workers=6,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    io.cprint('********** TEST STAGE **********')
    io.cprint('Reload best epoch:')

    # Try to load models
    model = MODEL.Net(args,15).to(device)
    model = nn.DataParallel(model, device_ids=args.gpu_idx)
    model.load_state_dict(torch.load('%s/%s/models/model.t7' % (args.proj, args.name)))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            logits, _ = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


def computation(args, io):

    MODEL = import_module(args.model)

    model = MODEL.Net(args).to('cuda')
    input = torch.zeros((1,3,1024)).to('cuda')
    flops, params = profile(model,inputs=(input,))
    print(flops,params)


if __name__ == "__main__":
    args = parse_arguments()

    _init_(args)

    io = IOStream(args.proj + '/' + args.name + '/train.log')
    io.cprint(str(args))

    #if not args.eval:
        #train(args, io)
    #else:
        #test(args, io)
    computation(args, io)