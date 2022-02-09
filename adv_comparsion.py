from deform_models.resnet18_deform_bottleneck import *
from deform_models.vgg_deform_bottleneck import *
from deform_models.vgg_deform_branch import *
from deform_models.vgg_deform_circle import *
from deform_models.resnet18 import *
from deform_models.vgg import *
import os
import sys
import glob
import numpy as np
import torch
import logging
import argparse
import utils
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import time
import re
import random
import hiddenlayer as hl
from torchvision import models
from distillation import Linf_PGD
from torchviz import make_dot

# model_type = ["vgg16", "deform_vgg16_bottleneck"]
localtime = time.asctime( time.localtime(time.time()))
x = re.split(r"[\s,(:)]",localtime)
default_EXP = " ".join(x[1:-1])
parser = argparse.ArgumentParser("NAT")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=10, help='test frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--save', type=str, default=default_EXP, help='experiment name')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.8, help='data portion for training weights')
parser.add_argument('--prefix', type=str, default='.', help='parent save path')
parser.add_argument('--model', type=str, default='vgg16', help='model type')
parser.add_argument('--store', type=int, default=1, help='Whether to store the model')
parser.add_argument('--eps', type=int, default=0.08, help='eps')
parser.add_argument('--scheduler', type=str, default='naive_cosine', help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.01, help='min learning rate')
args = parser.parse_args()
args.cutout = False
if not os.path.exists(args.prefix):
    os.makedirs(args.prefix)
args.save = os.path.join(args.prefix, "EXP", args.save)
args.cutout = False
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')+glob.glob('*.sh')+glob.glob('*.yml'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()
CIFAR_CLASSES = 10

def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info('GPU device = %d' % args.gpu)
    else:
        logging.info('no GPU available, use CPU!!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s" % args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    original_model = vgg16()
    deform_model = vgg16_deform_circle()

    x = torch.zeros(1, 3, 32, 32, dtype=torch.float, requires_grad=False)
    # net_out = original_model(x)
    # original_struct = make_dot(net_out)  # plot graph of variable, not of a nn.Module
    # original_struct.render("original_net_struct", view=False, directory=args.save)
    # net_out = deform_model(x)
    # deform_struct = make_dot(net_out)  # plot graph of variable, not of a nn.Module
    # deform_struct.render("deform_net_struct", view=False, directory=args.save)
    original_graph=hl.build_graph(original_model,x)
    original_graph.theme=hl.graph.THEMES['blue'].copy()
    original_graph.save(path=os.path.join(args.save,'{}.png'.format(original_model.__class__.__name__)),format='png')
    deform_graph=hl.build_graph(deform_model,x)
    deform_graph.theme=hl.graph.THEMES['blue'].copy()
    deform_graph.save(path=os.path.join(args.save,'{}.png'.format(deform_model.__class__.__name__)),format='png')


    original_model_optimizer = torch.optim.SGD(
        original_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    deform_model_optimizer = torch.optim.SGD(
        deform_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    random.shuffle(indices)
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4
    )

    valid_arch_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2
    )

    if args.scheduler == "naive_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            original_model_optimizer, float(args.epochs), eta_min=args.learning_rate_min
        )
    else:
        assert False, "unsupported scheduler type: %s" % args.scheduler

    original_model.to(device)
    deform_model.to(device)
    # 根据需求对比模型

    logging.info("comparing {} and {}".format(original_model.__class__.__name__, deform_model.__class__.__name__))
    logging.info("original param size = %fMB", utils.count_parameters_in_MB(original_model))
    logging.info("deform param size = %fMB", utils.count_parameters_in_MB(deform_model))
    original_model._model_optimizer = original_model_optimizer

    deform_model._model_optimizer = deform_model_optimizer

    

    for epoch in range(args.epochs+1):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        logging.info('Updating Original Parameters')
        update_w(train_queue, original_model, device, criterion)
        logging.info('Updating Deform Parameters')
        update_w(train_queue, deform_model, device, criterion)
        scheduler.step()
        

        if (epoch) % args.test_freq == 0:
            logging.info('Testing Original Model')
            acc1 = test_acc(valid_arch_queue, original_model, device)
            logging.info('Original Model Accuracy: {} Epoch:{}'.format(acc1, epoch))
            logging.info('Testing Deform Model')
            acc2 = test_acc(valid_arch_queue, original_model, device)
            logging.info('Deform Model Accuracy: {} Epoch:{}'.format(acc2, epoch))

    # save model
    if args.store == 1:
        utils.save(original_model, os.path.join(args.save, '{}_models.pt'.format(original_model.__class__.__name__)))
        utils.save(deform_model, os.path.join(args.save, '{}_models.pt'.format(deform_model.__class__.__name__)))

    logging.info('Training Complete')
    logging.info("Generating Adversial Examples")
    
    top1_original_clean = utils.AvgrageMeter()
    top1_deform_clean = utils.AvgrageMeter()
    top1_original_adv = utils.AvgrageMeter()
    top1_deform_adv = utils.AvgrageMeter()

    for step, (input, target) in enumerate(valid_arch_queue):
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)
        input_adv = Linf_PGD(deform_model, input, target, eps = args.eps, alpha=args.eps/10, steps=10)

        logits = original_model(input)
        acc_clean_original = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
        top1_original_clean.update(acc_clean_original.item(), n)

        logits = deform_model(input)
        acc_clean_deform = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
        top1_deform_clean.update(acc_clean_deform.item(), n)

        logits = original_model(input_adv)
        acc_adv_original = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
        top1_original_adv.update(acc_adv_original.item(), n)

        logits = deform_model(input_adv)
        acc_adv_deform = utils.accuracy(logits, target, topk=(1, 5))[0] / 100.0
        top1_deform_adv.update(acc_adv_deform.item(), n)

        if step % args.report_freq == 0:
            logging.info('Adversial Transferbility: Step=%03d acc_clean_original=%f acc_clean_deform=%f acc_adv_original=%f acc_adv_deform=%f',
                         step, top1_original_clean.avg, top1_deform_clean.avg, top1_original_adv.avg, top1_deform_adv.avg)

def update_w(train_queue, model, device, criterion):
    objs = utils.AvgrageMeter()
    normal_ent = utils.AvgrageMeter()
    reduce_ent = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)
        model._model_optimizer.zero_grad()
        # logits, loss = train(model, input, target)
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        model._model_optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('Updating W: Step=%03d Loss=%e Top1=%f Top5=%f',
                         step, objs.avg, top1.avg, top5.avg)

def test_acc(test_queue, model, device):
    # go over all the testing data to obtain the accuracy
    top1 = utils.AvgrageMeter()

    model.eval()

    for step, (test_input, test_target) in enumerate(test_queue):
        test_input = test_input.to(device)
        test_target = test_target.to(device)
        n = test_input.size(0)
        logits = model(test_input)
        accuracy = utils.accuracy(logits, test_target)[0]
        top1.update(accuracy.item(), n)
        if step > 20:
            break
    return top1.avg

if __name__ == '__main__':
    main()