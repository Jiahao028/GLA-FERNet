import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from sklearn.metrics import f1_score
from torch.autograd import Variable
from model_search_softmax_sogmoid import Network  # GAL-FERNet search softmax and sogmoid
#   from model_search_ran_softmax_sogmoid_k4 import Network  # GAL-FERNet Random search softmax and sogmoid


from architect import Architect
from data_loader import get_train_and_valid_loader,get_train_loader,get_test_loader,RafDataSet


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--set', type=str, default='RafDb', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size') #default=256
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')   #default=0.1
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=2, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs') #default=50
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')  #default=8
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 7

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_queue = get_train_loader('datapath', args.batch_size, 0, True, True)
  valid_queue = get_test_loader('datapath', args.batch_size, 0, False, True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  best_acc = 0.0
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj, train_f1 = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f; train_f1 %f', train_acc, train_f1)

    # validation
    # if args.epochs-epoch<=1:
    #   valid_acc, valid_obj = infer(valid_queue, model, criterion)
    #   logging.info('valid_acc %f', valid_acc)
    with torch.no_grad():
        # validation
        valid_acc, valid_obj, valid_f1 = infer(valid_queue, model, criterion)
        # logging.info('valid_acc %f', valid_acc)
        if valid_acc > best_acc:
          best_acc = valid_acc
        logging.info('valid_acc %f, valid_f1 %f best_acc %f', valid_acc, valid_f1, best_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  all_targets = []  # 用于存储所有目标值
  all_predictions = []  # 用于存储所有预测值
  # for step, (input, target) in enumerate(train_queue):
  for step, (input, target, indexes) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search, indexes = next(iter(valid_queue))   #    input_search, target_search = next(iter(valid_queue))
    #try:
    #  input_search, target_search = next(valid_queue_iter)
    #except:
    #  valid_queue_iter = iter(valid_queue)
    #  input_search, target_search = next(valid_queue_iter)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    if epoch>=15:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    # 收集预测值和目标值
    _, predicted = torch.max(logits, 1)  # 获取每个样本的预测类别
    all_predictions.extend(predicted.cpu().numpy())  # 转为 numpy 并追加到列表中
    all_targets.extend(target.cpu().numpy())  # 转为 numpy 并追加到列表中
    f1 = f1_score(all_targets, all_predictions, average='macro')


    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f %f', step, objs.avg, f1, top1.avg, top5.avg)

  return top1.avg, objs.avg, f1


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  all_targets = []  # 用于存储所有目标值
  all_predictions = []  # 用于存储所有预测值

  # for step, (input, target) in enumerate(valid_queue):
  for step, (input, target, indexes) in enumerate(valid_queue):
    #input = input.cuda()
    #target = target.cuda(non_blocking=True)
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()
    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    _, predicted = torch.max(logits, 1)  # 获取每个样本的预测类别
    all_predictions.extend(predicted.cpu().numpy())  # 转为 numpy 并追加到列表中
    all_targets.extend(target.cpu().numpy())  # 转为 numpy 并追加到列表中
    f1 = f1_score(all_targets, all_predictions, average='macro')

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f %f', step, objs.avg, f1, top1.avg, top5.avg)

  return top1.avg, objs.avg, f1


if __name__ == '__main__':
  main() 

