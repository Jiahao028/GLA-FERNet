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
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.autograd import Variable
from model import NetworkCIFAR as Network
from data_loader import get_train_and_valid_loader,get_train_loader,get_test_loader


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='datapath', help='location of the data corpus')
parser.add_argument('--set', type=str, default='RafDb', help='location of the data corpus') #cifar10
parser.add_argument('--batch_size', type=int, default=64, help='batch size')  #  default=96
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')   #default=0.025
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')  # default=20
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_queue = get_train_loader('datapath', args.batch_size, 0, True, True)
  valid_queue = get_test_loader('datapath', args.batch_size, 0, False, True)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  best_acc = 0.0
  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj, train_precision, train_recall, train_f1 = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f, train_precision %f, train_recall %f, train_f1 %f', train_acc, train_precision, train_recall, train_f1)

    valid_acc, valid_obj, valid_precision, valid_recall, valid_f1= infer(valid_queue, model, criterion)
    if valid_acc > best_acc:
        best_acc = valid_acc
    logging.info('valid_acc %f, valid_precision %f, valid_recall %f, valid_f1 %f, best_acc %f', valid_acc, valid_precision, valid_recall, valid_f1, best_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  all_targets = []
  all_predictions = []

  model.train()

  for step, (input, target, indexes) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    # 计算 Precision, Recall, 和 F1-score
    _, predicted = torch.max(logits, 1)  # 获取预测的类别

    all_predictions.extend(predicted.cpu().numpy())  # 转为 numpy 并追加到列表中
    all_targets.extend(target.cpu().numpy())  # 转为 numpy 并追加到列表中

    f1 = f1_score(all_targets, all_predictions, average='macro')
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')


    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f %f %f %f', step, objs.avg, precision, recall, f1, top1.avg, top5.avg)

  return top1.avg, objs.avg, precision, recall, f1


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  all_targets = []
  all_predictions = []

  model.eval()

  for step, (input, target, indexes) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda()

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    # 计算 Precision, Recall, 和 F1-score
    _, predicted = torch.max(logits, 1)  # 获取预测的类别

    all_predictions.extend(predicted.cpu().numpy())  # 转为 numpy 并追加到列表中
    all_targets.extend(target.cpu().numpy())  # 转为 numpy 并追加到列表中

    f1 = f1_score(all_targets, all_predictions, average='macro')
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f %f %f %f', step, objs.avg, precision, recall, f1, top1.avg, top5.avg)

  return top1.avg, objs.avg, precision, recall, f1


if __name__ == '__main__':
  main() 

