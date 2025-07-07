import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import torch.nn as nn
import random

from tqdm import trange
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable

from dataset_utility import dataset, ToTensor
from logging_utility import logwrapper, plotwrapper
from model.RANet import RANet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser()

#train args
parser.add_argument('--model_name', type=str, default='RANet')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--root', type=str, default='../../dataset')
parser.add_argument('--dataset', type=str, default='RAVEN')
parser.add_argument('--fig_type', type=str, default='all')
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--save', type=str, default='./results/checkpoint/')
parser.add_argument('--log_plot', type=str, default='./results/log_plot/')
parser.add_argument('--log_txt', type=str, default='./results/log_txt/')
parser.add_argument('--train_once', type=bool, default=False)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--att_term', type=float, default=6e-5)
parser.add_argument('--decay', type=float, default=1e-5)
parser.add_argument('--patience', type=int, default=50)

#model args
parser.add_argument('--image_size', type=int, default=160)
parser.add_argument('--feat_channel', type=int, default=32)
parser.add_argument('--cnn_layers', type=int, default=10)
parser.add_argument('--reasoning_dims', type=list, default=[64,32,5])
parser.add_argument('--ffn_divider', type=int, default=4)

args = parser.parse_args()

if args.fig_type == 'all':
    args.train_once = True
else:
    args.train_once = False

if args.dataset == 'PGM':
    args.train_once = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device: {device}')
torch.manual_seed(args.seed)
if torch.cuda.is_available:
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)
if not os.path.exists(args.log_plot):
    os.makedirs(args.log_plot)
if not os.path.exists(args.log_txt):
    os.makedirs(args.log_txt)

log = logwrapper(args.log_plot)

model = RANet()
model.to(device)

total_params = sum(p.numel() for p in model.parameters())

print(f'Total number of parameters: {total_params}')

save_name = args.model_name + '_' + args.dataset + '_' + args.fig_type

save_path_model = os.path.join(args.save, save_name)

if not os.path.exists(save_path_model):
    os.makedirs(save_path_model)

tf = transforms.Compose([ToTensor()])    

train_set = dataset(os.path.join(args.root, args.dataset), 'train', args.fig_type, args.img_size, tf, args.train_once)
valid_set = dataset(os.path.join(args.root, args.dataset), 'val', args.fig_type, args.img_size, tf, args.train_once)
test_set = dataset(os.path.join(args.root, args.dataset), 'test', args.fig_type, args.img_size, tf, args.train_once)

print('train length', len(train_set), args.fig_type)
print('validation length', len(valid_set), args.fig_type)
print('test length', len(test_set), args.fig_type)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

time_now = datetime.now().strftime('%D-%H:%M:%S')  
save_log_name = os.path.join(args.log_txt, 'log_{:s}.txt'.format(save_name)) 
with open(save_log_name, 'a') as f:
    f.write('\n------ lr: {:f}, batch_size: {:d}, img_size: {:d}, time: {:s} ------\n'.format(
        args.lr, args.batch_size, args.img_size, time_now))
f.close() 

def train(epoch):
    model.train()
    metrics = {'loss': [], 'correct': [], 'count': [], 'att_loss': []}

    train_loader_iter = iter(train_loader)
    for _ in trange(len(train_loader_iter)):
        image, target = next(train_loader_iter)

        image = image.to(device)
        target = target.to(device)

        predict, maps = model(image)

        loss = F.cross_entropy(predict, target)
        att_sum = maps.sum() / args.batch_size
        total_loss = loss + args.att_term * att_sum

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pred = torch.max(predict, 1)[1]
        correct = pred.eq(target.data).cpu().sum().numpy()

        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        metrics['loss'].append(loss.item())
        metrics['att_loss'].append(att_sum.item())
    
    accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 
    print ('Epoch: {:d}/{:d},  CELoss: {:.3f}, AttLoss: {:.3f}, Accuracy: {:.3f}'.format(epoch, args.epochs, np.mean(metrics['loss']), np.mean(metrics['att_loss']) * args.att_term, accuracy))

    return metrics

def valid(epoch):
    model.eval()
    metrics = {'loss': [], 'correct': [], 'count': [], 'att_loss': []}

    valid_loader_iter = iter(valid_loader)
    for _ in trange(len(valid_loader_iter)):
        image, target = next(valid_loader_iter)

        image = image.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            predict, maps = model(image)

            att_sum = maps.sum() / args.batch_size
            loss = F.cross_entropy(predict, target)
            total_loss = loss + args.att_term * att_sum
           
        pred = torch.max(predict, 1)[1]

        correct = pred.eq(target.data).cpu().sum().numpy()

        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        metrics['loss'].append(loss.item())
        metrics['att_loss'].append(att_sum.item())

    accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 

    print ('Validaition Epoch: {:d}/{:d}, CELoss: {:.3f}, AttLoss: {:.3f}, Accuracy: {:.3f}'.format(epoch, args.epochs, np.mean(metrics['loss']), np.mean(metrics['att_loss']) * args.att_term, accuracy))
            
    return metrics

def test(epoch):
    model.eval()
    metrics = {'loss': [], 'correct': [], 'count': [], 'att_loss': []}

    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter)):
        image, target = next(test_loader_iter)

        image = image.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            predict, maps = model(image)

            att_sum = maps.sum() / args.batch_size
            loss = F.cross_entropy(predict, target)
            total_loss = loss + args.att_term * att_sum
           
        pred = torch.max(predict, 1)[1]

        correct = pred.eq(target.data).cpu().sum().numpy()

        metrics['correct'].append(correct)
        metrics['count'].append(target.size(0))
        metrics['loss'].append(loss.item())
        metrics['att_loss'].append(att_sum.item())

    accuracy = 100 * np.sum(metrics['correct']) / np.sum(metrics['count']) 

    print ('Testing Epoch: {:d}/{:d}, CELoss: {:.3f}, AttLoss: {:.3f}, Accuracy: {:.3f} \n'.format(epoch, args.epochs, np.mean(metrics['loss']), np.mean(metrics['att_loss']) * args.att_term, accuracy))
            
    return metrics

if __name__ == '__main__':
    min_loss_valid = float('inf')
    patience = args.patience
    patience_counter = 0

    for epoch in range(1, args.epochs+1):
        metrics_train = train(epoch)

        # Save model
        if epoch > 0:
            save_name = os.path.join(save_path_model, 'model_{:02d}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)
        
        metrics_valid = valid(epoch)
        metrics_test = test(epoch)

        loss_train = np.mean(metrics_train['loss'])
        loss_valid = np.mean(metrics_valid['loss'])
        loss_test = np.mean(metrics_test['loss'])
        acc_train = 100 * np.sum(metrics_train['correct']) / np.sum(metrics_train['count'])
        acc_valid = 100 * np.sum(metrics_valid['correct']) / np.sum(metrics_valid['count'])
        acc_test = 100 * np.sum(metrics_test['correct']) / np.sum(metrics_test['count'])

        loss = {'train': loss_train, 'valid': loss_valid, 'test':loss_test}
        acc = {'train': acc_train, 'valid': acc_valid, 'test':acc_test}
        log_name = f'{args.model_name}_{args.dataset}_{args.fig_type}_Loss'
        log.write_scalars(log_name, loss, epoch)
        log_name = f'{args.model_name}_{args.dataset}_{args.fig_type}_Acc'
        log.write_scalars(log_name, acc, epoch)

        with open(save_log_name, 'a') as f:
            f.write('Epoch: {:02d}\n'.format(epoch))
            f.write('Train Accuracy: {:.3f}, Loss: {:.3f}\n'.format(acc_train, loss_train))
            f.write('Test Accuracy: {:.3f}, Loss: {:.3f}\n'.format(acc_test, loss_test))

        f.close()

        if loss_valid < min_loss_valid:
            min_loss_valid = loss_valid
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping after {} epochs without improvement in validation loss.".format(patience))
            break