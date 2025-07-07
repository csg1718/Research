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
from model.TANet_vis import TANet

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='RANet')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--root', type=str, default='../../dataset')
parser.add_argument('--dataset', type=str, default='RAVEN')
parser.add_argument('--fig_type', type=str, default='all')
parser.add_argument('--img_size', type=int, default=160)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--save', type=str, default='./results/checkpoint/')
parser.add_argument('--train_once', type=bool, default=False)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--best_epoch', type=int)

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

model = TANet()
model.to(device)

save_name = args.model_name + '_' + args.dataset + '_' + args.fig_type

save_path_model = os.path.join(args.save, save_name)

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



if __name__ == '__main__':
    model.load_state_dict(torch.load(save_path_model+'/model_'+str(args.best_epoch)+'.pth'))
    model.eval()

    sim_list = []
    test_loader_iter = iter(test_loader)
    for _ in trange(len(test_loader_iter)):
        image, target = next(test_loader_iter)

        image = image.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(image)

            mid_feature = output[-1]
            first_row, second_row = mid_feature[:,0], mid_feature[:,1]
            cos_sim = F.cosine_similarity(first_row, second_row, dim=1)

            similarity = cos_sim.mean().item()
            
            sim_list.append(similarity)
    
    avg_sim = sum(sim_list) / len(sim_list)
    print(f'similarity: {avg_sim}')


