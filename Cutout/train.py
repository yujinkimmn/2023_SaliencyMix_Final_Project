# This train code is used for make models of Cutout and baseline.

# train command example
# For make baseline(only traditional data augmentation appied) model,
# nohup python Cutout/train.py --dataset cifar10 --model wideresnet  --data_augmentation > /home/pbl9/group10/results/logs/cifar10_wideresnet_baseline_new.log 2>&1 &
# nohup python Cutout/train.py --dataset cifar10 --model resnet50 --data_augmentation > /home/pbl9/group10/results/logs/cifar10_resnet50_baseline_new.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python Cutout/train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout > /home/pbl9/group10/results/logs/cifar10_resnet18_cutout_new.log 2>&1 &


import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18, ResNet50  # modify here to add ResNet50
from model.wide_resnet import WideResNet

# add 'resnet50' here for our experiment
model_options = ['resnet18', 'wideresnet', 'resnet50']
dataset_options = ['cifar10', 'cifar100', 'svhn']

# this is for train command options
parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# path to save the results
test_id = args.dataset + '_' + args.model + '_cutout_new'

print(args)

############################################  Image Preprocessing  ############################################
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
# Normalize for CIFAR-10
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
# add augmentation when data_augmentation is set to True
if args.data_augmentation:
    # apply cropping and flipping (traditional data augmentation)
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
# transform to tensor
train_transform.transforms.append(transforms.ToTensor())
# add normalize
train_transform.transforms.append(normalize)

# add cutout when cutout is set to True
if args.cutout:
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

############################################ Load dataset ############################################

# download data depending on the dataset that we want to use
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)

# Data Loader (Input Pipeline)
# apply our batch size and shuffle data here
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)



############################################ Load models ############################################

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
# Added here for resnet50
elif args.model == 'resnet50':
    cnn = ResNet50(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)


############################################ Define loss function and optimizer ############################################

# load model to GPU
cnn = cnn.cuda()
# define loss function
criterion = nn.CrossEntropyLoss().cuda()
# define optimizer 
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)


############################################ Set learning rate scheduler ############################################

# set learning rate scheduler option
if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    
############################################ Training ############################################

# to save the logs to the path 
filename = '/home/pbl9/group10/results/csv_results/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

# test function to calculate test accuracy for each epoch
def test(loader):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        # set images and labels on cuda 
        images = images.cuda()
        labels = labels.cuda()

        # get the result of test set
        with torch.no_grad():
            pred = cnn(images)

        # classified label of our model
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        # Add to correct if the result is right with the real label
        correct += (pred == labels).sum().item()

    # Calculate validation accuracy 
    val_acc = correct / total
    # Change to train mode to continue next train 
    cnn.train()
    return val_acc

# Training loop
for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    # to show process of training visually 
    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        # Make images and labels on gpu 
        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        # Get output from model 
        pred = cnn(images)  

        # Calculate loss with loss function and backward pass
        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        # Accumulate loss
        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    # Calculate test accuracy 
    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    #scheduler.step(epoch)  # Use this line for PyTorch <1.4
    scheduler.step()     # Use this line for PyTorch >=1.4 
                        # we used pytorch 2.0.1 so the second option is used 

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

# Save the checkpoint 
torch.save(cnn.state_dict(), '/home/pbl9/group10/results/checkpoints/' + test_id + '.pt')
csv_logger.close()
