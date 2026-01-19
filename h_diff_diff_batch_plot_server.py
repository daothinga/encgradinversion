
import argparse
import os
import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt
import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
from torchmetrics import TotalVariation

tv = TotalVariation()

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

import torchsummary

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--dataset', type=str, default="CIFAR100",
                    choices=["CIFAR100", "LFWPeople", "ImageNet"],
                    help='dataset to use.')
parser.add_argument('--model', type=str, default="LeNet",
                    choices=["LeNet", "ResNet18", "EfficientNet0"],
                    help='model to use.')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size for gradient sharing.')
parser.add_argument('--index', type=str, default="[25, 30, 181, 50, 28, 34, 10, 2, 3, 5, 91, 100, 90, 18, 56, 8, 102, 202, 300, 400, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]",
                    help='the indices for leaking images on CIFAR.')
parser.add_argument('--image', type=str,default="",
                    help='the path to customized image.')
args = parser.parse_args()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)


# index_list = map(int, args.index.strip('[]').split(','))
# index_list = [int(i) for i in index_list]

# index_list = np.arange(256)

dataset = args.dataset

if dataset == 'CIFAR100':
    dst = datasets.CIFAR100("~/.torch", download=True)
    n_images = 5000
    n_classes = 100
    w = 32

if dataset == 'LFWPeople':
    from torch.utils import data

    class LFWDeepFunneledDataset(data.Dataset):
        def __init__(self, root):
            self.root = root
            self.samples = []
            self.class_to_idx = {}
            for idx, name in enumerate(sorted(os.listdir(root))):
                dir_path = os.path.join(root, name)
                if not os.path.isdir(dir_path):
                    continue
                self.class_to_idx[name] = idx
                for fname in os.listdir(dir_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(dir_path, fname), idx))

        def __getitem__(self, index):
            path, label = self.samples[index]
            image = Image.open(path).convert("RGB")
            return image, label

        def __len__(self):
            return len(self.samples)

    lfw_root = os.path.join("archive", "lfw-deepfunneled", "lfw-deepfunneled")
    dst = LFWDeepFunneledDataset(lfw_root)
    n_images = min(1000, len(dst))
    n_classes = len(dst.class_to_idx)
    # w = 128
    w = 32



if dataset == 'ImageNet':
    from torch.utils import data
    import os

    class simpleDataset(data.Dataset):
        
        # initialise function of class
        def __init__(self, root, filenames, labels):
            # the data directory 
            self.root = root
            # the list of filename
            self.filenames = filenames
            # the list of label
            self.labels = labels

        # obtain the sample with the given index
        def __getitem__(self, index):
            # obtain filenames from list
            image_filename = self.filenames[index]
            # Load data and label
            image = Image.open(os.path.join(self.root, image_filename))
            label = self.labels[index]
            
            # output of Dataset must be tensor
            image = transforms.ToTensor()(image)
            label = torch.as_tensor(label, dtype=torch.int64)

            image = tt(image)   # Nga added
            return image, label
        
        # the total number of samples (optional)
        def __len__(self):
            return len(self.filenames)
        
    root = "Imagenet"

    # assume we have 3 jpg images
    # filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    filenames = os.listdir("Imagenet/")
    # print(filenames)

    # the class of image might be ['black cat', 'tabby cat', 'tabby cat']
    labels = np.arange(1000)

    dst = simpleDataset(root=root,
                           filenames=filenames,
                           labels=labels
                           )

    n_images = 1000
    n_classes = 1000
    w = 32

print(dst)

tp = transforms.Compose([
    transforms.Resize(w),
    transforms.CenterCrop(w),
    transforms.ToTensor()
])


# n_images = 5000
n_test_images = 500
n_batch_images = args.batch

# index_list = np.random.randint(low=0,high=50000,size=(n_images,))
training_index_list = np.arange(n_test_images, n_images)
# test_index_list = np.arange(n_test_images)



if dataset == 'ImageNet':
    training_index_list = np.arange(0, 500)

print('n_images: ', n_images, n_batch_images)

# model = 'ResNet18'
model = args.model
# model = 'EfficientNet0'


tp = transforms.Compose([
    transforms.Resize(w),
    transforms.CenterCrop(w),
    transforms.ToTensor()
])
tt = transforms.ToPILImage()



from models.vision import weights_init, CFN, CCNN, LeNet, ResNet18, EfficinetNet0


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

# net.body.register_forward_hook(get_activation('body'))  # CCNN
if model == 'LeNet':
    n_h = 12*w*w//16  #Hidden units in LeNet 
    net = LeNet(num_classes=n_classes,n_h = n_h).to(device)
    net.conv4.register_forward_hook(get_activation('conv4'))   # LeNet

if model == 'ResNet18':
    n_h = 24*w*w//16  #Hidden units in ResNet18
    net = ResNet18(n_classes, n_h).to(device)
    net.apply(weights_init)
    net.layer4.register_forward_hook(get_activation('layer4'))  # ResNet18()    
    
if model == 'EfficientNet0':
    n_h = 1280
    net = EfficinetNet0("b0", num_classses=100).to(device)
    print('This is EfficientNet b0')
    net.features.register_forward_hook(get_activation('features'))  


# torchsummary.summary(net, (3, 32, 32))
# torch.manual_seed(1234)


gt_data = []
gt_label = []
gt_onehot_label = []


##### Store training data here
for i in range(n_images-n_test_images):
    tmp = tp(dst[training_index_list[i]][0]).to(device)
    # print(min(tmp), max(tmp))
    # print(tmp.size())
    gt_data.append(tmp.view(1,*tmp.size()))

    tmp_label = torch.Tensor([dst[training_index_list[i]][1]]).long().to(device)
    tmp_label = tmp_label.view(1,)
    # print(tmp_label)
    
    # assert tmp_label not in gt_label
    
    gt_label.append(tmp_label)

    gt_onehot_label.append(label_to_onehot(tmp_label, num_classes=n_classes))



m = torch.nn.AvgPool2d(kernel_size = 4)
pool = nn.AdaptiveAvgPool2d(1)

criterion = cross_entropy_for_onehot

optimizer1 = torch.optim.Adam(net.parameters())   # for training at victim side



# compute original gradient 
h_diff1 = []        # Method 1
h_diff2 = []        # Method 2
duplicate_label = []
n_epochs = 20
acc = []
for epoch in range(n_epochs):
    if epoch%100 == 0:
        print('epoch ', epoch)
    optimizer1.zero_grad()


    

    if dataset == 'ImageNet':
        rand_list = np.random.randint(low=0,high=100, size=(n_batch_images))
    else:
        rand_list = np.random.randint(low=0,high=n_images-n_test_images, size=(n_batch_images))
    
    print(rand_list)
    label = [gt_label[k] for k in rand_list]
    dupes = [x for n, x in enumerate(label) if x in label[:n]]
    duplicate_label.append(len(dupes)*10**(-4))

    ########### Extract real hidden layer
    pred_list = []
    y = []
    h_original = []

    for i in range(n_batch_images):
        index_i = rand_list[i]
        pred = net(gt_data[index_i])
        if model == 'LeNet':
            h_ = activation['conv4']   # LeNet
            
        if model == 'ResNet18':
            h_ = activation['layer4']    # Resnet18
            h_ = m(h_)

        
        if model == 'EfficientNet0':
            tmp = activation['features']
            # h_ = pool(tmp)
            h_ = tmp


        # print('nh: ', n_h)
        h_ = torch.reshape(h_, (1, n_h))
        h_original.append(h_)
        
        pred_list.append(net(gt_data[index_i]))
    
        y_ = criterion(pred, gt_onehot_label[index_i])
        y.append(y_)

    dy_dx_list = []
    dy_dx = ()
    for i in range(n_batch_images):
        index_i = rand_list[i]
        dy_dx_ = torch.autograd.grad(y[i], net.parameters(), retain_graph = True)
        # print(len(dy_dx_))
        dy_dx_list.append(dy_dx_)
        if i == 0:
            dy_dx += dy_dx_
        else:
            dy_dx = tuple(map(lambda x, y: x + y, dy_dx, dy_dx_))
        
        # if epoch%10 == 0:
        #     print(dy_dx_[-1])

    dy_dx = tuple(map(lambda x: x/n_batch_images,dy_dx)) # take average gradient

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    # print(original_dy_dx[-1])


    optimizer1.zero_grad()
    for i in range(n_batch_images):
        y[i].backward()
    optimizer1.step()


    ######## Try to recover hidden layer before softmax
    h_list1 = []
    for i in range(n_batch_images):
        index_i = rand_list[i]
        num = original_dy_dx[-2][gt_label[index_i],:]
        denum  = original_dy_dx[-1][gt_label[index_i]]
        # print(num)
        # print(denum)
        h_ = torch.tensor(torch.div(num, denum), dtype=torch.float64)
        h_list1.append(h_)


    
    # Compare between real and recovered hidden layers
    diff1_ = 0
    for i in range(n_batch_images):
        diff1_ += ((h_list1[i] - h_original[i])**2).mean()
       
    diff1_ = diff1_/n_batch_images

    h_diff1.append(diff1_)
    print(diff1_)


    # Use an image to estimate gradient at last layer
    index_i = 3
    pred = net(gt_data[index_i])
    if model == 'LeNet':
        h_ = activation['conv4']   # LeNet
        
    if model == 'ResNet18':
        h_ = activation['layer4']    # Resnet18
        h_ = m(h_)

    h_ = torch.reshape(h_, (1, n_h))
    y_ = criterion(pred, gt_onehot_label[index_i])

    dy_dx_ = torch.autograd.grad(y_, net.parameters(), retain_graph = True)
    tmp = dy_dx_[-1].detach().cpu().numpy()
    dy_dx_last = [i for i in tmp if i>0]


    mean_ = sum(dy_dx_last)/(n_classes-1)

    
    orig_last = original_dy_dx[-1].detach().cpu().numpy()
    orig_fc = original_dy_dx[-2].detach().cpu().numpy()

    h_list2 = []
    for j in range(n_h):
    # for j in range(2):
        a = np.zeros((n_batch_images, n_batch_images)) + mean_
        b = []
        for k in range(n_batch_images):
            index_k = rand_list[k]
            label_k = int(gt_label[index_k].item())
            a[k,k] = n_batch_images*orig_last[label_k] - mean_*(n_batch_images-1)
            b.append(n_batch_images*orig_fc[label_k, j])

        h_ = np.linalg.solve(a,b)
        h_list2.append(h_)

        
    
    

    h_list_ = h_list2
    h_list_ = np.moveaxis(h_list_, 0, -1)

    h_list2 = []
    for i in range(n_batch_images):
        h_list2.append(torch.tensor(h_list_[i:i+1], dtype=torch.float64, device=device))




    
    # Compare between real and recovered hidden layers
    diff2_ = 0
    for i in range(n_batch_images):
        diff2_ += ((h_list2[i] - h_original[i])**2).mean()
        # print(h_list[i]) #, np.mean(h_original[i]))
    
    diff2_ = diff2_/n_batch_images

    h_diff2.append(diff2_)
    print(diff2_)

print('Mean of diff1: ', sum(h_diff1)/n_epochs)
print('Mean of diff2: ', sum(h_diff2)/n_epochs)

h_diff1_cpu = [float(x.detach().cpu()) for x in h_diff1]
h_diff2_cpu = [float(x.detach().cpu()) for x in h_diff2]
mean_diff1 = float(sum(h_diff1_cpu)/len(h_diff1_cpu))
mean_diff2 = float(sum(h_diff2_cpu)/len(h_diff2_cpu))

# csfont = {'fontname': 'Times New Roman'}
plt.rcParams['font.family'] = 'Times New Roman'   # Set font type globally

plt.plot(range(1,n_epochs+1), h_diff1_cpu, 'bo--', label = 'OHA')
plt.plot(range(1,n_epochs+1), h_diff2_cpu, 'g*--', label = 'GRA')
plt.xticks([0, 5, 10, 15, 20], ['0','5', '10', '15', '20'])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.xlabel('Training epochs', fontsize=20)
plt.ylabel('MSE of Image Encoding Reconstruction', fontsize=20)
plt.yscale("log")
f_name = 'B_' + str(n_batch_images) + '_' + dataset + '.eps'
plt.savefig(f_name, bbox_inches='tight')
plt.show()

np.savetxt('h_diff1.txt', h_diff1_cpu)
np.savetxt('h_diff2.txt', h_diff2_cpu)

# Append summary row for table (dataset, batch, mean_diff1, mean_diff2)
csv_path = 'table_ohagra.csv'
write_header = not os.path.exists(csv_path)
with open(csv_path, 'a') as f:
    if write_header:
        f.write('dataset,batch,mean_diff1,mean_diff2\n')
    f.write(f'{dataset},{n_batch_images},{mean_diff1},{mean_diff2}\n')

# """
