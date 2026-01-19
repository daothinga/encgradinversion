# Check performance with different coefficient values

import argparse
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchmetrics.image import TotalVariation

import skimage
from skimage import metrics

from numpy.linalg import norm

import time

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True)
# LPIPS needs the images to be in the [-1, 1] range.




device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda"
print("Running on %s" % device)

tv = TotalVariation().to(device)

print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

model = 'LeNet'
dataset = 'ImageNet'
method_IR = 'Ours'

# LeNet
a_g = [0.1]
a_h = [10**3]
a_tv = [10**(-6)]
a_norm = [10**(-9)]

a_g_str = ['0.1']
a_h_str = ['10**3']
a_tv_str = ['10^(-6)']
a_norm_str = ['10^(-9)']

# ResNet18
if model == 'ResNet18':
    a_tv = [10**(-3)]
    a_tv_str = ['10^(-3)']

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients.')
parser.add_argument('--attack', type=str, default=method_IR,
                    choices=['DLG', 'IGI', 'ING', 'Ours'],
                    help='attack method to run.')
parser.add_argument('--model', type=str, default=model,
                    choices=['LeNet', 'ResNet18', 'EfficientNet0', 'MobileNet'],
                    help='model to use.')
parser.add_argument('--a_g', type=str, default="0",
                    help='alpha gradient')
parser.add_argument('--a_h', type=str, default="0",
                    help='alpha hidden')
parser.add_argument('--a_tv', type=str, default="0",
                    help='alpha total variation')
parser.add_argument('--a_norm', type=str, default="0",
                    help='alpha six norm')
parser.add_argument('--n_images', type=str, default="4",
                    help='number of images')
parser.add_argument('--n', type=str, default="0",
                    help='running index')
parser.add_argument('--method', type=str, default="2",
                    help='method for image encoding reconstruction')
args = parser.parse_args()

method_IR = args.attack
model = args.model

index_a_g = int(args.a_g)
index_a_h = int(args.a_h)
index_a_tv = int(args.a_tv)
index_a_norm = int(args.a_norm)
running_index = int(args.n)
method_IER = int(args.method)

print('index: ', index_a_g, index_a_h, index_a_tv, index_a_norm)

n_images = int(args.n_images)

if running_index == 0:
    with open("Performance_metrics.txt", "a") as f:
        f.write("\n Batch size = %d  \n"%n_images)

with open("Performance_metrics.txt", "a") as f:
    f.write("%s  "%a_g_str[index_a_g])
    f.write("%s  "%a_h_str[index_a_h])
    f.write("%s  "%a_tv_str[index_a_tv])
    f.write("%s  "%a_norm_str[index_a_norm])

n_rows = 1


tt = transforms.ToPILImage()


if dataset == 'CIFAR100':
    dst = datasets.CIFAR100("~/.torch", download=True)
    n_total_images = 50000
    n_classes = 100
    w = 32

if dataset == 'LFWPeople':
    dst = datasets.LFWPeople("~/.torch", download=True)
    n_total_images = 13233
    n_classes = 5749
    w = 128




# ImageNet
from torch.utils import data
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
        image = Image.open(os.path.join(self.root, image_filename)).convert("RGB")
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
import os

filenames = os.listdir("Imagenet/")

# the class of image might be ['black cat', 'tabby cat', 'tabby cat']
labels = np.arange(1000)


if dataset == 'ImageNet':
    dst = simpleDataset(root=root,
                           filenames=filenames,
                           labels=labels
                           )

    n_total_images = 500
    n_classes = 1000
    w = 256   # Default value
    w = 64
    # w = 512

print(dst)

tp = transforms.Compose([
    transforms.Resize(w),
    transforms.CenterCrop(w),
    transforms.ToTensor()
])


def draw_figures(history):
    plt.figure(figsize=(4,8))
   
    for i in range(n_images):

        plt.subplot(n_rows, n_images//n_rows, i +1)
        plt.imshow(history[i])
        plt.axis('off') 

    f_name = 'B_' + str(n_images) + '_w_' + str(w) + '_'+ dataset + '_ours' + '.eps'
    plt.savefig(f_name, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(4,8))

    for i in range(n_images):

        plt.subplot(n_rows, n_images//n_rows, i +1)
        plt.imshow(tt(gt_data[i][0].to(device)))
        plt.axis('off') 

    f_name = 'B_' + str(n_images) + '_w_' + str(w) + '_'+ dataset + '_original' + '.eps'
    plt.savefig(f_name, bbox_inches='tight')



if dataset == 'CIFAR100':
    image_index_list = np.loadtxt('index_list_cifar.txt', dtype=int)

if dataset == 'LFWPeople':
    image_index_list = np.loadtxt('index_list_LWF.txt', dtype=int)

if dataset == 'ImageNet':
    image_index_list = np.loadtxt('index_list_ImageNet.txt', dtype=int)

index_list = image_index_list[running_index,:n_images]


gt_data = []
gt_label = []
gt_onehot_label = []


for i in range(n_images):
    tmp = tp(dst[index_list[i]][0]).to(device)
    print(tmp.size())
    gt_data.append(tmp.view(1,*tmp.size()))   # size: (1,3,32,32)

    tmp_label = torch.Tensor([dst[index_list[i]][1]]).long().to(device)
    tmp_label = tmp_label.view(1,)
    print(i, index_list[i], tmp_label)
    if tmp_label in gt_label:
        print('Duplicate labels....')
    
    # assert tmp_label not in gt_label
    
    gt_label.append(tmp_label)

    gt_onehot_label.append(label_to_onehot(tmp_label, num_classes=n_classes))



torch.manual_seed(1234)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook

from models.vision import weights_init, LeNet, ResNet18, EfficinetNet0, MobileNet
if model == 'LeNet':
    n_h  = 12*w*w//16  #Hidden units in LeNet 
    net = LeNet(n_classes, n_h).to(device)
    net.conv4.register_forward_hook(get_activation('conv4'))  # LeNet()


if model == 'ResNet18':
    n_h  = 24*w*w//16  #Hidden units in ResNet18
    net = ResNet18(n_classes, n_h).to(device)
    net.apply(weights_init)
    net.layer4.register_forward_hook(get_activation('layer4'))  # ResNet18()




if model == 'EfficientNet0':
    n_h = 1280
    net = EfficinetNet0("b0", num_classses=n_classes).to(device)
    print('This is EfficientNet b0')
    net.features.register_forward_hook(get_activation('features'))  

    


if model == 'MobileNet':
    n_h = w*w
    net = MobileNet(ch_in=3,n_classes=n_classes, n_h=n_h).to(device)
    print('MobileNet')
    net.model.register_forward_hook(get_activation('model'))
    




m = torch.nn.AvgPool2d(kernel_size = 4)



criterion = cross_entropy_for_onehot

y = []

for i in range(n_images):
    pred = net(gt_data[i])

    if model == 'LeNet':
        h_ = activation['conv4']   # LeNet
    
    if model == 'ResNet18':
        h_ = activation['layer4']    # Resnet18
        h_ = m(h_)
    
    if model == 'EfficientNet0':
        # pool = nn.AdaptiveAvgPool2d(1)
        tmp = activation['features']
        # h_ = pool(tmp)
        h_ = tmp

    if model == 'MobileNet':
        tmp = activation['model']
        h_ = tmp

    h_ = torch.reshape(h_, (1, n_h))

    y_ = criterion(pred, gt_onehot_label[i])
    y.append(y_)

dy_dx_list = []
dy_dx = ()
for i in range(n_images):
    dy_dx_ = torch.autograd.grad(y[i], net.parameters(), retain_graph = True)
    # print(len(dy_dx_))
    dy_dx_list.append(dy_dx_)
    if i == 0:
        dy_dx += dy_dx_
    else:
        dy_dx = tuple(map(lambda x, y: x + y, dy_dx, dy_dx_))

dy_dx = tuple(map(lambda x: x/n_images,dy_dx)) # take average gradient

original_dy_dx = list((_.detach().clone() for _ in dy_dx))


###### Initialization of dummy images and labels
# Apply Idea of paper: "See through Gradients: Image Batch Recovery via GradInversion"


r1, r2 = 0, 1

# # LeNet

if model == 'LeNet' and dataset == 'CIFAR100':  
    alpha_sixnorm_list = [10**(-7)]
    alpha_tv_list = [10**(-5)]
    alpha_l = 10**3


# LWFPeople
if model == 'LeNet' and dataset == 'LFWPeople':  # No weight_init()
    alpha_sixnorm_list = [10**(-9)]
    alpha_tv_list = [10**(-6)]
    alpha_l = 10**3


if model == 'LeNet' and dataset == 'ImageNet':  # No weight_init() for w = 256 or 512
    alpha_sixnorm_list = [10**(-8)]
    alpha_tv_list = [10**(-7)]
    alpha_l = 10**4



if model == 'LeNet' and dataset == 'ImageNet' and w==1024:  
    alpha_sixnorm_list = [10**(-9)]
    alpha_tv_list = [10**(-8)]
    alpha_l = 10**4

if model == 'LeNet' and dataset == 'ImageNet' and w==32:  
    alpha_sixnorm_list = [10**(-8)]
    alpha_tv_list = [10**(-5)]
    alpha_l = 10**3


if model == 'LeNet' and dataset == 'ImageNet' and w==64:  
    alpha_sixnorm_list = [10**(-8)]
    alpha_tv_list = [10**(-5)]
    alpha_l = 10**4


if model == 'LeNet' and dataset == 'ImageNet' and w==128:  
    alpha_sixnorm_list = [10**(-8)]
    alpha_tv_list = [10**(-6)]
    alpha_l = 10**4



# ResNet18

if model == 'ResNet18' and (dataset == 'CIFAR100' or dataset == 'LFWPeople'):  # weight_init()     # Need to find optimal values for Imagenet in ResNet18 & MobileNet
    alpha_l = 10**4
    alpha_sixnorm_list = [10**(-10)]
    alpha_tv_list = [10**(-7)]




if model == 'ResNet18' and dataset == 'ImageNet':  # weight_init()      # For resolution = 256
    alpha_l = 10**4
    alpha_sixnorm_list = [10**(-10)]
    alpha_tv_list = [10**(-8)]



if model == 'MobileNet' and dataset == 'CIFAR100':

    ### Version 1
    alpha_sixnorm_list = [10**(-11)]
    alpha_tv_list = [10**(-8)]
    alpha_l = 10**3

if model == 'MobileNet' and dataset == 'LFWPeople' :
    alpha_sixnorm_list = [10**(-11)]
    alpha_tv_list = [10**(-9)]
    alpha_l = 10**3


if model == 'MobileNet' and dataset == 'ImageNet' :   # For resolution = 256
    alpha_sixnorm_list = [10**(-11)]
    alpha_tv_list = [10**(-9)]
    alpha_l = 10**4


if method_IR == 'Ours':
    n_epochs1 = 800    # Training dummy images using image representation
    
    if dataset == 'LFWPeople':
        n_epochs1 = 1500
       
    if dataset == 'ImageNet':
        # n_epochs1 = 1500
        n_epochs1 = 1000
else:
    n_epochs1 = 0       #  DLA


start_time = time.time()

### Method 1: Reconstructed hidden values when weak gradients = 0
######### Try to recover hidden layer before softmax
if method_IER == 1:
    h_list = []
    for i in range(n_images):
        num = original_dy_dx[-2][gt_label[i],:]
        denum  = original_dy_dx[-1][gt_label[i]]
        h_ = torch.div(num, denum)
        h_list.append(h_)


###### Propose method 2 for hidden recovery
### Reconstructed hidden values when weak gradients = exp(y_c)/sum_j(y_j)
# """
if method_IER == 2:
    mean_ = 1/n_classes
    h_list2 = []
    for j in range(n_h):
        a = np.zeros((n_images, n_images)) + mean_
        b = []
        for k in range(n_images):
            a[k,k] = n_images*original_dy_dx[-1][gt_label[k]].cpu() - mean_*(n_images-1)
            b.append(n_images*original_dy_dx[-2][gt_label[k], j].cpu())
        

        h_ = np.linalg.solve(a,b)
        h_list2.append(h_)

    h_list_ = h_list2
    h_list_ = np.moveaxis(h_list_, 0, -1)

    h_list = []
    for i in range(n_images):
        h_list.append(torch.tensor(h_list_[i:i+1], dtype=torch.double))
# """



for alpha_sixnorm in alpha_sixnorm_list:
    for alpha_tv in alpha_tv_list:
        history = []
        MSE = 0
        SSIM = 0
        PSNR = 0
        FFT = 0
        LPIPS = 0

        invert_images = []  # To store dummy data

        for i in range(n_images):  # Reconstruct images independently
            print('Image: ', i)

            tmp = (r1 - r2) * torch.rand(gt_data[i].size()) + r2

            torch.save(tmp, 'dummy_image.pt')
    
            dummy_data_i = tmp.to(device).requires_grad_(True)

        
            optimizer_init = torch.optim.Adam([dummy_data_i], lr=0.01)  # to initialize images

            for iter in range(n_epochs1):
                def closure():         
                    optimizer_init.zero_grad()

                    net(dummy_data_i)
                    if model == 'LeNet':
                        h = activation['conv4']   # LeNet
                    if model == 'ResNet18':
                        h = activation['layer4']    # Resnet18
                        h = m(h)
                    if model == 'EfficientNet0':
                        tmp = activation['features']
                        h = tmp

                    if model == 'MobileNet':
                        tmp = activation['model']
                        h = tmp


                    n_elements = torch.numel(h)
                    
                    if method_IER == 1:
                        h = torch.reshape(h, (1, n_elements))
                    elif method_IER == 2:
                        h = torch.reshape(h, (1, n_elements)).double().to(device)
                    

                    l2_loss = torch.cdist(h_list[i].to(device), h, p=2.0).sum()/(h_list[i]**2).sum()
                    
                    
                    six_norm_loss = (dummy_data_i**6).sum()
                    loss_tv = tv(dummy_data_i)

                    loss = alpha_l*l2_loss + alpha_sixnorm*six_norm_loss + alpha_tv*loss_tv

                    if iter%100 == 0:
                        print('l2_loss: ', l2_loss)
                        print('loss_norm: ', six_norm_loss)
                        print('loss_tv: ', loss_tv)
                        print(loss)


                    loss.backward(retain_graph=True)     # Compute gradient             
                    return loss

                optimizer_init.step(closure)

                if iter == (n_epochs1-1):
                    tmp = dummy_data_i.clone().detach()
                    dummy_data_i = torch.clip(tmp, 0, 1).to(device).requires_grad_(True) 
               


            history.append(tt(dummy_data_i[0].to(device)))
            invert_images.append(dummy_data_i)

        
           
            tmp_mse = torch.mean((gt_data[i][0] - invert_images[i][0])**2)
            tmp_ssim = metrics.structural_similarity(gt_data[i][0].cpu().detach().numpy(), invert_images[i][0].cpu().detach().numpy(), win_size=7, data_range = 1, channel_axis=0)
            tmp_psnr = metrics.peak_signal_noise_ratio(gt_data[i][0].cpu().detach().numpy(), invert_images[i][0].cpu().detach().numpy())

            fft_image1 = np.real(np.fft.fft2(gt_data[i][0].cpu().detach().numpy())).flatten()
            fft_image2 = np.real(np.fft.fft2(invert_images[i][0].cpu().detach().numpy())).flatten()
            tmp_fft_real = np.dot(fft_image1, fft_image2)/(norm(fft_image1)*norm(fft_image2))

            fft_image1 = np.imag(np.fft.fft2(gt_data[i][0].cpu().detach().numpy())).flatten()
            fft_image2 = np.imag(np.fft.fft2(invert_images[i][0].cpu().detach().numpy())).flatten()
            tmp_fft_img = np.dot(fft_image1, fft_image2)/(norm(fft_image1)*norm(fft_image2))

            tmp_lpips = lpips(gt_data[i], invert_images[i])

            MSE +=tmp_mse
            SSIM +=tmp_ssim
            PSNR +=tmp_psnr
            FFT +=(tmp_fft_real + tmp_fft_img)/2
            LPIPS +=tmp_lpips

        MSE, SSIM, PSNR, FFT, LPIPS = MSE/n_images, SSIM/n_images, PSNR/n_images, FFT/n_images, LPIPS/n_images

        print('MSE, SSIM, PSNR, FFT, LPIPS of batched images: ', MSE, SSIM, PSNR, FFT, LPIPS)

stop_time = time.time()

print('Running time of our method: ', stop_time-start_time)
        
draw_figures(history)

if method_IR == 'Ours':
    # Write in a file
    with open("Performance_metrics.txt", "a") as f:
        f.write("%.6f  "%MSE)
        f.write("%.6f  "%SSIM)
        f.write("%.6f  "%PSNR)
        f.write("%.6f  "%FFT)
        f.write("%.6f  \n"%LPIPS)


start_time_DLG = time.time()


########## Phase 3: Training images using shared Gradients

dummy_data_list = []

for i in range(n_images):
    dummy_data_ = invert_images[i].to(device).requires_grad_(True)
    dummy_label_ = torch.rand(gt_onehot_label[i].size()).to(device).requires_grad_(True)

    dummy_data_list.append(dummy_data_)
    dummy_data_list.append(dummy_label_)


if method_IR == 'DLG':
    n_epochs2 = 1000  # Number of training iterations 
    learning_rate = 0.5   # one image
    #learning_rate = 0.2   # two image
    if n_images == 4:
        learning_rate = 0.02
        n_epochs2 = 3000
elif method_IR == 'IGI':          # Improved gradient inversion attacks and defenses in FL
    n_epochs2 = 400
    if n_images == 1:
        learning_rate = 0.4   # one image
        n_epochs2 = 100
    if n_images == 2:
        learning_rate = 0.1   # two images
    if n_images == 4:
        learning_rate = 0.05  # four images
    if n_images >= 8:
        learning_rate = 0.1
        n_epochs2 = 400
elif method_IR == 'ING':
    if n_images == 1:
        n_epochs2 = 1500
        learning_rate = 0.05
    if n_images == 2:
        n_epochs2 = 2500
        learning_rate = 0.05
    if n_images == 4:
        n_epochs2 = 4500
        learning_rate = 0.05
    if n_images >= 8:
        n_epochs2 = 8000
        learning_rate = 0.05
else:
    n_epochs2 = 0
    learning_rate = 0.1


if method_IR == 'ING':
    optimizer = torch.optim.Adam(dummy_data_list, lr=learning_rate)
else:
    optimizer = torch.optim.LBFGS(dummy_data_list, lr=learning_rate, max_iter=1)  # for training at attacker (only gradient)
history = []   # Save the recovered images






optimizer1 = torch.optim.Adam(net.parameters())   # for training at victim side

# compute original gradient 
n_epochs_para = 1    # Number of training times for network parameters


alpha_h_ = a_h[index_a_h]
alpha_g_ = a_g[index_a_g]
alpha_tv_ = a_tv[index_a_tv]
alpha_norm_ = a_norm[index_a_norm]


for epoch in range(n_epochs_para):
    optimizer1.zero_grad()

    pred_list = []
    y = []
    for i in range(n_images):
        pred = net(gt_data[i])
        pred_list.append(net(gt_data[i]))
    
        y_ = criterion(pred, gt_onehot_label[i])
        y.append(y_)
    

    if epoch < n_epochs_para - 1:
        print('1') 
        for i in range(n_images):
            y[i].backward()
        optimizer1.step()

    else:
        print('2') 
        dy_dx_list = []
        dy_dx = ()
        for i in range(n_images):
            dy_dx_ = torch.autograd.grad(y[i], net.parameters())
            # print(len(dy_dx_))
            dy_dx_list.append(dy_dx_)
            if i == 0:
                dy_dx += dy_dx_
            else:
                dy_dx = tuple(map(lambda x, y: x + y, dy_dx, dy_dx_))

        dy_dx = tuple(map(lambda x: x/n_images,dy_dx)) # take average gradient

        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        #### Try to recover images
        h_list = []
        for i in range(n_images):
            num = original_dy_dx[-2][gt_label[i],:]
            denum  = original_dy_dx[-1][gt_label[i]]
            # print(num.size())
            # print(denum.size())
            h_ = torch.div(num, denum)
            h_list.append(h_)



        for iters in range(n_epochs2):
            def closure():

                optimizer.zero_grad()      
                h_dummy_list = []
                dummy_dy_dx_list = ()

                for i in range(n_images):
                    dummy_pred = net(dummy_data_list[2*i]) 
                    if model == 'LeNet':
                        h_dummy = activation['conv4']    # LeNet
                    if model == 'ResNet18':
                        h_dummy = activation['layer4']     # Resnet18
                        h_dummy = m(h_dummy)

                    if model == 'EfficientNet0':
                        tmp = activation['features']
                        # h_ = pool(tmp)
                        h_dummy = tmp
                    
                    

                    h_dummy = torch.reshape(h_dummy, (1, n_h))
                    
                    h_dummy_list.append(h_dummy)

                    dummy_onehot_label = F.softmax(dummy_data_list[2*i+1], dim=-1)
                    dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                    dummy_dy_dx_ = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                    if i == 0:
                        dummy_dy_dx_list += dummy_dy_dx_
                    else:
                        dummy_dy_dx_list = tuple(map(lambda x, y: x + y, dummy_dy_dx_list, dummy_dy_dx_))

                dummy_dy_dx_list = tuple(map(lambda x: x/n_images,dummy_dy_dx_list)) # take average gradient
                

                loss_g = 0
                
                # If Our work
                if method_IR == 'Ours':
                    for index, (gx, gy) in enumerate(zip(dummy_dy_dx_list, original_dy_dx)): 
                    # if index < 8:
                        loss_g += ((gx - gy) ** 2).sum() 


                    
                    loss_h = 0
                    for i in range(n_images):
                        loss_h += torch.cdist(h_list[i].to(device), h_dummy_list[i], p=2.0).sum()/(h_list[i]**2).sum()
                    loss_h = loss_h/n_images
                    

                
                    loss_tv = 0
                    for i in range(n_images):
                        loss_tv += tv(dummy_data_list[2*i])
                    loss_tv = loss_tv/n_images

                   
                    loss_norm = 0
                    for i in range(n_images):
                        loss_norm += (dummy_data_list[2*i]**6).sum()
                    loss_norm = loss_norm/n_images

                

                    loss = alpha_g_*loss_g + alpha_h_*loss_h + alpha_tv_*loss_tv + alpha_norm_*loss_norm    
                    
                    
                elif method_IR == 'DLG':   # If DLG
                    loss  = 0
                
                    for index, (gx, gy) in enumerate(zip(dummy_dy_dx_list, original_dy_dx)): 
                        loss += ((gx - gy) ** 2).sum()
                    if n_images == 1:
                        loss = 10**3*loss   # n_image = 1
                    elif n_images == 4:
                        loss = 10**4*loss
                
                elif method_IR == 'IGI':
                    for index, (gx, gy) in enumerate(zip(dummy_dy_dx_list, original_dy_dx)): 
                        loss_g += ((gx - gy) ** 2).sum()

                    loss_tv = 0
                    for i in range(n_images):
                        loss_tv += alpha_tv_*tv(dummy_data_list[2*i])

                    loss_clip = 0
                    for i in range(n_images):
                        clip_i = torch.minimum(torch.maximum(dummy_data_list[2*i], torch.tensor(0)), torch.tensor(1))
                        loss_clip += ((dummy_data_list[2*i] - clip_i)**2).sum()

                    loss_scale = 0
                    for i in range(n_images):
                        min_, max_ = torch.min(dummy_data_list[2*i]), torch.max(dummy_data_list[2*i])
                        scale_i = (dummy_data_list[2*i] - min_)/(max_ - min_ + 10**(-5))
                        loss_scale += ((dummy_data_list[2*i] - scale_i)**2).sum()

                    if n_images == 1:
                        loss = 10**2*loss_g + 3*loss_tv  
                    if n_images == 2:
                        loss = 10**2*loss_g + 10*loss_tv  + 5*loss_clip
                    if n_images == 4:
                        loss = 10**2*loss_g + loss_tv  + 5*loss_clip
                    if n_images >= 8:
                        loss = 300*loss_g + loss_tv  + loss_clip
                elif method_IR == 'ING':
                    for index, (gx, gy) in enumerate(zip(dummy_dy_dx_list, original_dy_dx)): 
                        loss_g += 1- nn.CosineSimilarity(dim = 0, eps=1e-6)(torch.flatten(gx), torch.flatten(gy))

                    loss_tv = 0
                    for i in range(n_images):
                        loss_tv += alpha_tv_*tv(dummy_data_list[2*i])

                    if n_images == 1:
                        loss = loss_g + 10*loss_tv
                    elif n_images == 2:
                        loss = 10*loss_g + 5*loss_tv
                    elif n_images == 4:
                        loss = 100*loss_g + 10*loss_tv
                    elif n_images >= 8:
                        loss = 300*loss_g + loss_tv

                loss.backward()     # Compute gradient             
                return loss
            
        
            optimizer.step(closure)
            if iters %100 == 0:        
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())      

            
            if iters == (n_epochs2-1):
                for index in range(n_images):
                    tmp = dummy_data_list[2*index].clone().detach()
                    dummy_data_list[2*index] = torch.clip(tmp, 0, 1).to(device).requires_grad_(True) 
                

        for i in range(n_images):
            history.append(tt(dummy_data_list[2*i][0].to(device)))
        
        MSE = 0
        SSIM = 0
        PSNR = 0
        FFT = 0
        LPIPS = 0

        #### Matching between recovered and true images
        list_images = np.zeros(shape=(n_images,n_images))

        for i in range(n_images):
            for j in range(n_images):
                tmp_ssim = metrics.structural_similarity(gt_data[i][0].cpu().detach().numpy(), dummy_data_list[2*j][0].cpu().detach().numpy(), win_size=7, data_range = 1, channel_axis=0)
                list_images[i, j] = tmp_ssim


        for i in range(n_images):
            # print(torch.min(dummy_data_list[2*i]), torch.max(dummy_data_list[2*i]))
            j = np.argmax(list_images[i,:])
            # print('Image j: ', j)

            tmp_mse = torch.mean((gt_data[i][0]-dummy_data_list[2*j][0])**2)
            tmp_ssim = metrics.structural_similarity(gt_data[i][0].cpu().detach().numpy(), dummy_data_list[2*j][0].cpu().detach().numpy(), win_size=7, data_range = 1, channel_axis=0)
            tmp_psnr = metrics.peak_signal_noise_ratio(gt_data[i][0].cpu().detach().numpy(), dummy_data_list[2*j][0].cpu().detach().numpy())
            
            fft_image1 = np.real(np.fft.fft2(gt_data[i][0].cpu().detach().numpy())).flatten()
            fft_image2 = np.real(np.fft.fft2(dummy_data_list[2*j][0].cpu().detach().numpy())).flatten()
            tmp_fft_real = np.dot(fft_image1, fft_image2)/(norm(fft_image1)*norm(fft_image2))

            fft_image1 = np.imag(np.fft.fft2(gt_data[i][0].cpu().detach().numpy())).flatten()
            fft_image2 = np.imag(np.fft.fft2(dummy_data_list[2*j][0].cpu().detach().numpy())).flatten()
            tmp_fft_img = np.dot(fft_image1, fft_image2)/(norm(fft_image1)*norm(fft_image2))

            tmp_lpips = lpips(gt_data[i], dummy_data_list[2*j])
            # print('MSE, SSIM, PSNR, FFT, LPIPS between original and reconstructed image 1: ', tmp_mse, tmp_ssim, tmp_psnr, (tmp_fft_real + tmp_fft_img)/2, tmp_lpips)


            MSE +=tmp_mse
            SSIM +=tmp_ssim
            PSNR +=tmp_psnr
            FFT +=(tmp_fft_real + tmp_fft_img)/2
            LPIPS +=tmp_lpips

        MSE, SSIM, PSNR, FFT, LPIPS = MSE/n_images, SSIM/n_images, PSNR/n_images, FFT/n_images, LPIPS/n_images

        print('MSE, SSIM, PSNR, FFT, LPIPS of batched images: ', MSE, SSIM, PSNR, FFT, LPIPS)


stop_time_DLG = time.time()

print('Running time of DLG: ', stop_time_DLG - start_time_DLG)
     


draw_figures(history)
if method_IR == 'DLG':
    with open("Performance_metrics.txt", "a") as f:
        f.write("%.6f  "%MSE)
        f.write("%.6f  "%SSIM)
        f.write("%.6f  "%PSNR)
        f.write("%.6f  "%FFT)
        f.write("%.6f  \n"%LPIPS)

# """
