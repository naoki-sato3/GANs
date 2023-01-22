from __future__ import print_function
import argparse
import os
import random
import wandb
import uuid
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
from mycleanfid import fid
from adabelief_pytorch import AdaBelief

#Argument Set#
parser = argparse.ArgumentParser()
parser.add_argument('--Dataset', default='bedroom')
parser.add_argument('--Dataroot', default='./data', help='path to default')
parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--imagesize', type=int, default=64, help='input image is 64x64')
parser.add_argument('--nc', type=int, default=3, help='color image have 3 channels')
parser.add_argument('--nz', type=int, default=100, help='size of generator input')
parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in generator')
parser.add_argument('--ndf', type=int, default=64, help='size of fearture maps in discriminator')
parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--cuda', default=True)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--d_lr', type=float, default=3e-4, help='learning rate of discriminator')
parser.add_argument('--g_lr', type=float, default=1e-4, help='leraning rate of generator')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam')
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam | AdaBelief | RMSProp')
parser.add_argument('--model', default='DCGAN')
args = parser.parse_args()

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        return

    def forward(self, input):
        output = self.main(input)
        return output
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:     
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        return
        
    def forward(self, input):
        output = self.main(input)
        return output

if __name__ == '__main__':
    #Wandb Setup#
    model_name = f'{args.model}'
    wandb_project_name = "DCGAN_LSUN-Bedroom"
    exp_name_suffix = str(uuid.uuid4())
    wandb_exp_name = f"b={args.batchsize},"
    wandb.init(config = args,
            project = wandb_project_name,
            name = wandb_exp_name,
            entity = args.wandb_entity)
    wandb.init(settings=wandb.Settings(start_method='fork'))
    opt = wandb.config

    SEED=999
    random.seed(SEED)
    torch.manual_seed(SEED)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0')
    print(f'Using {device} device.')
    
    transform=transforms.Compose([transforms.Resize(args.imagesize),
                                  transforms.CenterCrop(args.imagesize),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    dataset = dset.LSUN(root='./data', classes=['bedroom_train'], transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=0)   

    # Create the generator
    if cuda:
        netG = Generator(args.ngpu).to(device)
    netG.apply(weights_init)
    #print(netG)

    # Create the Discriminator
    if cuda:
        netD = Discriminator(args.ngpu).to(device)
    netD.apply(weights_init)
    
    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.

    # Setup optimizers for both G and D
    ##Adam##
    if(args.optimizer == "Adam"):
        optimizerD = optim.Adam(netD.parameters(), lr=3e-4, betas=(args.beta1, args.beta2))
        optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))

    ##AdaBelief##
    elif(args.optimizer == "AdaBelief"):
        optimizerD = AdaBelief(netD.parameters(), lr=3e-5, eps=1e-16, betas=(args.beta1, args.beta2), rectify=True,weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False)
        optimizerG = AdaBelief(netG.parameters(), lr=3e-4, eps=1e-16, betas=(args.beta1, args.beta2), rectify=True, weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False)

    ##RMSProp##
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=3e-5, alpha=0.99, eps=1e-08)
        optimizerG = optim.RMSprop(netG.parameters(), lr=1e-4, alpha=0.99, eps=1e-08)
    
    output_dir = f'./output'
    real_folder = f'./output/all_real_imgs_lsun'
    if not os.path.exists(real_folder):
        os.mkdir(real_folder)
        for i in tqdm(range()):
            vutils.save_image(dataset[i][0], real_folder + '/{}.png'.format(i), normalize=True)

    fake_folder = f'./output/all_fake_imgs_lsun'
    if not os.path.exists(fake_folder):
        os.mkdir(fake_folder)

    print("Generator Parameters:",sum(p.numel() for p in netG.parameters() if p.requires_grad))
    print("Discriminator Parameters:",sum(p.numel() for p in netD.parameters() if p.requires_grad))
    
    iters = 0
    frechet_dist = 500
    first_loop = True
    
    print("Starting Training Loop...")
    for epoch in range(args.num_epochs):
        for i, data in enumerate(dataloader, 0):
            optimizerD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            if(iters % 500 == 0):
                norm_D = 0
                parameters_D = [p for p in netD.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters_D:
                    param_norm = p.grad.detach().data.norm(2)
                    norm_D += param_norm.item() ** 2
                norm_D = norm_D ** 0.5

            optimizerG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            
            if(iters % 500 == 0):
                norm_G = 0
                parameters_G = [p for p in netG.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters_G:
                    param_norm = p.grad.detach().data.norm(2)
                    norm_G += param_norm.item() ** 2
                norm_G = norm_G ** 0.5
                
            if (iters % 10000 == 0) & (iters != 0):
                print('iters:%d, calculate FID...' % iters)
                fid_batch_size = 64
                fake_image_num_sample = 50000
                generation_loop_iter = int(fake_image_num_sample/fid_batch_size)
                netG.eval()
                for i in range(generation_loop_iter):
                    noise = torch.randn(fid_batch_size, int(args.nz), 1, 1).to(device)
                    fake = netG(noise)
                    for j in range(fake.shape[0]):
                        vutils.save_image(fake.detach()[j,...],fake_folder + '/{}.png'.format(j + i * fid_batch_size), normalize=True)
                netG.train()
                with torch.no_grad():
                    frechet_dist=fid.compute_fid(real_folder, fake_folder, first_loop, device=device)
                first_loop = False

            if (iters % 500 == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fixed_noise = torch.randn(args.batchsize, args.nz, 1, 1, device=device)
                    fake_display = netG(fixed_noise).detach().cpu()
                    gen_grid = vutils.make_grid(fake_display, padding=2, normalize=True)
                    gen_imgs = wandb.Image(gen_grid)   

            wandb.log({
                'loss_D': errD.item(),
                'loss_G': errG.item(),
                'norm_D': norm_D,
                'norm_G': norm_G,
                'FID': frechet_dist,
                'gen_imgs': gen_imgs,
                'epoch': epoch,
                'steps': iters,
                'batch_size': args.batchsize
            })
            
            if i % 500 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFID: %.4f\t'
                    % (epoch+1, args.num_epochs, i, len(dataloader), errD.item(), errG.item(),frechet_dist))

            if frechet_dist <= 70:
                break

            iters+=1

        if frechet_dist <= 70:
            print('break loop')
            break

    print('steps:%d, loss_D:%.4f, loss_G:%.4f,FID:%.4f' %(iters,errD.item(),errG.item(),frechet_dist))