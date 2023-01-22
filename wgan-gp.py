import argparse
import os
import wandb
import uuid

import torchvision.transforms as transforms
import torchvision.datasets as dset
from mycleanfid import fid
import torchvision.utils as vutils
from adabelief_pytorch import AdaBelief
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--d_lr", type=float, default=3e-4, help="Discriminator learning rate")
parser.add_argument("--g_lr", type=float, default=1e-4, help="Generator learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam | AdaBelief | RMSProp')
parser.add_argument("--model", default="WGAN-GP")
args = parser.parse_args()
print(args)

#Wandb Setup#
model_name = f'{args.model}'
wandb_project_name = "WGANGP_CelebA"
exp_name_suffix = str(uuid.uuid4())
wandb_exp_name = f"b={args.batch_size},"
wandb.init(config = args,
           project = wandb_project_name,
           name = wandb_exp_name,
           entity = args.wandb_entity)
wandb.init(settings=wandb.Settings(start_method='fork'))
opt = wandb.config

img_shape = (args.channels, args.img_size, args.img_size)

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0")
print(f'Using {device} device.')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):    
    def __init__(
        self,
        n_input_noize_z = 100,
        n_channels = 3,
        n_fmaps = 64
    ):
        super( Generator, self ).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(n_input_noize_z, n_fmaps*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_fmaps*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( n_fmaps*8, n_fmaps*4, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( n_fmaps*4, n_fmaps*2, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( n_fmaps*2, n_fmaps, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( n_fmaps, n_channels, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.Tanh()
        )
        return

    def forward( self, input ):        
        output = self.layer(input)
        return output

class Discriminator(nn.Module):    
    def __init__(
       self,
       n_channels = 3,
       n_fmaps = 64,
       activate = False
    ):
        super( Discriminator, self ).__init__() 
        self.layer = nn.Sequential(
            nn.Conv2d(n_channels, n_fmaps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_fmaps, n_fmaps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_fmaps*2, n_fmaps*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_fmaps*4, n_fmaps*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_fmaps*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )
        self.activate = activate
        if(self.activate):
            self.activate_layer = nn.Sigmoid()      
        return

    def forward(self, input):
        output = self.layer(input)
        if(self.activate):
            output = self.activate_layer(output)
        return output.view(-1)

netG = Generator()
netD = Discriminator()

if cuda:
    netG.to(device)
    netD.to(device)

netG.apply(weights_init)
netD.apply(weights_init)

transform=transforms.Compose([ transforms.Resize([64,64]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

dataset = dset.CelebA(root='./data', split='train', transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

#Optimizers
##Adam##
if(args.optimizer == "Adam"):
    optimizerD = optim.Adam(netD.parameters(), lr=3e-4, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(args.beta1, args.beta2))

##AdaBelief##
elif(args.optimizer == "AdaBelief"):
    optimizerD = AdaBelief(netD.parameters(), lr=3e-5, eps=1e-16, betas=(args.beta1, args.beta2), rectify=True,weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False)
    optimizerG = AdaBelief(netG.parameters(), lr=5e-4, eps=1e-16, betas=(args.beta1, args.beta2), rectify=True, weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False)

##RMSProp##
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=1e-4, alpha=0.99, eps=1e-08)
    optimizerG = optim.RMSprop(netG.parameters(), lr=3e-4, alpha=0.99, eps=1e-08)

def calc_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
        alpha = alpha.to(device)
        interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolatesv,
        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
        create_graph=True, retain_graph=True, only_inputs=True
    )
    gradients = gradients[0].view(real_data.size(0), -1)
    gradient_penalty_loss = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
    return gradient_penalty_loss

real_folder = f'./output/all_real_imgs_celeba'
if not os.path.exists(real_folder):
    os.mkdir(real_folder)
    for i in tqdm(range(len(dataset))):
        vutils.save_image(dataset[i][0], real_folder + '/{}.png'.format(i), normalize=True)

fake_folder = f'./output/all_fake_imgs_celeba'
if not os.path.exists(fake_folder):
    os.mkdir(fake_folder)

print("Generator Parameters:",sum(p.numel() for p in netG.parameters() if p.requires_grad))
print("Discriminator Parameters:",sum(p.numel() for p in netD.parameters() if p.requires_grad))

lambda_gp = 10
frechet_dist =  500
iters = 0
first_loop = True

for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        if imgs.size()[0] != args.batch_size:
            break
        real_imgs = imgs.to(device)
        z = torch.randn(size=(args.batch_size, args.latent_dim, 1, 1)).to(device)
        fake_imgs = netG(z)
        optimizerD.zero_grad()
        real_validity = netD(real_imgs)
        fake_validity = netD(fake_imgs)
        gradient_penalty = calc_gradient_penalty(netD, real_imgs, fake_imgs, device)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
        d_loss.backward()

        if(iters % 500 == 0):
            norm_D = 0
            parameters_D = [p for p in netD.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters_D:
                param_norm = p.grad.detach().data.norm(2)
                norm_D += param_norm.item() ** 2
            norm_D = norm_D ** 0.5

        optimizerD.step()

        if iters % args.n_critic == 0:
            optimizerG.zero_grad()
            fake_imgs = netG(z)
            fake_validity = netD(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()

        if(iters % 500 == 0):
            norm_G = 0
            parameters_G = [p for p in netG.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters_G:
                param_norm = p.grad.detach().data.norm(2)
                norm_G += param_norm.item() ** 2
            norm_G = norm_G ** 0.5

        if iters % args.n_critic == 0:
            optimizerG.step()

        if (iters % 10000 == 0) & (iters != 0):
            print('iters:%d, calculate FID...' % iters)
            fid_batch_size = 64
            fake_image_num_sample = 50000
            generation_loop_iter = int(fake_image_num_sample/fid_batch_size)
            netG.eval()
            for i in range(generation_loop_iter):
                noise = torch.randn(size=(fid_batch_size,100,1,1)).to(device)
                fake = netG(noise)
                for j in range(fake.shape[0]):
                    vutils.save_image(fake.detach()[j,...],fake_folder + '/{}.png'.format(j + i * fid_batch_size), normalize=True)
            netG.train()
            with torch.no_grad():
                frechet_dist=fid.compute_fid(real_folder, fake_folder, first_loop, device=device)
                first_loop = False

        #Check generated images
        if (iters % 500 == 0) or ((epoch == args.n_epochs-1) and (iters == len(dataloader)-1)):
            with torch.no_grad():
                fixed_noise = torch.randn(args.batch_size, args.latent_dim, 1, 1, device=device)
                fake_display = netG(fixed_noise).detach().cpu()
                gen_grid = vutils.make_grid(fake_display, padding=2, normalize=True)
                gen_imgs = wandb.Image(gen_grid)

        if(iters % 500 == 0):
            print("[Epoch %d/%d],[Batch %d/%d],[loss_D: %f],[loss_G: %f],[GP: %f],[FID: %f]"
                  % (epoch, args.n_epochs, iters, len(dataloader), d_loss.item(), g_loss.item(), gradient_penalty, frechet_dist))

        wandb.log({
            'loss_D': d_loss.item(),
            'loss_G': g_loss.item(),
            'norm_D': norm_D,
            'norm_G': norm_G,
            'GP': gradient_penalty,
            'FID': frechet_dist,
            'gen_imgs': gen_imgs,
            'epoch': epoch,
            'steps': iters,
            'batch_size': args.batch_size,
        })

        if(frechet_dist <= 50):
            break
        iters += 1

    if(frechet_dist <= 50):
        break
