# Existence and Estimation of Critical Batch Size for Training generative adversarial networks of two time-scale update rule
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for DCGAN, WGAN-GP, and BigGAN.

# Abstract
Previous results have shown that a two time-scale update rule (TTUR) using learning rates, such as constant and decaying learning rates, is practically useful for training generative adversarial networks (GANs) from the viewpoints of theory and practice. Moreover, not only the setting of learning rate but also the setting of batch size are important factors for training GANs of TTUR and influence the number of steps needed for training GANs of TTUR. This paper studies the relationship between batch size and the number of steps needed for training GANs of TTUR using constant learning rates. In theoretical parts, we show that, for TTUR using constant learning rates, the number of steps needed to find stationary points of loss functions of both a discriminator and a generator decreases as the batch size increases and that there exists a critical batch size minimizing the stochastic first order oracle (SFO) complexity.
In practical parts, we use the Fre ́chet inception distance (FID) as the performance measure for training GANs of TTUR, and we provide numerical results indicating that the number of steps needed to achieve a low FID score decreases as the batch size increases and that the SFO complexity increases once the batch size exceeds the measured critical batch size. Moreover, we show that measured critical batch sizes are close to estimated batch sizes based on our theoretical results.

# Downloads
・[LSUN-Bedroom dataset](https://www.yf.io/p/lsun)  
・[CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)  
・[ImageNet dataset](https://image-net.org/index.php)  

# Install Dependent Libraries
```
pip install -r requirements.txt
```

# Wandb Setup
Please change entity name `XXXXXX` to your wandb entitiy.
```
parser.add_argument("--wandb_entity", type=str, default='XXXXXX', help='entity of wandb team')
```

# Usage
Please change optimizer Adam, AdaBelief, or RMSProp.
```
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam | AdaBelief | RMSProp')
```
Training DCGAN on the LSUN-Bedroom dataset for Adam, AdaBelief, and RMSProp.  
```
python3 dcgan.py
```
Training WGAN-GP on the CelebA dataset for Adam, AdaBelief, and RMSProp.  
```
python3 wgan-gp.py
```
