# Existence and Estimation of Critical Batch Size for Training generative adversarial networks of two time-scale update rule
Code for reproducing experiments in our paper.  
Our experiments were based on the basic code for DCGAN, WGAN-GP, and BigGAN.

# Abstract
Previous results have shown that a two time-scale update rule (TTUR) using learning rates, such as constant and decaying learning rates, is practically useful for training generative adversarial networks (GANs) from the viewpoints of theory and practice. Moreover, not only the setting of learning rate but also the setting of batch size are important factors for training GANs of TTUR and influence the number of steps needed for training GANs of TTUR. This paper studies the relationship between batch size and the number of steps needed for training GANs of TTUR using constant learning rates. In theoretical parts, we show that, for TTUR using constant learning rates, the number of steps needed to find stationary points of loss functions of both a discriminator and a generator decreases as the batch size increases and that there exists a critical batch size minimizing the stochastic first order oracle (SFO) complexity.
In practical parts, we use the Fre ́chet inception distance (FID) as the performance measure for training GANs of TTUR, and we provide numerical results indicating that the number of steps needed to achieve a low FID score decreases as the batch size increases and that the SFO complexity increases once the batch size exceeds the measured critical batch size. Moreover, we show that measured critical batch sizes are close to estimated batch sizes based on our theoretical results.

# Additional Experiments Results to Refute Reviewer gyzT. 
## Reply for the Comment 6
As you know, GAN training was very unstable and often resulted in vanishing gradient, especially when the batch size was extremely small.
Therefore, we used a more gradual stopping condition. An example is shown below.
<img width="1212" alt="スクリーンショット 2023-04-01 14 42 47" src="https://user-images.githubusercontent.com/95958702/229267891-9690a97e-184e-4c97-8571-501b67f24423.png">
The figure above shows the results of training DCGAN with AdaBelief, with a batch size of 4. All other parameters are exactly the same as used in our paper.
**Top Left:** The loss function values for the generator and discriminator are plotted. The blue line is the generator plot and the orange line is the discriminator plot. **Top Right:** The FID score is plotted against the number of steps. The first FID score was measured at 400k steps. **Bottom Left:** The gradient of the generator loss function is plotted versus the number of steps. **Bottom Right:** The gradient of the discriminator loss function is plotted versus the number of steps.


Moreover, we can check that, on training DCGAN on LSUN-Bedroom, the measured critical batch size when the stopping condition is FID ≦ 50 is larger than the measured critical batch size when the stopping condition is FID ≦ 70. Please see the graph below.
<img width="724" alt="image" src="https://user-images.githubusercontent.com/95958702/229266481-e1eb93b0-555f-4fb6-b141-c5d63dbfed84.png">

This is a further plot of the graph for FID ≦ 50, in addition to Figure 2 in our paper.
It can be seen that as the stopping condition is tightened, the measured Critical Batch Size increases with AdaBelief and RMSProp.

## Reply for the Comment 7
Figures 2 and 4 show the average of three runs. We further show the graph below. 
![Nbb01r-4](https://user-images.githubusercontent.com/95958702/229267325-9cc4b9d5-f86c-43bb-9c51-4b8cbca3146b.png)
**Left:** a modified version of Figure 2. **Right:** a modified version of Figure 4.
The dotted line in the center is the average of the three runs, which is covered by the maximum and minimum of the three runs (shaded area).

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
