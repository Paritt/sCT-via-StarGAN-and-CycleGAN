import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'##GENERATOR##'
# Block
# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

#InceptionBlock
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.x1 = ConvBlock(in_channels, int(out_channels/4), 1, 1, 0)
        self.x2 = ConvBlock(in_channels, int(out_channels/4), 5, 1, 2)
        self.x3 = ConvBlock(in_channels, int(out_channels/4), 7, 1, 3)
        self.x4 = ConvBlock(in_channels, int(out_channels/4), 11, 1, 5)
        
    def forward(self, x):
        x1 = self.x1(x)
        x2 = self.x2(x)
        x3 = self.x3(x)
        x4 = self.x4(x)
        return torch.cat([x1,x2,x3,x4],1)

#DownBlock
class Down(nn.Module):
  def __init__(self, in_channels, out_channels, Inception=False):
    super().__init__()
    if Inception:
      self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            InceptionBlock(in_channels, out_channels)
        )
    else:
      self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
  def forward(self, x):
      return self.maxpool_conv(x)

#UpBlock
class Up(nn.Module):
  def __init__(self, in_channels, out_channels, Inception=False):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    if Inception:
      self.conv = InceptionBlock(in_channels, out_channels)
    else:
      self.conv = ConvBlock(in_channels, out_channels)
  def forward(self, x1, x2):
      x1 = self.up(x1)
      # input is CHW
      diffY = x2.size()[2] - x1.size()[2]
      diffX = x2.size()[3] - x1.size()[3]

      x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
      # if you have padding issues, see
      # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
      # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
      x = torch.cat([x2, x1], dim=1)
      return self.conv(x) 
  
#Generator Model
class Generator(nn.Module):
    def __init__(self, in_channels=1, features=32):
        super().__init__()
        self.inc = InceptionBlock(in_channels,features)
        self.down1 = Down(features,features*2, Inception=True)
        self.down2 = Down(features*2,features*4)
        self.down3 = Down(features*4,features*8)
        self.down4 = Down(features*8,features*16)
        self.botton_neck = ConvBlock(features*16, features*32)
        self.up1 = Up(features*32+features*16,features*16)
        self.up2 = Up(features*16+features*8,features*8)
        self.up3 = Up(features*8+features*4,features*4)
        self.up4 = Up(features*4+features*2,features*2,Inception=True)
        self.up5 = Up(features*2+features,features,Inception=True)
        self.last = nn.Sequential(nn.Conv2d(features, 1,1),nn.Sigmoid())
    
    def forward(self, x):
        x1 = self.inc(x)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        botton = self.botton_neck(d4)
        u1 = self.up1(botton,d4)
        u2 = self.up2(u1,d3)
        u3 = self.up3(u2,d2)
        u4 = self.up4(u3,d1)
        u5 = self.up5(u4,x1)
        last = self.last(u5)
        return last
    

'##DISCRIMINATOR##'
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=32):
        super().__init__()
        self.l1 = ConvBlock(in_channels, features,3, 2,1)
        self.l2 = ConvBlock(features, features*2,3, 2,1)
        self.l3 = ConvBlock(features*2, features*4,3, 2,1)
        self.l4 = ConvBlock(features*4, features*4,3, 1,1)
        self.last = nn.Sequential(nn.Conv2d(features*4,1,1),nn.Sigmoid())

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.last(x)
        return x
    
'##Loss##'
#Discriminator loss
def get_disc_loss(real_X, fake_X, disc_X, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the target labels and returns a adversarial 
            loss (which you aim to minimize)
    '''
    disc_fake_X_hat = disc_X(fake_X.detach()) # Detach generator
    disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
    disc_real_X_hat = disc_X(real_X)
    disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
    disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2    
    return disc_loss

#Generator loss

#Adversarial loss
def get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion):
    '''
    Return the adversarial loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
        gen_XY: the generator for class X to Y; takes images and returns the images 
            transformed to class Y
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the target labels and returns a adversarial 
                  loss (which you aim to minimize)
    '''
    fake_Y = gen_XY(real_X)
    disc_fake_Y_hat = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))    
    return adversarial_loss, fake_Y

#Identity loss
def get_identity_loss(real_X, gen_YX, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity 
                        loss (which you aim to minimize)
    '''
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(identity_X, real_X)
    return identity_loss, identity_X

#Cycle consistency loss
def get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion):
    '''
    Return the cycle consistency loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real_X: the real images from pile X
        fake_Y: the generated images of class Y
        gen_YX: the generator for class Y to X; takes images and returns the images 
            transformed to class X
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
                        those images put through a X->Y generator and then Y->X generator
                        and returns the cycle consistency loss (which you aim to minimize)
    '''
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(cycle_X, real_X) 
    return cycle_loss, cycle_X


##Totle generator loss##
def get_gen_loss(real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        real_A: the real images from pile A
        real_B: the real images from pile B
        gen_AB: the generator for class A to B; takes images and returns the images 
            transformed to class B
        gen_BA: the generator for class B to A; takes images and returns the images 
            transformed to class A
        disc_A: the discriminator for class A; takes images and returns real/fake class A
            prediction matrices
        disc_B: the discriminator for class B; takes images and returns real/fake class B
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator 
            predictions and the true labels and returns a adversarial 
            loss (which you aim to minimize)
        identity_criterion: the reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        lambda_identity: the weight of the identity loss
        lambda_cycle: the weight of the cycle-consistency loss
    '''
    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
    adv_loss_BA, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    adv_loss_AB, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    gen_adversarial_loss = adv_loss_BA + adv_loss_AB

    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
    identity_loss_A, identity_A = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_B, identity_B = get_identity_loss(real_B, gen_AB, identity_criterion)
    gen_identity_loss = identity_loss_A + identity_loss_B

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
    cycle_loss_BA, cycle_A = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_AB, cycle_B = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

    # Total loss
    gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adversarial_loss

    return gen_loss, fake_A, fake_B