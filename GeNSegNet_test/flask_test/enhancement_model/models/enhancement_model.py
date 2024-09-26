import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn


class EnhancementModel(BaseModel):
    """
       This class implements the enhancement model of GeNSeg-Net.
       The model training requires '--dataset_mode aligned' dataset.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='designed_G', dataset_mode='aligned')

        parser.add_argument('--df_dim_TG', type=int, default=384, help='The base channel num of disc')
        parser.add_argument('--d_depth_TG', type=int, default=[2, 3, 5, 5], help='Discriminator Depth')
        parser.add_argument('--patch_size_TG', type=int, default=4, help='Discriminator Depth')
        parser.add_argument('--d_norm_TG', type=str, default="ln", help='Discriminator Normalization')
        parser.add_argument('--d_window_size_TG', type=int, default=[8, 8, 8], help='discriminator mlp ratio')
        parser.add_argument('--d_act_TG', type=str, default="gelu", help='Discriminator activation layer')
        parser.add_argument('--img_size_TG', type=int, default=512, help='size of each image dimension')

        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the enhancement class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG, norm=opt.norm,
                                      use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:  # define a discriminator
            self.netD = networks.define_D(netD=opt.netD, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids,
                                          df_dim_TG=opt.df_dim_TG, d_depth_TG=opt.d_depth_TG, patch_size_TG=opt.patch_size_TG, d_norm_TG=opt.d_norm_TG,
                                          d_window_size_TG=opt.d_window_size_TG, d_act_TG=opt.d_act_TG, img_size_TG=opt.img_size_TG)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)


    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_B
        fake_AB = torch.cat((fake_AB, fake_AB, fake_AB), dim=1)
        pred_fake = self.netD(fake_AB.detach())
        fake_label = torch.full((pred_fake.shape[0], pred_fake.shape[1]), 0., dtype=torch.float, device='cuda:0')
        self.loss_D_fake = nn.MSELoss()(pred_fake, fake_label)
        # Real
        real_AB = self.real_B
        real_AB = torch.cat((real_AB, real_AB, real_AB), dim=1)
        pred_real = self.netD(real_AB)
        real_label = torch.full((pred_real.shape[0], pred_real.shape[1]), 1., dtype=torch.float, device='cuda:0')
        self.loss_D_real = nn.MSELoss()(pred_real, real_label)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        fake_AB = torch.cat((fake_AB, fake_AB, fake_AB), dim=1)
        pred_fake = self.netD(fake_AB)
        fake_label_G = torch.full((pred_fake.shape[0], pred_fake.shape[1]), 1., dtype=torch.float, device='cuda:0')
        self.loss_G_GAN = nn.MSELoss()(pred_fake, fake_label_G)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # update G's weights
