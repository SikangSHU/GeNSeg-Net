import argparse

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.TGnetworks.ViT_helper import DropPath, trunc_normal_
import math
from models.TGnetworks.ada import *
import scipy.signal
from TGtorch_utils.ops import upfirdn2d

###############################################################################
# Helper Functions
###############################################################################
class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':     # Choose
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':    # Choose
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the paper. But xavier and kaiming might work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':     # Choose
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch',
             use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: designed_G
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'designed_G':
        net = designed_generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(netD, init_type='normal', init_gain=0.02, gpu_ids=[], df_dim_TG=384, d_depth_TG=[2,3,5,5],
             patch_size_TG=4, d_norm_TG='ln', d_window_size_TG=[8,8,8], d_act_TG='gelu', img_size_TG=512):
    """Create a discriminator

    Parameters:
        netD (str)         -- the architecture's name: designed_D
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None

    if netD == 'designed_D':
        net = designed_discriminator(df_dim_TG=df_dim_TG, d_depth_TG=d_depth_TG, patch_size_TG=patch_size_TG, d_norm_TG=d_norm_TG,
                                     d_window_size_TG=d_window_size_TG, d_act_TG=d_act_TG, img_size_TG=img_size_TG)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
        self.block_size = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        S = self.block_size
        x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
        return x

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - typically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class designed_generator(nn.Module):
    """
       The designed generator in the enhancement model of GeNSeg-Net.
    """
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):

        assert(n_blocks >= 0)
        super(designed_generator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):    # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):          # add ResNet blocks
            model += [ResnetBlock_New(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, planes=256)]

        for i in range(n_downsampling):    # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ChannelAttention(nn.Module):

    def __init__(self, inplanes=1):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=inplanes // 4, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=inplanes // 4, out_channels=inplanes, kernel_size=1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc(self.max_pool(x))
        avg_out = self.fc(self.avg_pool(x))
        out = self.sigmoid(max_out + avg_out)
        return out


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_out, avg_out], dim=1)
        out = self.conv1(x)
        out = self.sigmoid(out)
        return out


class ResnetBlock_New(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, planes: int):

        super(ResnetBlock_New, self).__init__()
        expansion: int = 1
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

        # attention
        self.channel_atten = ChannelAttention(planes * expansion)
        self.spatial_atten = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = self.conv_block(x)
        atten = self.channel_atten(out) * out
        atten = self.spatial_atten(atten) * atten
        atten = atten + x
        out = self.relu(atten)
        return out


wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1': [0.7071067811865476, 0.7071067811865476],
    'db2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4': [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5': [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6': [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7': [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8': [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}


class matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


def count_matmul(m, x, y):
    m.total_ops += torch.DoubleTensor([int(0)])


class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def leakyrelu(x):
    return nn.functional.leaky_relu_(x, 0.2)


class CustomAct(nn.Module):
    def __init__(self, act_layer):
        super().__init__()
        if act_layer == "gelu":
            self.act_layer = gelu
        elif act_layer == "leakyrelu":
            self.act_layer = leakyrelu

    def forward(self, x):
        return self.act_layer(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=gelu, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = CustomAct(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., is_mask=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mat = matmul()
        self.is_mask = is_mask

    def forward(self, x):
        B, N, C = x.shape
        if self.is_mask == 1:
            H = W = int(math.sqrt(N))
            image = x.view(B, H, W, C).view(B * H, W, C)
            qkv = self.qkv(image).reshape(B * H, W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1, 2)
            x = x.reshape(B * H, W, C).view(B, H, W, C).view(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif self.is_mask == 2:
            H = W = int(math.sqrt(N))
            image = x.view(B, H, W, C).permute(0, 2, 1, 3).reshape(B * W, H, C)
            qkv = self.qkv(image).reshape(B * W, H, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1, 2).reshape(B * W, H, C).view(B, W, H, C).permute(0, 2, 1, 3).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

            attn = (self.mat(q, k.transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = self.mat(attn, v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "bn":
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == "in":
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == "pn":
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == "bn" or self.norm_type == "in":
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == "none":
            return x
        else:
            return self.norm(x)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class StageBlock(nn.Module):

    def __init__(self, depth, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=gelu, norm_layer=nn.LayerNorm):
        super().__init__()
        self.depth = depth
        self.block = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer
            ) for i in range(depth)])

    def forward(self, x):
        for blk in self.block:
            x = blk(x)
        return x


class DisBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=leakyrelu, norm_layer=nn.LayerNorm, separate=0):
        super().__init__()
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            is_mask=separate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = CustomNorm(norm_layer, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gain = np.sqrt(0.5) if norm_layer == "none" else 1

    def forward(self, x):
        x = x * self.gain + self.drop_path(self.attn(self.norm1(x))) * self.gain
        x = x * self.gain + self.drop_path(self.mlp(self.norm2(x))) * self.gain
        return x


class designed_discriminator(nn.Module):
    """
       The designed discriminator in the enhancement model of GeNSeg-Net.
    """
    def __init__(self, df_dim_TG=384, d_depth_TG=[2,3,5,5], patch_size_TG=4, d_norm_TG='ln', d_window_size_TG=[8,8,8],
                 d_act_TG='gelu', img_size_TG=512, diff_aug_TG='filter,translation,erase_ratio,color,hue', num_classes=1,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim = self.embed_dim = df_dim_TG

        depth = d_depth_TG
        self.patch_size = patch_size = patch_size_TG
        norm_layer = d_norm_TG
        self.window_size = d_window_size_TG
        self.diff_aug_TG = diff_aug_TG

        act_layer = d_act_TG
        if patch_size != 6:
            self.fRGB_1 = nn.Conv2d(3, embed_dim // 8, kernel_size=patch_size * 2, stride=patch_size, padding=patch_size // 2)
            self.fRGB_2 = nn.Conv2d(3, embed_dim // 8, kernel_size=patch_size * 2, stride=patch_size * 2, padding=0)
            self.fRGB_3 = nn.Conv2d(3, embed_dim // 4, kernel_size=patch_size * 4, stride=patch_size * 4, padding=0)
            self.fRGB_4 = nn.Conv2d(3, embed_dim // 2, kernel_size=patch_size * 8, stride=patch_size * 8, padding=0)

            num_patches_1 = (img_size_TG // patch_size) ** 2
            num_patches_2 = ((img_size_TG // 2) // patch_size) ** 2
            num_patches_3 = ((img_size_TG // 4) // patch_size) ** 2
            num_patches_4 = ((img_size_TG // 8) // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, num_patches_1, embed_dim // 8))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, num_patches_2, embed_dim // 4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, num_patches_3, embed_dim // 2))
        self.pos_embed_4 = nn.Parameter(torch.zeros(1, num_patches_4, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth[1])]
        self.blocks_1 = nn.ModuleList([
            DisBlock(
                dim=embed_dim // 8, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks_2 = nn.ModuleList([
            DisBlock(
                dim=embed_dim // 4, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks_3 = nn.ModuleList([
            DisBlock(
                dim=embed_dim // 2, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks_4 = nn.ModuleList([
            DisBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth[3])])
        self.last_block = nn.Sequential(
           Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], act_layer=act_layer, norm_layer=norm_layer))

        self.norm = CustomNorm(norm_layer, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed_1, std=.02)
        trunc_normal_(self.pos_embed_2, std=.02)
        trunc_normal_(self.pos_embed_3, std=.02)
        trunc_normal_(self.pos_embed_4, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        if 'filter' in self.diff_aug_TG:
            Hz_lo = np.asarray(wavelets['sym2'])
            Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size))
            Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2
            Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2
            Hz_fbank = np.eye(4, 1)
            for i in range(1, Hz_fbank.shape[0]):
                Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
                Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
                Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2: (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
            Hz_fbank = torch.as_tensor(Hz_fbank, dtype=torch.float32)
            self.register_buffer('Hz_fbank', torch.as_tensor(Hz_fbank, dtype=torch.float32))
        else:
            self.Hz_fbank = None
        if 'geo' in self.diff_aug_TG:
            self.register_buffer('Hz_geom', upfirdn2d.setup_filter(wavelets['sym6']))
        else:
            self.Hz_geom = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B, _, H, W = x.size()
        H = W = H // self.patch_size

        x_1 = self.fRGB_1(x).flatten(2).permute(0, 2, 1)
        x_2 = self.fRGB_2(x).flatten(2).permute(0, 2, 1)
        x_3 = self.fRGB_3(x).flatten(2).permute(0, 2, 1)
        x_4 = self.fRGB_4(x).flatten(2).permute(0, 2, 1)

        x = x_1 + self.pos_embed_1
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size[0])
        x = x.view(-1, self.window_size[0] * self.window_size[0], C)
        for blk in self.blocks_1:
            x = blk(x)
        x = x.view(-1, self.window_size[0], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], H, W).view(B, H * W, C)
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)

        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_2], dim=-1)

        x = x + self.pos_embed_2
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size[1])
        x = x.view(-1, self.window_size[1] * self.window_size[1], C)
        for blk in self.blocks_2:
            x = blk(x)
        x = x.view(-1, self.window_size[1], self.window_size[1], C)
        x = window_reverse(x, self.window_size[1], H, W).view(B, H * W, C)
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)

        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_3], dim=-1)

        x = x + self.pos_embed_3
        B, _, C = x.size()
        x = x.view(B, H, W, C)
        x = window_partition(x, self.window_size[2])
        x = x.view(-1, self.window_size[2] * self.window_size[2], C)
        for blk in self.blocks_3:
            x = blk(x)
        x = x.view(-1, self.window_size[2], self.window_size[2], C)
        x = window_reverse(x, self.window_size[2], H, W).view(B, H * W, C)
        _, _, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = nn.AvgPool2d(2)(x)

        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = torch.cat([x, x_4], dim=-1)

        x = x + self.pos_embed_4
        for blk in self.blocks_4:
            x = blk(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.last_block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x