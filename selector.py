import sys
sys.path.append("./thirdparty/RAFT/core")

import os
from os.path import join as pjoin
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
import torch.nn.functional as F

from extractor import BasicEncoder

def load_image(imfile, device, size=None):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    if size is not None:
        img = v2.functional.resize(img, size=size)
    return img[None].to(device)

class CovisHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        # 1x1 convolution
        self.conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """x: n, c, h, w"""
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.sigmoid(self.conv2(out))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_dim)
        )
        self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa

# ASPP 模块，用于多尺度上下文特征融合
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.out_conv(x_cat)


class EnhancedCovisHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.block = nn.Sequential(
            ResidualBlock(input_dim, 256),   # 第一个残差块
            CBAM(256),                       # 第一个 CBAM 模块
            ResidualBlock(256, 256, dilation=2),  # 扩张残差块，扩大感受野
            CBAM(256),                       # 第二个 CBAM 模块
            ASPP(256, 128),                  # ASPP 模块，多尺度信息融合
            nn.Conv2d(128, 1, kernel_size=1) 
        )

    def forward(self, x):
        return self.sigmoid(self.block(x))

class OffsetGenerator(nn.Module):
    """深层 offset 生成模块，kernel_size=3 对应 offset 通道数 = 2*3*3 = 18"""
    def __init__(self, in_channels, kernel_size=3, padding=1):
        super().__init__()
        offset_channels = 2 * kernel_size * kernel_size
        self.offset_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, offset_channels, kernel_size=3, padding=padding, bias=True)
        )
    
    def forward(self, x):
        return self.offset_gen(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 模块，自适应通道加权"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RecognizeHead(nn.Module):
    def __init__(self, input_dim, hidden_dim1=16, hidden_dim2=16):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim1, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(num_features=hidden_dim1)
        self.conv2 = nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(num_features=hidden_dim2)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """x: n, c, h, w"""
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        # Average. n, c, h, w -> n, c
        n, c, h, w = out.shape
        out = out.view(n, c, -1).mean(-1, keepdim=False)
        out = self.sigmoid(self.fc(out))
        return out

class FrameSelector(nn.Module):
    def __init__(self, output_size, use_fmap=False, add_recognize_head=False):
        super().__init__()

        assert output_size in [[3, 4], [6, 8], [12, 16]]

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')
        self.use_fmap = use_fmap
        if use_fmap:
            fmapdims = 256
        else:
            fmapdims = 0
        
        if output_size == [3, 4]:
            self.maxpool = nn.AvgPool2d(8, 8)
            corrdims = 12
        if output_size == [6, 8]:
            self.maxpool = nn.AvgPool2d(4, 4)
            corrdims = 48
        if output_size == [12, 16]:
            self.maxpool = nn.AvgPool2d(2, 2)
            corrdims = 192

        self.covis_head = EnhancedCovisHead(input_dim=corrdims + fmapdims)

        self.recognize_head = None
        if add_recognize_head:
            self.recognize_head = RecognizeHead(input_dim=corrdims + fmapdims)

        # ############ SelfSupervisedHomography #############
        # self.correlation = nn.Sequential(
        #     nn.Conv2d(256 *2, 128, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 3, padding=1)
        # )
        # self.regressor = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 9)
        # )        
        # self.fc = nn.Sequential(
        #     nn.Linear(1, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid()
        # )
        
    def homography(self, feat1, feat2):
        corr_feat = self.correlation(torch.cat([feat1, feat2], dim=1))
        H_params = self.regressor(corr_feat)
        return H_params.reshape(-1, 3, 3)


    def corr(self, fmap1, fmap2):
        n, c, h1, w1 = fmap1.shape
        
        corr = torch.einsum('nchw,nckl->nhwkl', fmap1, fmap2)
        return corr / torch.sqrt(torch.tensor(c, device=fmap1.device, dtype=torch.float32))

    def forward(self, image1, image2, gt1=None, gt2=None):
        # tik = perf_counter()
        fmap1, fmap2 = self.fnet([image1, image2])
        # tok = perf_counter()
        # print(f"Encoder: {(tok - tik) * 1e3:.3f} ms")
        
        # fmap1 = self.learn_pool(fmap1)
        # fmap2 = self.learn_pool(fmap2)

        fmap1 = self.maxpool(fmap1)
        fmap2 = self.maxpool(fmap2)

        # H12 = self.homography(fmap1, fmap2)
        # H21 = self.homography(fmap2, fmap1)

        # overlap_ratio = (gt1 * gt2).sum(dim=[1,2,3]) / (gt1.sum(dim=[1,2,3]) + 1e-5)
        
        # weights = self.fc(overlap_ratio.unsqueeze(1))
        # alpha = weights.squeeze()

        corr = self.corr(fmap1, fmap2)
        # corr = self.corr_module(fmap1, fmap2)

        n, h, w, _, _ = corr.shape
        corr_map1 = corr.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        corr_map2 = corr.reshape(n, -1, h, w)

        if self.use_fmap:
            corr_map1 = torch.concatenate( [corr_map1, fmap1], 1 )
            corr_map2 = torch.concatenate( [corr_map2, fmap2], 1 )

        # pdb.set_trace()
        out1 = self.covis_head(corr_map1)
        out2 = self.covis_head(corr_map2)

        # if self.recognize_head is not None:
        #     rscore1 = self.recognize_head(corr_map1)
        #     rscore2 = self.recognize_head(corr_map2)
        # else:
        #     return out1, out2, H12, H21, alpha

        # return out1, out2, rscore1, rscore2
        return out1, out2
    
    @staticmethod
    def load_model(experiment_dir, eval=True):
        if 'recognizer' in experiment_dir:
            rec_ckpt_path = pjoin(experiment_dir, 'recognize_head_ckpt.pt')
            rec_ckpt = torch.load(rec_ckpt_path)
            with open( pjoin(experiment_dir, 'config.json'), 'r' ) as f:
                rec_config = json.load(f)

            base_dir = os.path.split(experiment_dir)[0]
            base_ckpt_path = pjoin(base_dir, 'selector_ckpt.pt')
            base_ckpt = torch.load(base_ckpt_path)
            with open( pjoin(base_dir, 'config.json'), 'r' ) as f:
                config = json.load(f)

            model = FrameSelector(output_size=config['outsize'], use_fmap=config.get('use_fmap', False), add_recognize_head=True)
            ret1 = model.load_state_dict(base_ckpt, strict=False)
            ret2 = model.recognize_head.load_state_dict(rec_ckpt, strict=False)
            return model, config, rec_config
            
        else:
            ckpt_path = pjoin(experiment_dir, "selector_ckpt.pt")
            with open( pjoin(experiment_dir, 'config.json'), 'r' ) as f:
                config = json.load(f)

            model = FrameSelector(output_size=config['outsize'], use_fmap=config.get('use_fmap', False))
            ckpt = torch.load(ckpt_path)
            ret = model.load_state_dict(ckpt, strict=False)
            for k in ret.unexpected_keys:
                assert k.split('.')[0] in ['correlation', 'regressor', 'fc']

            return model, config