import torch
import torch.nn as nn
from utils import weights_init_normal, render_layer_RGB, load_network, to_normal, render_layer, render_layer_color

class SPECmodel(nn.Module):
    def __init__(self):
        super(SPECmodel, self).__init__()

        # Encoder layers
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.InstanceNorm2d(16), nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )

        # Decoder layers
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )

        self.deconv0 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(256 * 2, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(128 * 2, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv4 = nn.Sequential(
            nn.Conv2d(32 * 2, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(16 * 2, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.InstanceNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        xmid = torch.cat([self.mid(x5)] + [x5], 1)

        x0d = torch.cat([self.deconv0(xmid)] + [x4], 1)
        x1d = torch.cat([self.deconv1(x0d)] + [x3], 1)
        x2d = torch.cat([self.deconv2(x1d)] + [x2], 1)
        x3d = torch.cat([self.deconv3(x2d)] + [x1], 1)
        x4d = torch.cat([self.deconv4(x3d)] + [x0], 1)

        x_out = [self.output(x4d)]
        if len(x_out) > 1:
            return tuple(x_out)
        else:
            return x_out[0]

class INmodel(nn.Module):
    def __init__(self):
        super(INmodel, self).__init__()

        # Encoder layers
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.InstanceNorm2d(16), nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )

        # Decoder layers
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )

        self.deconv0 = nn.Sequential(
            nn.Conv2d(256 * 2, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(256 * 2, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv2 = nn.Sequential(
            nn.Conv2d(128 * 2, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.deconv4 = nn.Sequential(
            nn.Conv2d(32 * 2, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(16 * 2, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 2, 3, 1, 1),
            nn.InstanceNorm2d(2),
            nn.Tanh(),
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        xmid = torch.cat([self.mid(x5)] + [x5], 1)

        x0d = torch.cat([self.deconv0(xmid)] + [x4], 1)
        x1d = torch.cat([self.deconv1(x0d)] + [x3], 1)
        x2d = torch.cat([self.deconv2(x1d)] + [x2], 1)
        x3d = torch.cat([self.deconv3(x2d)] + [x1], 1)
        x4d = torch.cat([self.deconv4(x3d)] + [x0], 1)

        x_out = [self.output(x4d)]

        if len(x_out) > 1:
            return tuple(x_out)
        else:
            return x_out[0]


class InverseRenderModel(nn.Module):
    def __init__(self, pretrain=None):
        super(InverseRenderModel, self).__init__()
        self.apply(weights_init_normal)
        self.normalModel = INmodel()
        #self.IRModel = InverseRenderModel()
        #self.normalModel.apply(weights_init_normal)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pretrain is not None:
            self.normalModel = load_network(net=self.normalModel, load_path=pretrain, device=device)
        for p in self.parameters():
            p.requires_grad = False
        for p in self.normalModel.parameters():
            p.requires_grad = True
        #for q in self.IRModel.parameters():
            #q.requires_grad = False
        # Encoder layers
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.InstanceNorm2d(16), nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )
        self.L_SHcoeffs = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 12),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    
    def forward(self, x, x_norm, mask):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        xmid = torch.cat([self.mid(x5)] + [x5], 1)
        #xmid = self.avgpool(xmid)
        xmid = torch.flatten(xmid, 1)

        zeros = torch.zeros([x.size()[0], 1]).cuda()
        ones = torch.ones([x.size()[0], 1]).cuda()
        feature = self.L_SHcoeffs(xmid)
        coeff = feature[:,0:9]
        color = feature[:,9:12]
        coeff_norm = torch.linalg.norm(coeff, ord=2, dim=1, keepdim=True)#.detach()
        coeff = coeff / coeff_norm
        coe_output = coeff
        coeff = coeff.unsqueeze(2)
        coeff = torch.cat((coeff,coeff,coeff), 2)
        normal = self.normalModel(x)
        normal = to_normal(normal, mask)

        shading, shading_norm = render_layer(normal.permute(0, 2, 3, 1), coeff)# + 1
        albedo = x_norm/(shading)

        return albedo, shading_norm, normal, coe_output


class InverseRenderModelRGB(nn.Module):
    def __init__(self, pretrain=None):
        super(InverseRenderModelRGB, self).__init__()
        #self.apply(weights_init_normal)
        self.normalModel = INmodel()
        #self.IRModel = InverseRenderModel()
        #self.normalModel.apply(weights_init_normal)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pretrain is not None:
            self.normalModel = load_network(net=self.normalModel, load_path=pretrain, device=device)
        for p in self.normalModel.parameters():
            p.requires_grad = False
        for q in self.IRModel.parameters():
            q.requires_grad = False
        # Encoder layers
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.InstanceNorm2d(16), nn.ReLU(True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 2, 1), nn.InstanceNorm2d(32), nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), nn.InstanceNorm2d(64), nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.ReLU(True)
        )
        self.mid = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
        )
        self.L_SHcoeffs = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 24),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    
    def forward(self, x, x_norm, mask):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        xmid = torch.cat([self.mid(x5)] + [x5], 1)
        #xmid = self.avgpool(xmid)
        xmid = torch.flatten(xmid, 1)

        zeros = torch.zeros([x.size()[0], 1, 3]).cuda()
        #coeff = torch.cat([zeros, self.L_SHcoeffs(xmid)], dim=1)
        coeff = self.L_SHcoeffs(xmid)
        coeff = coeff.view(-1, 8, 3)
        coeff = torch.cat([zeros, coeff], dim=1)
        coe_output = coeff[:, :, 0]
        normal = self.normalModel(x)
        normal = to_normal(normal, mask)

        shading, shading_norm = render_layer_RGB(normal.permute(0, 2, 3, 1), coeff)# + 1
        albedo = x_norm/(shading+1)

        return albedo, shading_norm, normal, coe_output




