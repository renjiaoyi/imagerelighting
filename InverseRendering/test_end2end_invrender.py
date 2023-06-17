import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import NewCapturedDataset, BuildDataset
from model import InverseRenderModel, InverseRenderModelRGB
from utils import angular_loss, to_normal, weights_init_normal, load_network, LowRankLoss

from utils import RankLossFunc

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='', help="root directory of the dataset")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batchsize", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-7, help="initial learning rate")
parser.add_argument("--cpus", type=int, default=4, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_comment = 'test-{}'.format(opt.lr)
writer = SummaryWriter(comment=net_comment)

net = InverseRenderModel()
net.apply(weights_init_normal)
opt.dataroot = './testingdata/obj_monkey'
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))

norm_transforms_ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transforms_ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

norm_transforms__ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
transforms__ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

rankloss = RankLossFunc()

def load_network(net, load_path, device):
    print('loading the model from %s' % load_path)
    state_dict = torch.load(load_path, map_location=device)
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata

    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
    print('loaded the model from %s' % load_path)
    return net

def test(inited_net, model_path, save_dir='build', parent_dir=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir+"/normals"):
        os.mkdir(save_dir+"/normals")
    if not os.path.exists(save_dir+"/masks"):
        os.mkdir(save_dir+"/masks")
    if not os.path.exists(save_dir+"/albedos"):
        os.mkdir(save_dir+"/albedos")

    #parent_dir = '/home/Documents/Dataset'
    net = load_network(net=inited_net, load_path=model_path, device=device)
    net.to(device=device)

    norm_transforms__ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transforms__ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    data_loader = DataLoader(BuildDataset(
        input_folder=os.path.join(parent_dir, 'imgs'),
        mask_folder=os.path.join(parent_dir, 'masks'),
        norm_transforms_=norm_transforms__,
        transforms_=transforms__), batch_size=1, shuffle=False)

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img_id = data['img_id'][0]
            input_image = data["input_img"].to(device)
            mask_image = data["mask_img"].to(device)
            nonorm_image = data["nonorm_img"].to(device)
            ab, sd, norm, _= net.forward(input_image, nonorm_image, mask_image)

            normal_img_pred = (norm + 1.0) / 2.0
            normal_img_pred = normal_img_pred * mask_image
            normal_img = normal_img_pred[0].cpu()
            normal_img = normal_img.permute(1, 2, 0)
            normal_img = normal_img.numpy()
            normal_img = (normal_img * 255).astype(np.uint8)
            normal_img = cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/normals/{}_normal.png'.format(save_dir, img_id), normal_img)

            shading_img = (sd) * mask_image
            shading_img = shading_img[0].cpu()
            shading_img = shading_img.permute(1, 2, 0)
            shading_img = shading_img.numpy()
            shading_img = cv2.cvtColor(shading_img, cv2.COLOR_RGB2BGR)
            shading_img = shading_img/np.max(shading_img)
            shading_img = (shading_img * 255).astype(np.uint8)
            cv2.imwrite('{}/{}_shading.png'.format(save_dir, img_id), shading_img)

            albedo_img = ab[0].cpu()
            albedo_img = albedo_img.permute(1, 2, 0)
            albedo_img = albedo_img.numpy()
            albedo_img = albedo_img/np.max(albedo_img)
            albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_RGB2BGR)
            albedo_img = (albedo_img * 255).astype(np.uint8)
            cv2.imwrite('{}/albedos/{}_albedo.png'.format(save_dir, img_id), albedo_img)

if __name__ == "__main__":
    s = time.time()
    test(net, 'path/invrender.pth', 'monkey_res',opt.dataroot)
    e = time.time()
    print('Total time cost = %fs' % (e - s))
