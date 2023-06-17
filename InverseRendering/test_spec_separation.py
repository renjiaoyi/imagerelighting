import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import BuildDataset2
from model import SPECmodel

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='', help="root directory of the dataset")
parser.add_argument("--epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batchsize", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--cpus", type=int, default=4, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()

print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = SPECmodel()
net = net.to(device)

input_folder = './testingdata/spec/imgs'
mask_folder = './testingdata/spec/masks'

norm_transforms__ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transforms__ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

test_data_loader = DataLoader(
    BuildDataset2(
        input_folder=input_folder,
        mask_folder=mask_folder,
        norm_transforms_=norm_transforms__,
        transforms_=transforms__),
    batch_size=opt.batchsize,
    shuffle=False,
    num_workers=opt.cpus
)

mse_loss = torch.nn.MSELoss()

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


def test(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir+"/diff"):
        os.mkdir(save_dir+"/diff")
    if not os.path.exists(save_dir+"/spec"):
        os.mkdir(save_dir+"/spec")
    model = load_network(net, './path/specular.pth', device)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            input_img = data["input_img"].to(device)
            mask = data['mask_img'].to(device)

            origin_img = data['nonorm_img'][0]
            img_id = data['img_id'][0]

            spec_pred = model(input_img)

            spec_pred = spec_pred * mask

            spec_pred = spec_pred + 1
            spec_pred = spec_pred / 2
            spec_pred = torch.clip(spec_pred, min=0, max=1)
            spec_pred = spec_pred * mask

            spec_pred_img = spec_pred[0].cpu().numpy()
            input_img = input_img[0].cpu().numpy()
            spec_pred_img = np.transpose(spec_pred_img, (1, 2, 0))
            origin_img = np.transpose(origin_img, (1, 2, 0))
            input_img = np.transpose(input_img, (1, 2, 0))
            diffuse_pred_img = origin_img - spec_pred_img
            diffuse_pred_img = np.clip(diffuse_pred_img, 0, 1.0)

            spec_pred_img = np.uint8(spec_pred_img * 255)
            spec_pred_img = cv2.cvtColor(spec_pred_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/spec/{}.png'.format(save_dir, img_id), spec_pred_img)
            
            origin_img = np.uint8(origin_img * 255)
            origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)
            input_img = np.uint8(input_img * 255)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            diffuse_pred_img = np.uint8(diffuse_pred_img * 255)
            diffuse_pred_img = cv2.cvtColor(diffuse_pred_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('{}/diff/{}.png'.format(save_dir, img_id), diffuse_pred_img)
            print(i)


if __name__ == '__main__':
    test('specular_res')
