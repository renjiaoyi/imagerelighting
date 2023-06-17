import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import NewCapturedDataset, BuildDataset
from model import InverseRenderModel, InverseRenderModelRGB

from utils import load_network, LowRankLoss, RankLossFunc

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='', help="root directory of the dataset")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batchsize", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--cpus", type=int, default=4, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_comment = 'jointtraining-{}'.format(opt.lr)
writer = SummaryWriter(comment=net_comment)

IvModel = InverseRenderModel(pretrain='path/initial_2.pth')
IvModel = load_network(net=IvModel, load_path='path/round1.pth', device=device)

opt.dataroot = '/home/Documents/RelitDataset'
opt.testroot = '/home/Documents/Dataset'
optimizer = torch.optim.Adam(IvModel.parameters(), lr=opt.lr, betas=(0.9, 0.999))

norm_transforms_ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transforms_ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

train_data_loader = DataLoader(
    NewCapturedDataset(
        opt.dataroot,
        opt.batchsize,
        norm_transforms_=norm_transforms_,
        transforms_=transforms_),
    batch_size=1,
    shuffle=True,
    num_workers=opt.cpus
)

norm_transforms__ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
transforms__ = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
test_data_loader = DataLoader(BuildDataset(
        input_folder=os.path.join(opt.testroot, 'imgs'),
        mask_folder=os.path.join(opt.testroot, 'masks'),
        norm_transforms_=norm_transforms__,
        transforms_=transforms__), batch_size=1, shuffle=False)
rankloss = LowRankLoss()

def start_train(net):
    net.to(device=device)
    save_path = "./path/" + net_comment
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(opt.epochs):
        train_once(net, epoch, train_data_loader)
        print("Epoch %03d/%03d --- " % (epoch + 1, opt.epochs))
        torch.save(net.state_dict(), os.path.join(save_path, "{}.pth".format(epoch)))

def test_during_training(model, data_loader):
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img_id = data['img_id'][0]
            input_image = data["input_img"].to(device)
            mask_image = data["mask_img"].to(device)
            nonorm_image = data["nonorm_img"].to(device)
            ab, sd, norm, _ = model.forward(input_image, nonorm_image, mask_image)

            normal_img_pred = (norm + 1.0) / 2.0
            normal_img_pred = normal_img_pred * mask_image
            normal_img = normal_img_pred[0].cpu()
            normal_img = normal_img.permute(1, 2, 0)
            normal_img = normal_img.numpy()
            normal_img = (normal_img * 255).astype(np.uint8)
            writer.add_image(img_id+"_normal", normal_img, dataformats='HWC')

            shading_img = (sd) * mask_image
            shading_img = shading_img[0].cpu()
            shading_img = shading_img.permute(1, 2, 0)
            shading_img = shading_img.numpy()
            shading_img = shading_img/np.max(shading_img)
            shading_img = (shading_img * 255).astype(np.uint8)
            writer.add_image(img_id+"_shading", shading_img, dataformats='HWC')
 
            albedo_img = ab[0].cpu()
            albedo_img = albedo_img.permute(1, 2, 0)
            albedo_img = albedo_img.numpy()
            albedo_img = albedo_img/np.max(albedo_img)
            albedo_img = (albedo_img * 255).astype(np.uint8)
            writer.add_image(img_id+"_albedo", albedo_img, dataformats='HWC')




def train_once(model, epoch, data_loader):
    model.train()
    len_batch = len(data_loader)
    total_loss = 0.0

    for i, data in enumerate(data_loader):

        input_img = data["input_tensors"][0].to(device)
        input_nonm_img = data['input_nonorm_tensors'][0].to(device)
        mask = data['mask_tensor'][0].to(device)
        prior = data['prior_nonorm'][0].to(device).expand_as(input_nonm_img)

        optimizer.zero_grad()
        ab_pred, _, _, coeff = model(input_img, input_nonm_img, mask)
        ab_pred = torch.clip(ab_pred, 0, 1.0)

        loss1 = rankloss(ab_pred, mask)
        loss2 = normloss(prior, ab_pred)
        loss = loss1
        iteration = epoch * len_batch + i
        if iteration%500 == 0:
            model.eval()
            ab, sd, norm, _ = model(input_img, input_nonm_img, mask)
            for j in range(12):
                normal_img = (norm[j]+1.0)/2.0
                normal_img = normal_img * mask
                normal_img = normal_img.detach().cpu()
                normal_img = normal_img.permute(1, 2, 0)
                normal_img = normal_img.numpy()
                normal_img = (normal_img * 255).astype(np.uint8)
                normal_img = cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('test/{}_{}_normal.png'.format(str(iteration), str(j)), normal_img)

                shading_img = (sd[j]) * mask
                shading_img = shading_img.detach().cpu()
                shading_img = shading_img.permute(1, 2, 0)
                shading_img = shading_img.numpy()
                shading_img = shading_img/np.max(shading_img)
                shading_img = (shading_img * 255).astype(np.uint8)
                shading_img = cv2.cvtColor(shading_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('test/{}_{}_shading.png'.format(str(iteration), str(j)), shading_img)
    
                albedo_img = ab[j].detach().cpu()
                albedo_img = albedo_img.permute(1, 2, 0)
                albedo_img = albedo_img.numpy()
                albedo_img = np.clip(albedo_img, 0, 1.0)
                albedo_img = (albedo_img * 255).astype(np.uint8)
                albedo_img = cv2.cvtColor(albedo_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('test/{}_{}_albedo.png'.format(str(iteration), str(j)), albedo_img)
            test_during_training(model, test_data_loader)
            model.train()
        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            loss.backward()

        optimizer.step()

        writer.add_scalar("loss", loss.item(), iteration)
        writer.add_scalar("rank-loss", loss1.item(), iteration)
        f i % 10 == 0:
            print("TRAINING epoch = {}, iteration = {}, loss={}".format(epoch, iteration, loss.item()))
    print('TRAINING epoch={}, mean loss={}'.format(epoch, total_loss / len_batch))

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

    parent_dir = '/home/Documents/Dataset'
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
        input_folder=os.path.join(parent_dir, 'Input'),
        mask_folder=os.path.join(parent_dir, 'Mask'),
        norm_transforms_=norm_transforms__,
        transforms_=transforms__), batch_size=1, shuffle=False)

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            img_id = data['img_id'][0]
            input_image = data["input_img"].to(device)
            mask_image = data["mask_img"].to(device)
            nonorm_image = data["nonorm_img"].to(device)
            ab, sd, norm, _ = model.forward(input_image, nonorm_image, mask_image)

            normal_img_pred = (norm + 1.0) / 2.0
            normal_img_pred = normal_img_pred * mask_image
            normal_img = normal_img_pred[0].cpu()
            normal_img = normal_img.permute(1, 2, 0)
            normal_img = normal_img.numpy()
            normal_img = (normal_img * 255).astype(np.uint8)
            writer.add_image(img_id+"_normal", normal_img, dataformats='HWC')
            cv2.imwrite('{}/{}_normal.png'.format(save_dir, img_id), normal_img)

            shading_img = (sd) * mask_image
            shading_img = shading_img[0].cpu()
            shading_img = shading_img.permute(1, 2, 0)
            shading_img = shading_img.numpy()
            shading_img = shading_img/np.max(shading_img)
            shading_img = (shading_img * 255).astype(np.uint8)
            cv2.imwrite('{}/{}_shading.png'.format(save_dir, img_id), shading_img)

            albedo_img = ab[0].cpu()
            albedo_img = albedo_img.permute(1, 2, 0)
            albedo_img = albedo_img.numpy()
            albedo_img = albedo_img/np.max(albedo_img)
            albedo_img = (albedo_img * 255).astype(np.uint8)
            cv2.imwrite('{}/{}_albedo.png'.format(save_dir, img_id), albedo_img)

if __name__ == "__main__":
    start_train(IvModel)