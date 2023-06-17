import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import SPECdataset
from model import SPECmodel
from utils import weights_init_normal, RankLossFunc_Spec

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='', help="root directory of the dataset")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batchsize", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-6, help="initial learning rate")
parser.add_argument("--cpus", type=int, default=4, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_comment = 'specular-{}'.format(opt.lr)
writer = SummaryWriter(comment=net_comment)

net = SPECmodel()
net = load_network(net, load_path='path/pretrain.pth', device=device)
net.cuda()

opt.dataroot = '/spec_data'
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
norm_transforms__ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transforms__ = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
train_data_loader = DataLoader(
    SPECdataset(
        opt.dataroot,
        opt.batchsize,
        norm_transforms_=norm_transforms__,
        transforms_=transforms__),
    batch_size=opt.batchsize,
    shuffle=False,
    num_workers=opt.cpus
)

mse_loss = torch.nn.MSELoss()
rankloss = RankLossFunc_Spec()


def start_train(net):
    net.to(device=device)
    save_path = "./path/" + net_comment
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in range(opt.epochs):
        train_once(net, epoch, train_data_loader)
        print("Epoch %03d/%03d --- " % (epoch + 1, opt.epochs))
        if epoch%100 == 0:
            torch.save(net.state_dict(), os.path.join(save_path, "{}.pth".format(epoch)))

def train_once(model, epoch, data_loader):
    model.train()
    len_batch = len(data_loader)
    total_loss = 0.0

    for i, data in enumerate(data_loader):

        input_img = data["input_tensors"][0].to(device)
        input_nonm_img = data['input_nonorm_tensors'][0].to(device)
        mask = data['mask_tensor'][0].to(device)

        optimizer.zero_grad()
        spec_pred = model(input_img)
        spec_pred = spec_pred*mask

        spec_pred = spec_pred +1
        spec_pred = spec_pred/2
        spec_pred = torch.clip(spec_pred, min=0,max=1)
        spec_pred = spec_pred *mask

        loss = rankloss(input_nonm_img,spec_pred, mask)
        
        iteration = epoch * len_batch + i

        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            loss.backward()

        optimizer.step()
        writer.add_scalar("loss", loss.item(), iteration)
        if i % 10 == 0:
            print("TRAINING epoch = {}, iteration = {}, loss={}".format(epoch, iteration, loss.item()))
    print('TRAINING epoch={}, mean loss={}'.format(epoch, total_loss / len_batch))


if __name__ == "__main__":
    start_train(net)