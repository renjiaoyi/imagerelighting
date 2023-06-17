import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import glob

from func import make_video, apply_bg, make_video_wide, apply_bg_naive

dataroot = "./obj_tree"
batch_size = 1
hdrname = 'dichololo_ni'
lighting_dir='./coeffs/' +hdrname
cutmap = './cutmaps/'

output = dataroot+"_results"
if not os.path.exists(output):
    os.mkdir(output)

class ANDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.albedo_files = glob.glob(root + "/albedos/*.png")
        self.normal_files = glob.glob(root + "/normals/*.png")
        self.mask_files = glob.glob(root + "/masks/*.png")

    def __getitem__(self, index):
        albedo = Image.open(self.albedo_files[index])
        albedo = albedo.convert("RGB")
        normal = Image.open(self.normal_files[index])
        normal = normal.convert("RGB")
        mask = Image.open(self.mask_files[index])
        mask = mask.convert("RGB")

        a_item = self.transform(albedo)
        n_item = self.transform(normal)
        m_item = self.transform(mask)

        file_name = self.albedo_files[index].split("/")[-1]

        return {"albedo": a_item, "normal": n_item, "mask": m_item, "file": file_name}

    def __len__(self):
        return max(len(self.albedo_files), len(self.normal_files), len(self.mask_files))


# Dataset loader
transforms_ = [transforms.Resize((1024, 1024)), transforms.ToTensor()]

dataloader = DataLoader(
    ANDataset(dataroot, transforms_=transforms_), batch_size=batch_size, shuffle=False
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(lighting_dir)
dats = os.listdir(lighting_dir)
s = time.time()
filenames = []

for i, data in enumerate(dataloader):
    filenames.append(data["file"][0])


end = time.time()
i=0
for i in range(len(filenames)):
    apply_bg_naive(dataroot+"/masks/"+filenames[i], hdrname, dataroot, i, len(dats), cutmap)
    if not os.path.exists("./{}/video".format(output)):
        os.mkdir("./{}/video".format(output))
    make_video_wide('./{}/combined'.format(dataroot), hdrname, "./{}/video/{}_{}_{}_wide_naive.mp4".format(output, i, hdrname, len(dats)), i,512)


    