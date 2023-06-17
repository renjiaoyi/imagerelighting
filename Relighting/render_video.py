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
from func import make_video, apply_bg, make_video_wide

dataroot ="./obj_tree"
batch_size = 1
hdrname = 'veranda'
lighting_dir='./coeffs/' +hdrname
cutmap = './cutmaps/'

output = dataroot+"_relit"
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


def prepare_sh(data):
    data = data.permute(0, 2, 3, 1)
    return data

def prepare_sp(data):
    data[:,0,:,:] = -2*data[:,0,:,:]*data[:,2,:,:]
    data[:,1,:,:] = -2*data[:,1,:,:]*data[:,2,:,:]
    data[:,2,:,:] = -2*data[:,2,:,:]*data[:,2,:,:] + 1
    data = data.permute(0, 2, 3, 1)
    return data

def sp_layer(nm, L_SHcoeffs, gamma):
    img = torch.zeros(1,3,1024,1024)
    c1 = torch.tensor(0.429043)
    c2 = torch.tensor(0.511664)
    c3 = torch.tensor(0.743125)
    c4 = torch.tensor(0.886227)
    c5 = torch.tensor(0.247708)
    L_SHcoeffs0 = L_SHcoeffs
    for i in range(0,9):
        L_SHcoeffs[:]=0
        L_SHcoeffs[:,i,:]=1

    
        M_row1 = torch.stack(
            [
                c1 * L_SHcoeffs[:, 8, :],
                c1 * L_SHcoeffs[:, 4, :],
                c1 * L_SHcoeffs[:, 7, :],
                c2 * L_SHcoeffs[:, 3, :],
            ],
            dim=1,
        )
        M_row2 = torch.stack(
            [
                c1 * L_SHcoeffs[:, 4, :],
                -c1 * L_SHcoeffs[:, 8, :],
                c1 * L_SHcoeffs[:, 5, :],
                c2 * L_SHcoeffs[:, 1, :],
            ],
            dim=1,
        )
        M_row3 = torch.stack(
            [
                c1 * L_SHcoeffs[:, 7, :],
                c1 * L_SHcoeffs[:, 5, :],
                c3 * L_SHcoeffs[:, 6, :],
                c2 * L_SHcoeffs[:, 2, :],
            ],
            dim=1,
        )
        M_row4 = torch.stack(
            [
                c2 * L_SHcoeffs[:, 3, :],
                c2 * L_SHcoeffs[:, 1, :],
                c2 * L_SHcoeffs[:, 2, :],
                c4 * L_SHcoeffs[:, 0, :] - c5 * L_SHcoeffs[:, 6, :],
            ],
            dim=1,
        )

        M = torch.stack([M_row1, M_row2, M_row3, M_row4], dim=1)
        total_npix = nm.size()[:3]
        ones = torch.ones(total_npix)
        nm_homo = torch.cat([nm, torch.unsqueeze(ones, dim=-1)], dim=-1)

        M = torch.unsqueeze(torch.unsqueeze(M, dim=1), dim=1)

        nm_homo = torch.unsqueeze(torch.unsqueeze(nm_homo, dim=-1), dim=-1)
        tmp = torch.sum(nm_homo * M, dim=-3)
        E = torch.sum(tmp * nm_homo[:, :, :, :, 0, :], dim=-2)

        imgnew = E.pow(gamma)
        imgnew2 = imgnew[:,:,:,0]*L_SHcoeffs0[0,i,0]+imgnew[:,:,:,1]*L_SHcoeffs0[0,i,1]+imgnew[:,:,:,2]*L_SHcoeffs0[0,i,2]
        imgnew2 = E*torch.unsqueeze(imgnew2, dim=3)
        img = imgnew2.permute(0, 3, 1, 2)+img

    return img

def sh_layer(am, nm, L_SHcoeffs):

    c1 = torch.tensor(0.429043)
    c2 = torch.tensor(0.511664)
    c3 = torch.tensor(0.743125)
    c4 = torch.tensor(0.886227)
    c5 = torch.tensor(0.247708)

    
    M_row1 = torch.stack(
        [
            c1 * L_SHcoeffs[:, 8, :],
            c1 * L_SHcoeffs[:, 4, :],
            c1 * L_SHcoeffs[:, 7, :],
            c2 * L_SHcoeffs[:, 3, :],
        ],
        dim=1,
    )
    M_row2 = torch.stack(
        [
            c1 * L_SHcoeffs[:, 4, :],
            -c1 * L_SHcoeffs[:, 8, :],
            c1 * L_SHcoeffs[:, 5, :],
            c2 * L_SHcoeffs[:, 1, :],
        ],
        dim=1,
    )
    M_row3 = torch.stack(
        [
            c1 * L_SHcoeffs[:, 7, :],
            c1 * L_SHcoeffs[:, 5, :],
            c3 * L_SHcoeffs[:, 6, :],
            c2 * L_SHcoeffs[:, 2, :],
        ],
        dim=1,
    )
    M_row4 = torch.stack(
        [
            c2 * L_SHcoeffs[:, 3, :],
            c2 * L_SHcoeffs[:, 1, :],
            c2 * L_SHcoeffs[:, 2, :],
            c4 * L_SHcoeffs[:, 0, :] - c5 * L_SHcoeffs[:, 6, :],
        ],
        dim=1,
    )

    M = torch.stack([M_row1, M_row2, M_row3, M_row4], dim=1)
    total_npix = nm.size()[:3]
    ones = torch.ones(total_npix)
    nm_homo = torch.cat([nm, torch.unsqueeze(ones, dim=-1)], dim=-1)

    M = torch.unsqueeze(torch.unsqueeze(M, dim=1), dim=1)

    nm_homo = torch.unsqueeze(torch.unsqueeze(nm_homo, dim=-1), dim=-1)
    tmp = torch.sum(nm_homo * M, dim=-3)
    E = torch.sum(tmp * nm_homo[:, :, :, :, 0, :], dim=-2)

    # compute intensity by product between irradiance and albedo
    i = E * am

    i = i.permute(0, 3, 1, 2)

    return i


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(lighting_dir)
dats = os.listdir(lighting_dir)

filenames = []

for i, data in enumerate(dataloader):
    filenames.append(data["file"][0])

if 1:
    for dat in dats:
        coeffs = np.loadtxt(os.path.join(lighting_dir, dat))
        L_SHcoeffs = torch.Tensor(coeffs)
        L_SHcoeffs = torch.unsqueeze(L_SHcoeffs, dim=0)
        s = time.time()
        for i, data in enumerate(dataloader):
            # Set model input
            albedo = Variable(data["albedo"]) * 0.1
            normal = Variable(data["normal"]) * 2 - 1  # xyz
            lm = torch.zeros_like(normal)
            lm = -normal
            mask = Variable(data["mask"])

            albedo_ = torch.zeros_like(albedo)
            

            rendered = (
                sh_layer(prepare_sh(albedo), prepare_sh(lm), L_SHcoeffs) * mask +0.05*sp_layer(prepare_sp(lm), L_SHcoeffs, 5) * mask
            )
            rendered = rendered/torch.max(rendered)
        
            save_image(rendered.detach().cpu().float(),"{}/{}_".format(output, i) + dat[0:-4]+".png")


for i in range(len(filenames)):
    apply_bg(dataroot+"/masks/"+filenames[i], hdrname, output, i, len(dats), cutmap)
    
    print(i)
    if not os.path.exists("{}/video".format(output)):
        os.mkdir("{}/video".format(output))
    if not os.path.exists("{}/video_wide".format(output)):
        os.mkdir("{}/video_wide".format(output))
    make_video_wide('{}/combined'.format(output), hdrname, "{}/video/{}_{}_{}.mp4".format(output, i, hdrname, len(dats)), i,512)

    