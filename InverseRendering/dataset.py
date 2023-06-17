import glob
import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset



def make_dataset(folder):
    images = []
    file_list = sorted(os.listdir(folder))
    for img_name in file_list:
        img_path = os.path.join(folder, img_name)
        images.append(img_path)
    return images

class MITDataset(Dataset):
    def __init__(self, input_folder, norm_transforms_, transforms_):
        super(MITDataset, self).__init__()
        self.root = input_folder
        self.norm_transform = norm_transforms_
        self.transform = transforms_
        self.img_files = glob.glob(self.root + "/Diffuse/*.png")
        self.length = len(self.img_files)
    
    def __getitem__(self, index):
        img_path = self.img_files[index]
        img_name = img_path.split('/')[-1]

        mask_path = self.root + '/Mask/' + img_name
        shading_path = self.root + '/Shading/' + img_name

        img = Image.open(img_path).convert('RGB')
        input_tensor = self.norm_transform(img)
        input_nonorm = self.transform(img)

        mask = Image.open(mask_path).convert('L')
        mask_tensor = self.transform(mask)
        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        shading = Image.open(shading_path).convert('RGB')
        shading_tensor = self.transform(shading)

        id = torch.IntTensor(1)
        id[0] = int(index)

        return {'mask_tensor': mask_tensor,
                'input_tensors': input_tensor,
                'input_nonorm_tensors': input_nonorm,
                'shading_tensor': shading_tensor,
                'image_id':id
                }

    def __len__(self):
        return self.length

class BuildDataset2(Dataset):
    def __init__(self, input_folder, mask_folder, norm_transforms_, transforms_):
        super(BuildDataset2, self).__init__()
        self.mask_folder_dataset = make_dataset(folder=mask_folder)
        self.input_folder_dataset = make_dataset(folder=input_folder)

        self.norm_transforms_ = norm_transforms_
        self.transforms_ = transforms_
        self.length = len(self.input_folder_dataset)

    def __getitem__(self, index):
        #mask_path = self.mask_folder_dataset[int(index/12)]
        mask_path = self.mask_folder_dataset[index]
        input_path = self.input_folder_dataset[index]

        start_index = input_path.rfind('/') + 1
        end_index = input_path.rfind('.')
        img_id = input_path[start_index:end_index]

        mask_data = Image.open(mask_path).convert('L')
        input_data = Image.open(input_path).convert('RGB')
        diffuse_data = input_data.copy()
        diffuse_data = self.transforms_(diffuse_data)

        input_tensor = self.norm_transforms_(input_data)
        mask_tensor = self.transforms_(mask_data)

        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        input_tensor = input_tensor * mask_tensor

        entry = {'input_img': input_tensor,
                 'img_id': img_id,
                 'mask_img': mask_tensor,
                 'nonorm_img':diffuse_data}
        return entry

    def __len__(self):
        return self.length

class BuildDataset_eval_light(Dataset):
    def __init__(self, input_folder, mask_folder, normal_folder, norm_transforms_, transforms_):
        super(BuildDataset_eval_light, self).__init__()
        self.mask_folder_dataset = make_dataset(folder=mask_folder)
        self.input_folder_dataset = make_dataset(folder=input_folder)
        self.normal_folder_dataset = make_dataset(folder=normal_folder)
        self.norm_transforms_ = norm_transforms_
        self.transforms_ = transforms_
        self.length = len(self.input_folder_dataset)

    def __getitem__(self, index):
        mask_path = self.mask_folder_dataset[index]
        input_path = self.input_folder_dataset[index]
        normal_path = self.normal_folder_dataset[index]
        start_index = input_path.rfind('/') + 1
        end_index = input_path.rfind('.')
        img_id = input_path[start_index:end_index]

        mask_data = Image.open(mask_path).convert('L')
        input_data = Image.open(input_path).convert('RGB')
        normal_data = Image.open(normal_path).convert('RGB')
        diffuse_data = input_data.copy()
        diffuse_data = self.transforms_(diffuse_data)

        input_tensor = self.norm_transforms_(input_data)
        normal_tensor = self.norm_transforms_(normal_data)
        mask_tensor = self.transforms_(mask_data)

        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        input_tensor = input_tensor * mask_tensor
        normal_tensor = normal_tensor * mask_tensor
        entry = {'input_img': input_tensor,
                 'img_id': img_id,
                 'mask_img': mask_tensor,
                 'normal_img':normal_tensor,
                 'nonorm_img':diffuse_data}
        return entry

    def __len__(self):
        return self.length

class BuildDataset(Dataset):
    def __init__(self, input_folder, mask_folder, norm_transforms_, transforms_):
        super(BuildDataset, self).__init__()
        self.mask_folder_dataset = make_dataset(folder=mask_folder)
        self.input_folder_dataset = make_dataset(folder=input_folder)

        self.norm_transforms_ = norm_transforms_
        self.transforms_ = transforms_
        self.length = len(self.input_folder_dataset)

    def __getitem__(self, index):
        mask_path = self.mask_folder_dataset[index]
        input_path = self.input_folder_dataset[index]

        start_index = input_path.rfind('/') + 1
        end_index = input_path.rfind('.')
        img_id = input_path[start_index:end_index]

        mask_data = Image.open(mask_path).convert('L')
        input_data = Image.open(input_path).convert('RGB')
        diffuse_data = input_data.copy()
        diffuse_data = self.transforms_(diffuse_data)

        input_tensor = self.norm_transforms_(input_data)
        mask_tensor = self.transforms_(mask_data)

        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        input_tensor = input_tensor * mask_tensor

        entry = {'input_img': input_tensor,
                 'img_id': img_id,
                 'mask_img': mask_tensor,
                 'nonorm_img':diffuse_data}
        return entry

    def __len__(self):
        return self.length

class SPECdataset(Dataset):
    def __init__(self, root, batchsize,norm_transforms_=None, transforms_=None):
        self.norm_transform = norm_transforms_
        self.transform = transforms_
        self.img_files = glob.glob(root + "/imgs/*/*.png")
        self.length = len(self.img_files)
        self.root = root
        self.batch_size = batchsize

        categories_map = {}
        imgs_folder = os.path.join(root, 'imgs')
        img_categories = os.listdir(imgs_folder)
        for category_dir in img_categories:
            category_path = os.path.join(imgs_folder, category_dir)
            if os.path.isdir(category_path):
                img_num = len(os.listdir(category_path))
                categories_map[category_dir] = img_num

        self.categories_map = categories_map


    def __getitem__(self, index):
        img_path = self.img_files[index]
        category_dir = img_path.split('/')[-2]
        mask_path = self.root + '/masks/' + category_dir + '.png'
        #prior_path = self.root + '/mean_img/' + category_dir + '.png'
        #prior_img = Image.open(prior_path).convert('RGB')
        #prior_tensor = self.norm_transform(prior_img)
        #prior_nonorm = self.transform(prior_img)

        img_num = self.categories_map[category_dir]

        mask = Image.open(mask_path).convert('L')
        mask_tensor = self.transform(mask)
        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        _, H, W = mask_tensor.shape

        count = 0
        input_tensors = torch.zeros((self.batch_size, 3, H, W))
        input_nonorm_tensors = torch.zeros((self.batch_size, 3, H, W))

        input_paths = []

        while count < self.batch_size:
            random_img_path = select_random_img_path(self.root, category_dir, img_num)
            random_img = Image.open(random_img_path).convert('RGB')

            input_paths.append(random_img_path)

            input_tensor = self.norm_transform(random_img)
            input_nonorm = self.transform(random_img)

            input_tensor = input_tensor * mask_tensor
            input_nonorm = input_nonorm * mask_tensor

            input_tensors[count] = input_tensor
            input_nonorm_tensors[count] = input_nonorm

            count += 1

        return {'mask_tensor': mask_tensor,
                'input_tensors': input_tensors,
                'input_nonorm_tensors': input_nonorm_tensors,
                'input_paths': input_paths,
                'category': category_dir,
                #'prior_tensor':prior_tensor,
                #'prior_nonorm':prior_nonorm,
                }

    def __len__(self):
        return self.length

class LIMEDataset(Dataset):
    def __init__(self, parent_folder, norm_transforms_, transforms_):
        super(LIMEDataset, self).__init__()

        mask_folder = os.path.join(parent_folder, 'Mask')
        normal_folder = os.path.join(parent_folder, 'Normal')
        input_folder = os.path.join(parent_folder, 'Input_Masked')

        self.mask_folder_dataset = make_dataset(folder=mask_folder)
        self.normal_folder_dataset = make_dataset(folder=normal_folder)
        self.input_folder_dataset = make_dataset(folder=input_folder)

        self.norm_transforms_ = norm_transforms_
        self.transforms_ = transforms_
        self.length = len(self.mask_folder_dataset)

    def __getitem__(self, index):
        mask_path = self.mask_folder_dataset[index]
        normal_path = self.normal_folder_dataset[index]
        input_path = self.input_folder_dataset[index]

        mask_data = Image.open(mask_path).convert('L')
        normal_data = Image.open(normal_path).convert('RGB')
        input_data = Image.open(input_path).convert('RGB')

        input_tensor = self.norm_transforms_(input_data)
        normal_tensor = self.transforms_(normal_data)
        mask_tensor = self.transforms_(mask_data)

        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        input_tensor = input_tensor * mask_tensor
        normal_tensor = normal_tensor * mask_tensor

        entry = {'normal_img': normal_tensor,
                 'input_img': input_tensor,
                 'mask_img': mask_tensor}
        return entry

    def __len__(self):
        return self.length

def select_random_img_path(root_path, category_dir, img_num):
    while True:
        inx = str(random.randint(0, img_num - 1))
        random_img_path = root_path + '/imgs/' + category_dir + '/' + inx + '.png'
        if os.path.exists(random_img_path):
            break

        random_img_path = root_path + '/imgs/' + category_dir + '/' + inx + '.jpg'
        if os.path.exists(random_img_path):
            break
        else:
            continue
    return random_img_path

class NewCapturedDataset(Dataset):
    def __init__(self, root, batchsize,norm_transforms_=None, transforms_=None):
        self.norm_transform = norm_transforms_
        self.transform = transforms_
        self.img_files = glob.glob(root + "/imgs/*/*.png")
        self.length = len(self.img_files)
        self.root = root
        self.batch_size = batchsize

        categories_map = {}
        imgs_folder = os.path.join(root, 'imgs')
        img_categories = os.listdir(imgs_folder)
        for category_dir in img_categories:
            category_path = os.path.join(imgs_folder, category_dir)
            if os.path.isdir(category_path):
                img_num = len(os.listdir(category_path))
                categories_map[category_dir] = img_num

        self.categories_map = categories_map


    def __getitem__(self, index):
        img_path = self.img_files[index]
        category_dir = img_path.split('/')[-2]
        mask_path = self.root + '/masks/' + category_dir + '.png'
        prior_path = self.root + '/mean_img/' + category_dir + '.png'
        prior_img = Image.open(prior_path).convert('RGB')
        prior_tensor = self.norm_transform(prior_img)
        prior_nonorm = self.transform(prior_img)

        img_num = self.categories_map[category_dir]

        mask = Image.open(mask_path).convert('L')
        mask_tensor = self.transform(mask)
        mask_tensor[mask_tensor < 0.5] = 0
        mask_tensor[mask_tensor >= 0.5] = 1.0

        _, H, W = mask_tensor.shape

        count = 0
        input_tensors = torch.zeros((self.batch_size, 3, H, W))
        input_nonorm_tensors = torch.zeros((self.batch_size, 3, H, W))

        input_paths = []

        while count < self.batch_size:
            random_img_path = select_random_img_path(self.root, category_dir, img_num)
            random_img = Image.open(random_img_path).convert('RGB')

            input_paths.append(random_img_path)

            input_tensor = self.norm_transform(random_img)
            input_nonorm = self.transform(random_img)

            input_tensor = input_tensor * mask_tensor
            input_nonorm = input_nonorm * mask_tensor

            input_tensors[count] = input_tensor
            input_nonorm_tensors[count] = input_nonorm

            count += 1

        return {'mask_tensor': mask_tensor,
                'input_tensors': input_tensors,
                'input_nonorm_tensors': input_nonorm_tensors,
                'input_paths': input_paths,
                'category': category_dir,
                'prior_tensor':prior_tensor,
                'prior_nonorm':prior_nonorm,
                }

    def __len__(self):
        return self.length



class TrainDataset(Dataset):
    def __init__(self, root, norm_transforms_=None, transforms_=None):
        self.norm_transform = transforms.Compose(norm_transforms_)
        self.transform = transforms.Compose(transforms_)

        assert len(os.listdir(os.path.join(root, "imgs"))) == len(
            os.listdir(os.path.join(root, "masks"))
        ) and len(os.listdir(os.path.join(root, "masks"))) == len(
            os.listdir(os.path.join(root, "normals"))
        )
        self.files_len = len(os.listdir(os.path.join(root, "masks")))

        self.img_files = [
            os.path.join(root, "imgs", "{}.png".format(i))
            for i in range(self.files_len)
        ]
        self.normal_files = [
            os.path.join(root, "normals", "{}.png".format(i))
            for i in range(self.files_len)
        ]
        self.mask_files = [
            os.path.join(root, "masks", "{}.png".format(i))
            for i in range(self.files_len)
        ]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        img = img.convert("RGB")
        normal = Image.open(self.normal_files[index])
        normal = normal.convert("RGB")
        mask = Image.open(self.mask_files[index])
        mask = mask.convert("RGB")

        img_item = self.norm_transform(img)
        normal_item = self.transform(normal)
        mask_item = self.transform(mask)

        return {"img": img_item, "normal": normal_item, "mask": mask_item}

    def __len__(self):
        return self.files_len


class PairDataset(Dataset):
    def __init__(self, norm_transforms_=None, transforms_=None):
        self.norm_transform = transforms.Compose(norm_transforms_)
        self.transform = transforms.Compose(transforms_)

        self.img_files = glob.glob("C:/dataset/captured/imgs/*/*.png")

    def __getitem__(self, index):
        img_path = self.img_files[index]

        dir_name = img_path.split("\\")[-2]
        mask_path = "C:/dataset/captured/masks/" + dir_name + ".png"
        label_path = "C:/dataset/captured/labels/" + dir_name + ".txt"

        img = Image.open(img_path)
        img = img.convert("RGB")

        with open(label_path) as l:
            num = int(l.read())
        inx = random.randint(0, num - 1)
        match_path = "C:/dataset/captured/imgs/" + dir_name + "/{}.png".format(inx)
        match = Image.open(match_path)
        match = match.convert("RGB")

        mask = Image.open(mask_path)
        mask = mask.convert("RGB")

        img = self.norm_transform(img)
        match = self.norm_transform(match)
        mask = self.transform(mask)

        return {"real_a": img, "real_b": match, "mask": mask}

    def __len__(self):
        return len(self.img_files)


class EvaluationDataset(Dataset):
    def __init__(self, root, norm_transforms_=None, transforms_=None):
        self.norm_transform = transforms.Compose(norm_transforms_)
        self.transform = transforms.Compose(transforms_)

        assert len(os.listdir(os.path.join(root, "imgs"))) == len(
            os.listdir(os.path.join(root, "masks"))
        ) and len(os.listdir(os.path.join(root, "masks"))) == len(
            os.listdir(os.path.join(root, "normals"))
        )
        self.files_len = len(os.listdir(os.path.join(root, "masks")))

        self.img_files = [
            os.path.join(root, "imgs", "{}.png".format(i))
            for i in range(self.files_len)
        ]
        self.normal_files = [
            os.path.join(root, "normals", "{}.png".format(i))
            for i in range(self.files_len)
        ]
        self.mask_files = [
            os.path.join(root, "masks", "{}.png".format(i))
            for i in range(self.files_len)
        ]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        img = img.convert("RGB")
        normal = Image.open(self.normal_files[index])
        normal = normal.convert("RGB")
        mask = Image.open(self.mask_files[index])
        mask = mask.convert("RGB")

        img_item = self.norm_transform(img)
        normal_item = self.transform(normal)
        mask_item = self.transform(mask)

        return {"img": img_item, "normal": normal_item, "mask": mask_item}

    def __len__(self):
        return self.files_len


class TestDataset(Dataset):
    def __init__(self, root, norm_transforms_=None, transforms_=None):
        self.norm_transform = transforms.Compose(norm_transforms_)
        self.transform = transforms.Compose(transforms_)

        assert len(os.listdir(os.path.join(root, "imgs"))) == len(
            os.listdir(os.path.join(root, "masks"))
        )
        self.files_len = len(os.listdir(os.path.join(root, "masks")))

        self.img_files = [
            os.path.join(root, "imgs", "{}.png".format(i))
            for i in range(self.files_len)
        ]
        self.mask_files = [
            os.path.join(root, "masks", "{}.png".format(i))
            for i in range(self.files_len)
        ]

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        img = img.convert("RGB")
        mask = Image.open(self.mask_files[index])
        mask = mask.convert("RGB")

        img_item = self.norm_transform(img)
        mask_item = self.transform(mask)

        return {"img": img_item, "mask": mask_item}

    def __len__(self):
        return self.files_len


class CapturedDataset(Dataset):
    def __init__(self, root, norm_transforms_=None, transforms_=None):
        self.norm_transform = transforms.Compose(norm_transforms_)
        self.transform = transforms.Compose(transforms_)

        self.img_files = glob.glob("C:/dataset/captured/imgs/*/*.png")

    def __getitem__(self, index):
        img_path = self.img_files[index]

        dir_name = img_path.split("\\")[-2]
        mask_path = "C:/dataset/captured/masks/" + dir_name + ".png"

        img = Image.open(img_path)
        img = img.convert("RGB")

        mask = Image.open(mask_path)
        mask = mask.convert("RGB")

        img = self.norm_transform(img)
        mask = self.transform(mask)

        return {"img": img, "mask": mask}

    def __len__(self):
        return len(self.img_files)


class PSM(Dataset):
    def __init__(self, root, norm_transforms_=None, transforms_=None):
        self.norm_transform = transforms.Compose(norm_transforms_)
        self.transform = transforms.Compose(transforms_)

        self.img_files = glob.glob("F:/psm/imgs/cat/*.png")

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img = Image.open(img_path)
        img = img.convert("RGB")

        mask_path = "F:/psm/masks/cat.mask.png"
        mask = Image.open(mask_path)
        mask = mask.convert("RGB")

        name = img_path.split("\\")[-1]

        img = self.norm_transform(img)
        mask = self.transform(mask)

        return {"img": img, "mask": mask, "name": name}

    def __len__(self):
        return len(self.img_files)

