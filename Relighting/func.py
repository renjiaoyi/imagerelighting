import os
import cv2
import numpy as np
from skimage.morphology import disk, erosion
import glob


def ensureSingleChannel(x):
    ret = x
    if len(x.shape) > 2:
        ret = ret[..., 0]

    return np.expand_dims(ret, -1)


def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    mask = ensureSingleChannel(mask)
    
    mask[mask < 128] = 0
    mask = erosion(
        mask[..., 0], disk(2)
    )  # Apply a erosion (channels need to be removed)

    return np.expand_dims(mask, -1).repeat(3, axis=-1)  # And added back

def make_video(img_path, hdrname, save_path, idx, framenum):
    fps = 10
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(save_path, fourcc, fps, (1024,1024),True)

    a = img_path+'/{}_'.format(idx)+hdrname+'_*.png'
    imgs = glob.glob(img_path+'/{}_'.format(idx)+hdrname+'_*.png')
    for i in range(0,framenum):
        frame = cv2.imread(os.path.join(img_path, '{}_'.format(idx)+hdrname+'_{}.png'.format(i)))
        print(i)
        frame = frame[:,512:1536,:]
        output.write(frame)
    
    output.release()
    print("finish")
def make_video_wide(img_path, hdrname, save_path, idx, framenum):
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(save_path, fourcc, fps, (2048,1024),True)

    a = img_path+'/{}_'.format(idx)+hdrname+'_*.png'
    imgs = glob.glob(img_path+'/{}_'.format(idx)+hdrname+'_*.png')
    for i in range(0,framenum):
        inew = i
        if inew>framenum-1:
            inew = inew-framenum+1
        frame = cv2.imread(os.path.join(img_path, '{}_'.format(idx)+hdrname+'_{}.png'.format(inew)))
        print(i)
        output.write(frame)
    
    output.release()
    print("finish")

def apply_bg(mask_path, hdrname, root, idx, samples,cutmap):
    mask = read_mask(mask_path)
    mask = cv2.resize(mask, (1024, 1024))
    mask_bool = (mask != 0)
    mask = np.array(mask_bool, dtype=int)

    for i in range(samples):
        bg = cv2.imread('{}'.format(cutmap)+hdrname+'/'+hdrname+'_{}.png'.format(i))
        bg = cv2.resize(bg, (2048,1024))
        img = cv2.imread('{}/{}_'.format(root,idx)+hdrname+'_2k.hdr_{}.png'.format(i+1))
        img = cv2.resize(img, (1024, 1024))
        img = img * mask

        inx_u = bg.shape[0] - 1024
        inx_d = bg.shape[0]
        inx_l = int(bg.shape[1] / 2 - img.shape[0] / 2) 
        inx_r = int(bg.shape[1] / 2 + img.shape[0] / 2) 

        img_ = np.zeros(bg.shape)
        img_[inx_u: inx_d, inx_l: inx_r, :] = img
        mask_ = np.zeros(bg.shape)
        mask_[inx_u: inx_d, inx_l: inx_r, :] = mask

        bg_bool = (mask_ == 0)
        bg_mask = np.array(bg_bool, dtype=int)

        bg = bg * bg_mask
        img_bg = img_ + bg

        if not os.path.exists('{}/combined'.format(root)):
            os.mkdir('{}/combined'.format(root))
        cv2.imwrite('{}/combined/{}_'.format(root, idx)+hdrname+'_{}.png'.format(i), img_bg)

def apply_bg_naive(mask_path, hdrname, root, idx, samples,cutmap):
    mask = read_mask(mask_path)
    mask = cv2.resize(mask, (1024, 1024))
    
    mask_bool = (mask != 0)
    #print(mask.max)
    mask = np.array(mask_bool, dtype=int)
    
    print(mask.max())
    img = cv2.imread('{}/'.format(root)+'0.png')
    img = cv2.resize(img, (1024, 1024))
    img = img * mask
    
    for i in range(samples):
        bg = cv2.imread('{}/'.format(cutmap)+hdrname+'/'+hdrname+'_{}.png'.format(i))
        bg = cv2.resize(bg, (2048,1024))
        
        inx_u = bg.shape[0] - 1024
        inx_d = bg.shape[0]
        inx_l = int(bg.shape[1] / 2 - img.shape[0] / 2) 
        inx_r = int(bg.shape[1] / 2 + img.shape[0] / 2) 
        img_ = np.zeros(bg.shape)
        img_[inx_u: inx_d, inx_l: inx_r, :] = img
        mask_ = np.zeros(bg.shape)
        mask_[inx_u: inx_d, inx_l: inx_r, :] = mask

        bg_bool = (mask_ == 0)
        bg_mask = np.array(bg_bool, dtype=int)

        bg = bg * bg_mask
        img_bg = img_ + bg
        if not os.path.exists('./{}/combined'.format(root)):
            os.mkdir('./{}/combined'.format(root))
        cv2.imwrite('./{}/combined/{}_'.format(root, idx)+hdrname+'_{}.png'.format(i), img_bg)

