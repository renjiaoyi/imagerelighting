import torch
import numpy as np
import torch.nn as nn
mean_image = nn.BatchNorm2d(3, affine=False).cuda()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def ssq_error_tensor(correct, estimate, mask):
    v = torch.sum(estimate ** 2 * mask, dim=2)
    alpha = torch.sum(correct * estimate * mask, dim=2)
    alpha[v>1e-5] = alpha[v>1e-5]/v[v>1e-5]
    alpha[v<=1e-5] = 0
    alpha = alpha.unsqueeze(2)
    alpha = alpha.expand_as(estimate)
    ssq = torch.sum(mask * (correct - alpha * estimate) ** 2, dim=2)
    total = torch.sum(mask * correct ** 2, dim=2)
    ssq = torch.sum(ssq, dim=1)
    total = torch.sum(total, dim=1)
    error = ssq/total
    return error

def ssq_error(correct, estimate, mask):
    #assert correct.ndim == 2
    L,M,N = correct.shape
    error = torch.zeros(L)
    error=error.cuda()
    for i in range(0,L):
        v = torch.sum(estimate[i,:,:] ** 2 * mask[i,:,:])
        if v > 1e-5:
            alpha = torch.sum(correct[i,:,:] * estimate[i,:,:] * mask[i,:,:]) / v
        else:
            alpha = 0.0
        error[i] = torch.sum(mask[i,:,:] * (correct[i,:,:] - alpha * estimate[i,:,:]) ** 2)
    return error

def local_error(correct, estimate, mask):
    # fixed as default value in MIT-Intrinsics
    window_size = 20
    window_shift = window_size // 2

    # L, M, N = correct.shape
    # ssq = total = torch.zeros(L)
    # zero = torch.zeros(L)
    # total=total.cuda()
    # zero=zero.cuda()
    # ssq=ssq.cuda()
    # local = torch.zeros(L)

    unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=window_shift)
    tmp1 = torch.unsqueeze(correct, 1)
    output1 = unfold(tmp1)
    output1 = output1.permute(0, 2, 1)

    tmp2 = torch.unsqueeze(estimate, 1)
    output2 = unfold(tmp2)
    output2 = output2.permute(0, 2, 1)

    tmp3 = torch.unsqueeze(mask, 1)
    output3 = unfold(tmp3)
    output3 = output3.permute(0, 2, 1)

    ssqs = ssq_error_tensor(output1, output2, output3)
    return ssqs

def to_normal(pq, m):
    norm = torch.sqrt(torch.sum(pq ** 2, dim=1, keepdim=True) + 1.0)
    xy = pq / norm
    z = 1.0 / norm

    return torch.cat([xy, z], dim=1) * m


def angular_loss(p, g, m):
    nm_prod = torch.sum(p * g, dim=1, keepdim=True)
    nm_prod = torch.clamp(nm_prod, -0.9999, 0.9999)
    nm_angle = torch.acos(nm_prod) * m

    return torch.mean(nm_angle ** 2)


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

class LowRankLoss(nn.Module):
    def __init__(self):
        super(LowRankLoss, self).__init__()
        return
    def forward(self, M, mask):
        loss = 0
        batch_size, C, H, W = M.shape
        avg = mean_image(M)
        #M = M - avg
        M = M * mask
        M = M.reshape((batch_size, C, H * W))
        M = M.reshape((batch_size, C * H * W))
        M1 = torch.transpose(M, 0, 1)#.detach()
        u, s, vh = torch.linalg.svd(M1, full_matrices=False)
        s[1:] = 0
        loss = torch.dist(torch.transpose(M, 0, 1), u @ torch.diag(s) @ vh)/torch.sum(mask)
        return loss*1e2

class RankLossFunc(nn.Module):
    def __init__(self):
        super(RankLossFunc, self).__init__()
        return
    def forward(self, M, mask):
        loss = 0
        batch_size, C, H, W = M.shape
        avg = mean_image(M)
        #M = M - avg
        M = M * mask
        M = M.reshape((batch_size, C, H * W))
        M = M.reshape((batch_size, C * H * W))
        M1 = torch.transpose(M, 0, 1).detach()
        u, s, vh = torch.linalg.svd(M1, full_matrices=False)
        s[1:] = 0
        loss = torch.dist(torch.transpose(M, 0, 1), u @ torch.diag(s) @ vh)/torch.sum(mask)
        return loss*1e2

class RankLossFunc_Spec(nn.Module):
    def __init__(self):
        super(RankLossFunc_Spec, self).__init__()
        return
    def forward(self, I, S, mask):
        loss = 0
        M0 = I-S
        M0 = M0*mask
        M0 = torch.clip(M0, 0.00001, 1.0)
        M = M0[:,:2,:,:]
        Msum = torch.sum(M0,1,keepdim=True)#.detach()
        Msum = torch.tile(Msum, (1, 2,1,1))
        M = M/Msum
        batch_size, C, H, W = M.shape
        M = M * mask
        
        M = M.reshape((batch_size, C, H * W))
        M = M.reshape((batch_size, C * H * W))
        M1 = torch.transpose(M, 0, 1).detach()
        u, s, vh = torch.linalg.svd(M1, full_matrices=False)
        s[1:] = 0
        loss = torch.dist(torch.transpose(M, 0, 1), u @ torch.diag(s) @ vh)/torch.sum(mask)
        return loss*1e2

class LMSELossFunc(nn.Module):
    def __init__(self):
        super(LMSELossFunc, self).__init__()
        return
    def forward(self, correct, estimate, mask):
        L = correct.shape[0]
        error = torch.zeros(L).cuda()
        # get error for each channel
        #print(correct.shape)
        for c in range(0, 3):
            local = local_error(correct[:, c, :, :], estimate[:, c, :, :], mask[:, c, :, :])
            error = error + local
        # average for each channel
        return torch.mean(error) / 3.0
class SMSELossFunc(nn.Module):
    def __init__(self):
        super(SMSELossFunc, self).__init__()
        return
    def forward(self, correct, estimate, mask):
        L = correct.shape[0]
        error = torch.zeros(L)
        # get error for each channel
        #print(correct.shape)
        ssq = total = torch.zeros(L)
        ssq=ssq.cuda()
        local = torch.zeros(L)
        for i in range(0,3):
            correct_curr = torch.squeeze(correct[ :, i, :,:])
            estimate_curr = torch.squeeze(estimate[ :, i, :,:])
            mask_curr = torch.squeeze(mask[ :, i, :,:])
            #for l = range(0,L):
            masksum = torch.sum(mask_curr,dim=1)
            masksum2 = torch.sum(masksum,dim=1)
            ssq += ssq_error(correct_curr, estimate_curr, mask_curr)/masksum2
        # average for each channel
        return torch.mean(ssq) / 3.0

class AMSELossFunc(nn.Module):
    def __init__(self):
        super(AMSELossFunc, self).__init__()
        return
    def forward(self, correct, estimate, mask):
        L = correct.shape[0]
        ssq = total = torch.zeros(L)
        tmp = torch.sum(mask,dim=1)
        tmp1 = torch.sum(tmp,dim=1)
        sum0 = torch.sum(tmp1,dim=1)
        for l in range(0,L):
            if sum0[l]>0:
                ssq[l]=torch.sum(mask[l,:,:,:]*(correct[l,:,:,:]-estimate[l,:,:,:])**2)/sum0[l]
        return torch.mean(ssq) 
class NormPirorLossFunc(nn.Module):
    def __init__(self):
        super(NormPirorLossFunc, self).__init__()
        return
    def forward(self, norm, norm_piror, mask):
        norm = (norm + 1.0) / 2.0
        norm = norm[:, [2, 1, 0], :, :]
        num = torch.sum(mask)
        loss = torch.dist(norm*mask, norm_piror*mask)/num
        return loss*10
def render_layer_color(nm, L_SHcoeffs, color, gamma=2.2):

    # M is only related with lighting
    c1 = torch.tensor(0.429043)
    c2 = torch.tensor(0.511664)
    c3 = torch.tensor(0.743125)
    c4 = torch.tensor(0.886227)
    c5 = torch.tensor(0.247708)

    # each row have shape (batch, 4, 3)
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

    # M is a 5d tensot with shape (batch,4,4,3[rgb]), the dim 1 and 2 are transposely equivalent
    M = torch.stack([M_row1, M_row2, M_row3, M_row4], dim=1)

    # find batch-spatial three dimensional mask of defined normals over nm
    # mask = torch.equal(torch.sum(nm,dim=-1),0)

    # extend Cartesian to homogeneous coords and extend its last for rgb individual multiplication dimension, nm_homo have shape (total_npix, 4)
    total_npix = nm.size()[:3]
    ones = torch.ones(total_npix)
    ones = ones.to(torch.device("cuda"))
    nm_homo = torch.cat([nm, torch.unsqueeze(ones, dim=-1)], dim=-1)

    # contruct batch-wise flatten M corresponding with nm_homo, such that multiplication between them is batch-wise
    M = torch.unsqueeze(torch.unsqueeze(M, dim=1), dim=1)

    # expand M for broadcasting, such that M has shape (npix,4,4,3)
    # expand nm_homo, such that nm_homo has shape (npix,4,1,1)
    nm_homo = torch.unsqueeze(torch.unsqueeze(nm_homo, dim=-1), dim=-1)
    # tmp have shape (npix, 4, 3[rgb])
    tmp = torch.sum(nm_homo * M, dim=-3)
    # E has shape (npix, 3[rbg])
    E = torch.sum(tmp * nm_homo[:, :, :, :, 0, :], dim=-2)
    color = torch.unsqueeze(torch.unsqueeze(color,dim=-2),dim=-2)
    #E_color = E#*color
    # compute intensity by product between irradiance and albedo
    i = E*color
    i = i.permute(0, 3, 1, 2)

    i = torch.clamp(i, min = 0.0) + 0.1

    i_max = torch.max(i[:, 0, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]
    i_min = torch.min(i[:, 0, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]

    i_max = i_max.unsqueeze(1).unsqueeze(1)
    i_max = i_max.expand(-1, 3, 256, 256)

    i_min = i_min.unsqueeze(1).unsqueeze(1)
    i_min = i_min.expand(-1, 3, 256, 256)

    i_norm = (i - i_min)/(i_max - i_min) +1e-4

    return i, i_norm
def render_layer(nm, L_SHcoeffs,gamma=2.2):

    # M is only related with lighting
    c1 = torch.tensor(0.429043)
    c2 = torch.tensor(0.511664)
    c3 = torch.tensor(0.743125)
    c4 = torch.tensor(0.886227)
    c5 = torch.tensor(0.247708)

    # each row have shape (batch, 4, 3)
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
            c4 * torch.abs(L_SHcoeffs[:, 0, :]) - c5 * L_SHcoeffs[:, 6, :],
        ],
        dim=1,
    )

    # M is a 5d tensot with shape (batch,4,4,3[rgb]), the dim 1 and 2 are transposely equivalent
    M = torch.stack([M_row1, M_row2, M_row3, M_row4], dim=1)

    # find batch-spatial three dimensional mask of defined normals over nm
    # mask = torch.equal(torch.sum(nm,dim=-1),0)

    # extend Cartesian to homogeneous coords and extend its last for rgb individual multiplication dimension, nm_homo have shape (total_npix, 4)
    total_npix = nm.size()[:3]
    ones = torch.ones(total_npix)
    ones = ones.to(torch.device("cuda"))
    nm_homo = torch.cat([nm, torch.unsqueeze(ones, dim=-1)], dim=-1)

    # contruct batch-wise flatten M corresponding with nm_homo, such that multiplication between them is batch-wise
    M = torch.unsqueeze(torch.unsqueeze(M, dim=1), dim=1)

    # expand M for broadcasting, such that M has shape (npix,4,4,3)
    # expand nm_homo, such that nm_homo has shape (npix,4,1,1)
    nm_homo = torch.unsqueeze(torch.unsqueeze(nm_homo, dim=-1), dim=-1)
    # tmp have shape (npix, 4, 3[rgb])
    tmp = torch.sum(nm_homo * M, dim=-3)
    # E has shape (npix, 3[rbg])
    E = torch.sum(tmp * nm_homo[:, :, :, :, 0, :], dim=-2)

    # compute intensity by product between irradiance and albedo
    i = E
    i = i.permute(0, 3, 1, 2)

    i = torch.clamp(i, min = 0.0) + 0.1

    i_max = torch.max(i[:, 0, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]
    i_min = torch.min(i[:, 0, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]

    i_max = i_max.unsqueeze(1).unsqueeze(1)
    i_max = i_max.expand(-1, 3, 256, 256)

    i_min = i_min.unsqueeze(1).unsqueeze(1)
    i_min = i_min.expand(-1, 3, 256, 256)

    i_norm = (i - i_min)/(i_max - i_min) +1e-4
    

    return i, i_norm

def render_layer_RGB(nm, L_SHcoeffs, gamma=2.2):

    # M is only related with lighting
    c1 = torch.tensor(0.429043)
    c2 = torch.tensor(0.511664)
    c3 = torch.tensor(0.743125)
    c4 = torch.tensor(0.886227)
    c5 = torch.tensor(0.247708)

    # each row have shape (batch, 4, 3)
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

    # M is a 5d tensot with shape (batch,4,4,3[rgb]), the dim 1 and 2 are transposely equivalent
    M = torch.stack([M_row1, M_row2, M_row3, M_row4], dim=1)

    # find batch-spatial three dimensional mask of defined normals over nm
    # mask = torch.equal(torch.sum(nm,dim=-1),0)

    # extend Cartesian to homogeneous coords and extend its last for rgb individual multiplication dimension, nm_homo have shape (total_npix, 4)
    total_npix = nm.size()[:3]
    ones = torch.ones(total_npix)
    ones = ones.to(torch.device("cuda"))
    nm_homo = torch.cat([nm, torch.unsqueeze(ones, dim=-1)], dim=-1)

    # contruct batch-wise flatten M corresponding with nm_homo, such that multiplication between them is batch-wise
    M = torch.unsqueeze(torch.unsqueeze(M, dim=1), dim=1)

    # expand M for broadcasting, such that M has shape (npix,4,4,3)
    # expand nm_homo, such that nm_homo has shape (npix,4,1,1)
    nm_homo = torch.unsqueeze(torch.unsqueeze(nm_homo, dim=-1), dim=-1)
    # tmp have shape (npix, 4, 3[rgb])
    tmp = torch.sum(nm_homo * M, dim=-3)
    # E has shape (npix, 3[rbg])
    E = torch.sum(tmp * nm_homo[:, :, :, :, 0, :], dim=-2)

    # compute intensity by product between irradiance and albedo
    i = E
    i = i.permute(0, 3, 1, 2)

    i_R_max = torch.max(i[:, 0, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]
    i_R_min = torch.min(i[:, 0, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]
    i_G_max = torch.max(i[:, 1, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]
    i_G_min = torch.min(i[:, 1, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]
    i_B_max = torch.max(i[:, 2, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]
    i_B_min = torch.min(i[:, 2, :, :].view(nm.size()[0], -1), dim=1, keepdim= True)[0]

    i_R_max = i_R_max.unsqueeze(1).unsqueeze(1)
    i_R_max = i_R_max.expand(-1, 1, 256, 256)
    i_G_max = i_G_max.unsqueeze(1).unsqueeze(1)
    i_G_max = i_G_max.expand(-1, 1, 256, 256)
    i_B_max = i_B_max.unsqueeze(1).unsqueeze(1)
    i_B_max = i_B_max.expand(-1, 1, 256, 256)

    i_R_min = i_R_min.unsqueeze(1).unsqueeze(1)
    i_R_min = i_R_min.expand(-1, 1, 256, 256)
    i_G_min = i_G_min.unsqueeze(1).unsqueeze(1)
    i_G_min = i_G_min.expand(-1, 1, 256, 256)
    i_B_min = i_B_min.unsqueeze(1).unsqueeze(1)
    i_B_min = i_B_min.expand(-1, 1, 256, 256)

    i_max = torch.cat([i_R_max, i_G_max, i_B_max], dim=1)
    i_min = torch.cat([i_R_min, i_G_min, i_B_min], dim=1)

    i_norm = (i - i_min)/(i_max - i_min) + 1e-4
    i = (i - i_min)/(i_max - i_min) + 1e-4

    return i, i_norm