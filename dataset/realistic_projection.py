from torch_scatter import scatter
import torch.nn as nn
import numpy as np
import torch
import random
import json
from math import pi as pi
import torch.nn.functional as F
import matplotlib.pyplot as plt
TRANS = -1.5

# realistic projection parameters

params = {'maxpoolz': 1, 'maxpoolxy': 5, 'maxpoolpadz': 0, 'maxpoolpadxy': 2,
          'convz': 1, 'convxy': 3, 'convsigmaxy': 3, 'convsigmaz': 1, 'convpadz': 0, 'convpadxy': 1,
          'imgbias': 0., 'depth_bias': 0.2, 'obj_ratio': 1, 'bg_clr': 0.0, 'bg_label': -1,
          'resolution': 256, 'depth': 8}




class Grid2Image(nn.Module):
    """A pytorch implementation to turn 3D grid to 2D image. 
       Maxpool: densifying the grid
       Convolution: smoothing via Gaussian
       Maximize: squeezing the depth channel
    """

    def __init__(self):
        super().__init__()
        torch.backends.cudnn.benchmark = False

        self.maxpool = nn.MaxPool3d((params['maxpoolz'], params['maxpoolxy'], params['maxpoolxy']),
                                    stride=1, padding=(params['maxpoolpadz'], params['maxpoolpadxy'],
                                                       params['maxpoolpadxy']), return_indices=True)
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(params['convz'], params['convxy'], params['convxy']),
                                    stride=1, padding=(params['convpadz'], params['convpadxy'], params['convpadxy']),
                                    bias=True)
        kn3d = get3DGaussianKernel(params['convxy'], params['convz'], sigma=params['convsigmaxy'],
                                   zsigma=params['convsigmaz'])
        self.conv.weight.data = torch.Tensor(kn3d).repeat(1, 1, 1, 1, 1)
        self.conv.bias.data.fill_(0)

        color_np = plt.cm.viridis(np.linspace(0, 1, 8))[:, :3]
        self.color_torch = torch.tensor(color_np)




    def forward(self, grid, num_classes):
        x = grid[:, :, :, :, 0].squeeze(-1).unsqueeze(1) 
        x, indices = self.maxpool(x)
        self.conv.to(x.device)
        x = self.conv(x)

        img, idx = torch.max(x, dim=2)
        img = img / torch.max(torch.max(img, dim=-1)[0], dim=-1)[0][:, :, None, None]
        img = 1 - img

        
        img = img.repeat(1, 3, 1, 1)
        img = (img.permute(0, 2, 3, 1) * 255)

        B, _, W, H = idx.shape
        idx = idx.squeeze(1) # [B, W, H]

        x_label = grid[:, :, :, :, 1].unsqueeze(-1)
        img_label = torch.gather(x_label.permute(0, 2, 3, 1, 4), 3, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 1)).squeeze(3)
        img_label[img_label == -1] = 255

        x_logits = grid[:, :, :, :, 2:2+num_classes]
        img_logits = torch.gather(x_logits.permute(0, 2, 3, 1, 4), 3, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, num_classes)).squeeze(3)
        img_logits[img_logits == -100] = 0

        return img.detach(), img_label.detach(), img_logits.detach()


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """
    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    return rot_mat


def points2grid(points, points_label, points_logits, resolution=params['resolution'], depth=params['depth'], trgt_data=None, num_classes=None):
    """Quantize each point cloud to a 3D grid.
    Args:
        points (torch.tensor): of size [B, _, 3]
    Returns:
        grid (torch.tensor): of size [B * self.num_views, depth, resolution, resolution]
    """

    batch, pnum, _ = points.shape
    pmax, pmin = points.max(dim=1)[0], points.min(dim=1)[0]
    pcent = (pmax + pmin) / 2
    pcent = pcent[:, None, :]
    prange = (pmax - pmin).max(dim=-1)[0][:, None, None]
    points = (points - pcent) / prange * 2.
    points[:, :, :2] = points[:, :, :2] * params['obj_ratio']

    depth_bias = 1 / (depth - 3)
    _x = (points[:, :, 0] + 1) / 2 * resolution
    _y = (points[:, :, 1] + 1) / 2 * resolution
    _z = ((points[:, :, 2] + 1) / 2 + depth_bias) / (1 + depth_bias) * (depth - 2)

    _x.ceil_()
    _y.ceil_()
    z_int = _z.ceil()

    _x = torch.clip(_x, 1, resolution - 2)
    _y = torch.clip(_y, 1, resolution - 2)
    _z = torch.clip(_z, 1, depth - 2)

    value = torch.cat([_z.unsqueeze(-1), points_label.unsqueeze(-1), points_logits], dim = 2) # batch, pnum, 1+1+11+11
    
    coordinates = z_int * resolution * resolution + _y * resolution + _x

    grid_points = torch.ones([batch, depth, resolution, resolution, 1], device=points.device).view(batch, -1, 1) * params['bg_clr']
    grid_label = torch.ones([batch, depth, resolution, resolution, 1], device=points.device).view(batch, -1, 1) * params['bg_label']
    grid_logits = torch.ones([batch, depth, resolution, resolution, num_classes], device=points.device).view(batch, -1, num_classes) * (-100)

    grid = torch.cat([grid_points, grid_label, grid_logits], dim = -1)
    grid = scatter(value, coordinates.long(), dim=1, out=grid, reduce="max")
    grid = grid.reshape((batch, depth, resolution, resolution, num_classes+2)).permute((0, 1, 3, 2, 4))

    return grid, _x, _y


class Realistic_Projection:
    """For creating images from PC based on the view information.
    """

    def __init__(self, trgt_data, num_classes):

        self.trgt_data = trgt_data
        self.num_classes = num_classes
        if trgt_data == 's3dis':
            y_axis = -np.pi * 0.2
            x_axis = -np.pi

            _views = np.asarray([
                [[x_axis, y_axis, 1 * np.pi / 4], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 1 * np.pi / 4], [-0.5, -0.5, TRANS]],
                [[x_axis, y_axis, 3 * np.pi / 4], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 3 * np.pi / 4], [-0.5, -0.5, TRANS]],
                [[x_axis, y_axis, 5 * np.pi / 4], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 5 * np.pi / 4], [-0.5, -0.5, TRANS]],
                [[x_axis, y_axis, 7 * np.pi / 4], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 7 * np.pi / 4], [-0.5, -0.5, TRANS]],
            ])

        else:
            y_axis = -np.pi * 0.25
            _views = np.asarray([
                [[0, y_axis, 1 * np.pi / 9], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 3 * np.pi / 9], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 7 * np.pi / 9], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 10 * np.pi / 9], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 11 * np.pi / 9], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 13 * np.pi / 9], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 15 * np.pi / 9], [-0.5, -0.5, TRANS]],
                [[0, y_axis, 17 * np.pi / 9], [-0.5, -0.5, TRANS]],
            ])


        self.num_views = _views.shape[0]
        self.rot_mat = torch.tensor(_views[:, 0, :]).float()
        self.grid2image = Grid2Image()

    def get_img(self, points, points_label, points_logits):
        b, _, _ = points.shape
        device = points.device
        
        keep_points, keep_label, keep_logits, keep_idx = self.point_transform(
            raw_points=torch.repeat_interleave(points, self.num_views, dim=0),
            points_label=torch.repeat_interleave(points_label, self.num_views, dim=0),
            points_logits=torch.repeat_interleave(points_logits, self.num_views, dim=0),
            rot_mat=self.rot_mat.repeat(b, 1, 1).to(device),
            num_classes=self.num_classes,
            )
        
        grid, x, y = points2grid(points=keep_points, points_label=keep_label, points_logits=keep_logits, 
                                    resolution=params['resolution'], depth=params['depth'], trgt_data=self.trgt_data, num_classes=self.num_classes)
        
        grid = grid.squeeze()
        img, img_label, img_logits = self.grid2image(grid, self.num_classes)

        return img, img_label, img_logits, keep_idx, x, y

    @staticmethod
    def point_transform(raw_points, points_label, points_logits, rot_mat, num_classes):
        """
        :param points: [batch, num_points, 3]
        :param rot_mat: [batch, 3]
        :return:
        """
        B, N, C = raw_points.shape
        rot_mat = euler2mat(rot_mat.squeeze(0)).transpose(1, 2)

        points = torch.matmul(raw_points, rot_mat)

        xyz = 2
        new_points = torch.zeros([B, N, C], dtype = torch.float).to(raw_points.device)
        new_points_label = torch.zeros([B, N], dtype = torch.float).to(raw_points.device)
        new_points_logits = torch.zeros([B, N, num_classes], dtype = torch.float).to(raw_points.device)
        keep_idx = torch.zeros([B, N]).to(raw_points.device)
        
        for i in range(B):
            points_i = points[i]
            points_label_i = points_label[i]
            points_logits_i = points_logits[i]

            xyz_max = torch.amax(points_i, axis=0)[0:3]
            xyz_min = torch.amin(points_i, axis=0)[0:3]
            xyz_midpoint = (xyz_max[xyz] + xyz_min[xyz]) / 2
            choose_idx = torch.where(points[i, :, xyz] <= xyz_midpoint)[0]#.float()
            choose_idx_noise_1 = choose_idx[torch.randint(0, len(choose_idx), (N - len(choose_idx),))]

            idx = torch.cat([choose_idx, choose_idx_noise_1])
            keep_idx[i] = idx

            new_points[i] = points_i[idx]
            new_points_label[i] = points_label_i[idx]
            new_points_logits[i] = points_logits_i[idx]

        return new_points, new_points_label, new_points_logits, keep_idx


def get2DGaussianKernel(ksize, sigma=0):
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()
    return kernel


def get3DGaussianKernel(ksize, depth, sigma=2, zsigma=2):
    kernel2d = get2DGaussianKernel(ksize, sigma)
    zs = (np.arange(depth, dtype=np.float32) - depth // 2)
    zkernel = np.exp(-(zs ** 2) / (2 * zsigma ** 2))
    kernel3d = np.repeat(kernel2d[None, :, :], depth, axis=0) * zkernel[:, None, None]
    kernel3d = kernel3d / torch.sum(kernel3d)
    return kernel3d
