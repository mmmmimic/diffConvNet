import torch
import sys
sys.path.append('./models/pointnet2')
from .pointnet2 import pointnet2_utils

def get_dist(src, dst):
    """
    Calculate the Euclidean distance between each point pair in two point clouds.
    Inputs:
        src[B, M, 3]: point cloud 1
        dst[B, N, 3]: point cloud 2
    Return: 
        dist[B, M, N]: distance matrix
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points[B, N, C]: input point features
        idx[B, M]: sample index data
    Return:
        new_points[B, M, C]: quried point features
    """
    new_points = pointnet2_utils.gather_operation(points.transpose(1,2).contiguous(), idx).transpose(1,2).contiguous()
    return new_points

def sample_and_group(radius, k, xyz, feat, centroid, dist):
    """
    Input:
        radius[B, M, 1]: search radius of each key point
        k(int): max number of samples in local region
        xyz[B, N, 3]: query points
        centroid[B, M, 3]: key points
        dist[B, M, N]: distance matrix
        feat[B, N, D]: input points features
    Return:
        cent_feat[B, M, D]: grouped features
        idx[B, M, k]: indices of selected neighbors
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, M, _ = centroid.shape
    
    idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, M, 1])
    
    idx[dist > radius] = N
    idx = idx.sort(dim=-1)[0][:, :, :k]
    group_first = idx[:, :, 0].view(B, M, 1).repeat([1, 1, k])
    mask = (idx == N)
    idx[mask] = group_first[mask]
    
    torch.cuda.empty_cache()
    idx = idx.int().contiguous()

    feat = feat.transpose(1,2).contiguous()
    cent_feat = pointnet2_utils.grouping_operation(feat, idx)
    cent_feat = cent_feat.transpose(1,2).transpose(-1, -2).contiguous()
    torch.cuda.empty_cache()
    
    return cent_feat, idx







