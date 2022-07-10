import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Conv1x1, diffConv

class Model(nn.Module):
    def __init__(self, args, output_channels=40):
        super().__init__()

        self.args = args

        init_feat = 32

        self.le = Conv1x1(3, init_feat) # local encoder

        radius = args.radius

        self.conv1 = diffConv(init_feat, init_feat*2, radius)

        self.conv2 = diffConv(init_feat*2, init_feat*4, radius*2)

        self.conv3 = diffConv(init_feat*4, init_feat*8, radius*4)

        self.conv4 = diffConv(init_feat*8, init_feat*16, radius*8)   

        self.last_conv = Conv1x1(init_feat*16, args.emb_dims)

        self.linear = nn.Sequential(
            nn.Linear(args.emb_dims*2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),          
            nn.Dropout(p=args.dropout),                     
            nn.Linear(512, output_channels)
        )
    
    def forward(self, x):
        xyz = x.clone()
        point_num = xyz.size(1)

        feat = self.le(x)

        l1_feat, l1_xyz = self.conv1(feat, xyz, point_num)

        l2_feat, l2_xyz = self.conv2(l1_feat, l1_xyz, point_num//2)      

        l3_feat, l3_xyz = self.conv3(l2_feat, l2_xyz, point_num//4)
        
        l4_feat, _ = self.conv4(l3_feat, l3_xyz, point_num//8)

        x = self.last_conv(l4_feat)

        batch_size = x.size(0) 
        x = x.transpose(1, 2) 
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) 
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) 
        x = torch.cat((x1, x2), dim=1)

        x = self.linear(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Classification')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--radius', type=float, default=0.005,
                        help='search radius')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = torch.rand(5, 1024, 3).to(device)
    model = Model(args).to(device)
    feat = model(pc)
    print(feat.shape)
    
    
    
    