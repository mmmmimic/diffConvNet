import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Conv1x1, diffConv, PointFeaturePropagation
from .pointnet2.pointnet2_modules import PointnetSAModule
import argparse

class Model(nn.Module):
    def __init__(self, args, num_classes=50, category=16):
        super().__init__()

        init_channel = 32
        self.le0 = Conv1x1(3, init_channel)

        self.le1 = PointnetSAModule(
                    npoint=args.num_points,
                    radius=0.05,
                    nsample=20,
                    mlp=[init_channel, init_channel, init_channel],
                    use_xyz=True,
                    bn=True
        )

        radius = args.radius

        # encoder
        self.conv1 = diffConv(init_channel, init_channel*2, radius)

        self.conv2 = diffConv(init_channel*2, init_channel*4, radius*4)

        self.conv3 = diffConv(init_channel*4, init_channel*8, radius*8)

        self.conv4 = diffConv(init_channel*8, init_channel*16, radius*16)

        self.conv5 = diffConv(init_channel*16, init_channel*32, radius*32)

        # decoder
        self.fp4 = PointFeaturePropagation(in_channel1=init_channel*32, in_channel2=init_channel*16, out_channel=init_channel*8)
        self.up_conv5 = diffConv(init_channel*8, init_channel*16, radius*16)

        self.fp3 = PointFeaturePropagation(in_channel1=init_channel*16, in_channel2=init_channel*8, out_channel=init_channel*4)
        self.up_conv4 = diffConv(init_channel*4, init_channel*8, radius*8)

        self.fp2 = PointFeaturePropagation(in_channel1=init_channel*8, in_channel2=init_channel*4, out_channel=init_channel*2)
        self.up_conv3 = diffConv(init_channel*2, init_channel*4, radius*4)

        self.fp1 = PointFeaturePropagation(in_channel1=init_channel*4, in_channel2=init_channel*2, out_channel=init_channel*1)
        self.up_conv2 = diffConv(init_channel, init_channel*2, radius)

        self.global_conv1 = nn.Sequential(Conv1x1(init_channel*32+3, init_channel*8), 
                                        Conv1x1(init_channel*8, init_channel*2)
                                        )
        self.global_conv2 = nn.Sequential(Conv1x1(init_channel*16+3, init_channel*4),
                                        Conv1x1(init_channel*4, init_channel*2)                       
            )

        self.up_conv1 =Conv1x1(init_channel*(2+2+2)+3+category, 256)

        self.last_conv = nn.Linear(256, num_classes, bias=False)

        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(256, 256//8, 1, bias=False),
                                nn.BatchNorm1d(256//8),
                                nn.GELU(),
                                nn.Conv1d(256//8, 256, 1, bias=False),
                                nn.Sigmoid())
        
        self.drop = nn.Dropout(args.dropout)
                                
    def forward(self, x, l=None):
        batch_size = x.size(0)
        xyz = x.clone()
        point_num = xyz.size(1)

        x = self.le0(x)

        x = x.transpose(1,2).contiguous()
        l1_xyz, l1_feat = self.le1(xyz, x) # [B, N, 3] -> [B, N, 16]
        l1_feat = l1_feat.transpose(1,2).contiguous()   
        x = x.transpose(1,2).contiguous()
        
        # encoder
        l1_feat, l1_xyz = self.conv1(l1_feat, l1_xyz, point_num) 
        l2_feat, l2_xyz = self.conv2(l1_feat, l1_xyz, point_num//4)
        l3_feat, l3_xyz = self.conv3(l2_feat, l2_xyz, point_num//8) 
        l4_feat, l4_xyz = self.conv4(l3_feat, l3_xyz, point_num//16)
        l5_feat, l5_xyz = self.conv5(l4_feat, l4_xyz, point_num//32)

        # encode global feature
        emb1 = self.global_conv1(torch.cat((l5_xyz, l5_feat), dim=-1)) 
        emb1 = torch.max(emb1, dim=1, keepdim=True)[0] 

        emb2 = self.global_conv2(torch.cat((l4_xyz, l4_feat), dim=-1))
        emb2 = torch.max(emb2, dim=1, keepdim=True)[0] 

        if l is not None:
            l = l.view(batch_size, 1, -1) # [B, 1, 16]
        emb = torch.cat((emb1, emb2, l), dim=-1) 

        emb = emb.expand(-1, point_num, -1) 

        #decoder
        l4_feat = self.fp4(l4_xyz, l5_xyz, l4_feat, l5_feat)
        l4_feat, l4_xyz = self.up_conv5(l4_feat, l4_xyz, point_num//16)

        l3_feat = self.fp3(l3_xyz, l4_xyz, l3_feat, l4_feat) 
        l3_feat, l3_xyz = self.up_conv4(l3_feat, l3_xyz, point_num//8) 

        l2_feat = self.fp2(l2_xyz, l3_xyz, l2_feat, l3_feat) 
        l2_feat, l2_xyz = self.up_conv3(l2_feat, l2_xyz, point_num//4) 

        l1_feat = self.fp1(l1_xyz, l2_xyz, l1_feat, l2_feat) 
        l1_feat, l1_xyz = self.up_conv2(l1_feat, l1_xyz, point_num)    

        # feature fusion
        feat = torch.cat((l1_xyz, l1_feat, emb), dim=-1) 
        feat = self.up_conv1(feat) 

        feat = feat.transpose(1,2)
        score = self.se(feat)
        feat = feat.transpose(1,2)
        score = score.transpose(1,2)
 
        feat = feat*score
        
        feat = self.drop(feat)
        feat = self.last_conv(feat)
        feat = feat.transpose(1,2) 
        return feat
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--radius', type=float, default=0.005,
                        help='search radius')
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pc = torch.rand(5, 1024, 3).to(device)
    lb = torch.ones(5, 16).to(device)
    model = Model(args).to(device)
    model = nn.DataParallel(model)
    feat = model(pc, lb)
    print(feat.shape)