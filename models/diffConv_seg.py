import torch
d = torch.load('../checkpoints/model_cls.pth')
new_d = {}
for k in d.keys():
    new_k = k
    if 'agg' in k:
        new_k = k.replace('agg', 'mat')
    if 'conv_v.0' in k:
        new_k = k.replace('conv_v.0', 'conv_v')
    if 'conv0' in k:
        new_k = k.replace('conv0', 'last_conv')
            
    new_d[new_k] = d[k]
    
torch.save(new_d, '../checkpoints/model_cls.pth')