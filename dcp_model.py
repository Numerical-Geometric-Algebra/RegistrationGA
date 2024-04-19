from dcp.model import DCP
import torch

class coco():
    def __init__(self):

        self.emb_nn = 'dgcnn'
        self.pointer = 'transformer'
        self.head = 'svd'
        self.n_blocks = 1
        self.dropout = 0
        self.ff_dims = 1024
        self.n_heads = 4
        self.exp_name = 'exp'
        self.emb_dims = 512
        self.cycle = False

# torch.Size([10, 3, 1024])
args = coco()
model_path = 'dcp/pretrained/dcp_v2.t7'

net = DCP(args)
net.load_state_dict(torch.load(model_path), strict=False)

net = net.float()