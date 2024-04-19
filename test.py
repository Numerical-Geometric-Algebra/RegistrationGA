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

args = coco()

filename = '/home/francisco/Code/Stanford Dataset/bunny/reconstruction/bun_zipper_res2.ply'


model_path = 'dcp/pretrained/dcp_v2.t7'

net = DCP(args)
net.load_state_dict(torch.load(model_path), strict=False)



# rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(verts, verts)
