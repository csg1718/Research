import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

class RAModule(nn.Module):
    def __init__(
        self,
        dim = 32,
        resolution = 20):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.embed = nn.Conv2d(self.dim*3, self.dim, kernel_size=1, padding=0)
        self.NormAct = nn.Sequential(
            nn.LayerNorm([self.dim, self.resolution, self.resolution]),
            nn.ReLU()
        )

    def forward(self, x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]

        diff12 = torch.abs(x1 - x2)
        diff23 = torch.abs(x2 - x3)
        diff31 = torch.abs(x3 - x1)

        cat_x1 = torch.cat([x1, diff12, diff31], dim=1)
        cat_x2 = torch.cat([x2, diff23, diff12], dim=1)
        cat_x3 = torch.cat([x3, diff31, diff23], dim=1)
        
        embed_x1 = self.embed(cat_x1)
        embed_x2 = self.embed(cat_x2)
        embed_x3 = self.embed(cat_x3)

        embed_x1 = F.softmax(embed_x1, dim=1)
        embed_x2 = F.softmax(embed_x2, dim=1)
        embed_x3 = F.softmax(embed_x3, dim=1)
        
        att_x1 = self.NormAct(embed_x1)
        att_x2 = self.NormAct(embed_x2)
        att_x3 = self.NormAct(embed_x3)

        aug_x1 = (x1 * att_x1).sum(2).sum(2)
        aug_x2 = (x2 * att_x2).sum(2).sum(2)
        aug_x3 = (x3 * att_x3).sum(2).sum(2)
        
        att_weight_x1 = att_x1.sum(1) / self.dim
        att_weight_x2 = att_x2.sum(1) / self.dim
        att_weight_x3 = att_x3.sum(1) / self.dim
        
        att_map = torch.stack([att_weight_x1, att_weight_x2, att_weight_x3],dim=1)
        attend_panel = torch.cat([aug_x1, aug_x2, aug_x3], dim=1)
        
        return att_map, attend_panel

class CNNModule(nn.Module):
    def __init__(self,
        dim = 32,
        layer_index = 4):
        super().__init__()
        self.dim = dim
        net = resnet18(weights=None)

        #Fiest 9 layers of ResNet-18
        conv = nn.Sequential(*list(net.children())[0:-4])

        #To match the dimension of RPM problems
        conv[0] = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)

        self.features = nn.Sequential(
            conv,
            #Last 1x1 conv layer
            nn.Conv2d(128, self.dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x = self.features(x)
        return x

class ReasoningModule(nn.Module):
    def __init__(self,
        chans):
        super().__init__()
        #6: the number of panels in pair of rows/colunms
        chans.insert(0,6)
        chan_pairs = list(zip(chans[:-1], chans[1:]))

        mlp = []
        for ind, (in_chan, out_chan) in enumerate(chan_pairs):
            mlp.append(nn.Linear(in_chan, out_chan))
            if ind < len(chans)-1:
                mlp.append(nn.ReLU(inplace = True))

        self.net = nn.Sequential(*mlp)

    def forward(self, x):
        return self.net(x)

class RANet(nn.Module):
    def __init__(self,
        image_size=160,
        dim=32,
        cnn_layers=10,
        mlp_dims=[64,32,5],
        ffn_divider=4):
        super(RANet, self).__init__()
        #parameters
        self.dim = dim
        assert (cnn_layers-2)%4 == 0, "The value of depth could be 6, 10, or 14"
        self.layer_index = (cnn_layers-2)/4-6
        self.stride = int(2 ** (cnn_layers/4 + 0.5))
        self.resolution = int(image_size/self.stride)

        #modules
        self.conv = CNNModule(dim=self.dim, layer_index=self.layer_index)
        self.attention = RAModule(dim=self.dim, resolution=self.resolution)
        self.reasoning = ReasoningModule(chans=mlp_dims)
        self.ff_ra = FeedForwardResidualBlock(self.dim, ffn_divider)
        self.ff_reasoning = FeedForwardResidualBlock(mlp_dims[-1]*self.dim)

        #scattering function
        self.scattering = Scattering()

    def forward(self, x: torch.Tensor):
        batch_size, num_panels, height, width = x.size()

        x = x.view(batch_size * num_panels, 1, height, width)
        x = self.conv(x)
        x = x.view(batch_size, num_panels, self.dim, self.resolution, self.resolution)
        
        row1 = x[:,:3].unsqueeze(1)
        row2 = x[:,3:6].unsqueeze(1)
        row3_p = x[:,6:8].unsqueeze(1).repeat(1,8,1,1,1,1)

        candidates = x[:,8:].unsqueeze(2)
        row3 = torch.cat([row3_p, candidates], dim=2)
        rows = torch.cat([row1, row2, row3], dim=1)
        
        rows = rows.view(-1, 3, self.dim, self.resolution, self.resolution)
        att_maps, att_rows = self.attention(rows)
        att_map = att_maps.view(batch_size, 10, 3, 1, self.resolution, self.resolution)
        att_rows = att_rows.view(batch_size,10,3,-1)
    
        x = self.ff_ra(att_rows)

        x = self.scattering(x, num_groups = self.dim)

        row1 = x[:,0].unsqueeze(1).repeat(1,8,1,1)
        row2 = x[:,1].unsqueeze(1).repeat(1,8,1,1)
        row3 = x[:,2:]
        
        row12 = torch.cat([row1,row2],dim=-1)
        row23 = torch.cat([row2,row3],dim=-1)
        row31 = torch.cat([row3,row1],dim=-1)

        common12 = self.reasoning(row12).flatten(2)
        common23 = self.reasoning(row23).flatten(2)
        common31 = self.reasoning(row31).flatten(2)

        common12 = self.ff_reasoning(common12)
        common23 = self.ff_reasoning(common23)
        common31 = self.ff_reasoning(common31)
        
        similarity1 = torch.einsum('bij,bij->bi', common12, common23)
        similarity2 = torch.einsum('bij,bij->bi', common12, common31)
        
        similarity = (similarity1+similarity2)/2
        common12 = common12[:,0,:].squeeze()
        return similarity, att_map, common12


class FeedForwardResidualBlock(nn.Module):
    def __init__(self, dim, ffn_divider=4):
        super(FeedForwardResidualBlock, self).__init__()
        self._projection = nn.Sequential(
            nn.Linear(dim, dim // ffn_divider),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim // ffn_divider),
            nn.Linear(dim // ffn_divider, dim)
        )

    def forward(self, x: torch.Tensor):
        return x + self._projection(x)


class Scattering(nn.Module):
    def forward(self, x, num_groups):
        """
        :param x: a Tensor with rank >= 3 and last dimension divisible by number of groups
        :param num_groups: number of groups
        """
        shape_1 = x.shape[:-1] + (num_groups,) + (x.shape[-1] // num_groups,)
        x = x.view(shape_1)
        x = x.transpose(-3, -2).contiguous()
        return x.flatten(start_dim=-2)

images = torch.randn(32,16,160,160)
labels = torch.tensor([2])

model = RANet()

output = model(images)
