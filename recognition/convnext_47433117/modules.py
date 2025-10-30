import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 7x7 DW conv preserves spatial dims because padding=3
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = shortcut + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=2, depths=[3,3,9,3], dims=[96,192,384,768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1., classifier_dropout=0.0):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # Stem
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0], 4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        ))
        # Downsample
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], 2, stride=2)
            ))
        # Stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[Block(dims[i], drop_path=dp_rates[cur+j], layer_scale_init_value=layer_scale_init_value)
                                    for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(classifier_dropout) if classifier_dropout>0 else nn.Identity()
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0) # type: ignore

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2,-1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format=="channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x-u).pow(2).mean(1, keepdim=True)
            x = (x-u)/torch.sqrt(s+self.eps)
            x = self.weight[:,None,None]*x + self.bias[:,None,None]
            return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob==0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],)+(1,)*(x.ndim-1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob)*random_tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    def norm_cdf(x): return (1.+math.erf(x/math.sqrt(2.)))/2
    with torch.no_grad():
        l = norm_cdf((a-mean)/std)
        u = norm_cdf((b-mean)/std)
        tensor.uniform_(2*l-1, 2*u-1)
        tensor.erfinv_()
        tensor.mul_(std*math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a,max=b)
    return tensor



def custom_small(drop_path_rate=0.15, layer_scale_init_value=1e-6, head_init_scale=1, classifier_dropout=0.3):
    return ConvNeXt(depths=[3,3,27,3], dims=[96,192,384,768],
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    head_init_scale=head_init_scale,
                    classifier_dropout=classifier_dropout)

def custom_model(drop_path_rate=0.15, layer_scale_init_value=1e-6, head_init_scale=1, classifier_dropout=0.3):
    return ConvNeXt(depths=[3,3,9,3], dims=[96,192,384,768],
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    head_init_scale=head_init_scale,
                    classifier_dropout=classifier_dropout)
