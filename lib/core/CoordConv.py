import torch
import torch.nn as nn
from einops import rearrange
# 原版本
# class AddCoords(nn.Module):

#     def __init__(self, with_r=False):
#         super().__init__()
#         self.with_r = with_r

#     def forward(self, input_tensor):
#         """
#         Args:
#             input_tensor: shape(batch, channel, x_dim, y_dim)
#         """
#         batch_size, _, x_dim, y_dim = input_tensor.size()

#         xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
#         yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

#         xx_channel = xx_channel.float() / (x_dim - 1)
#         yy_channel = yy_channel.float() / (y_dim - 1)

#         xx_channel = xx_channel * 2 - 1
#         yy_channel = yy_channel * 2 - 1

#         xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
#         yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

#         ret = torch.cat([
#             input_tensor,
#             xx_channel.type_as(input_tensor),
#             yy_channel.type_as(input_tensor)], dim=1)

#         if self.with_r:
#             rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
#             ret = torch.cat([ret, rr], dim=1)

#         return ret

# class CoordConv(nn.Module):

#     def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
#         super().__init__()
#         self.addcoords = AddCoords(with_r=with_r)
#         in_size = in_channels+2
#         if with_r:
#             in_size += 1
#         self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

#     def forward(self, x):
#         ret = self.addcoords(x)
#         ret = self.conv(ret)
#         return ret

# '''
# PyTorch 的另一种实现方式，可自动插入 x-y 尺寸/维度。
# An alternative implementation for PyTorch with auto-infering the x-y dimensions.
# '''
# class AddCoords(nn.Module):

#     def __init__(self, with_r=False):
#         super().__init__()
#         self.with_r = with_r

#     def forward(self, input_tensor):
#         """
#         Args:
#             input_tensor: shape(batch, channel, x_dim, y_dim)
#         """
#         batch_size, _, x_dim, y_dim = input_tensor.size()

#         xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1).transpose(1, 2)
#         yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1)
#         #print(xx_channel)

#         xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
#         yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

#         xx_channel_01 = xx_channel.float() / (x_dim - 1)
#         yy_channel_01 = yy_channel.float() / (y_dim - 1)

#         xx_channel = xx_channel_01 * 2 - 1
#         yy_channel = yy_channel_01 * 2 - 1

#         ret = torch.cat([
#             input_tensor,
#             xx_channel.type_as(input_tensor),
#             yy_channel.type_as(input_tensor)], dim=1)

#         if self.with_r:
#             rr = torch.sqrt(torch.pow(xx_channel_01.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel_01.type_as(input_tensor) - 0.5, 2))
#             ret = torch.cat([ret, rr], dim=1)

#         return ret


# class CoordConv(nn.Module):

#     def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
#         super().__init__()
#         self.addcoords = AddCoords(with_r=with_r)
#         in_size = in_channels+2
#         if with_r:
#             in_size += 1
#         #self.conv = nn.Conv2d(in_size, out_channels, **kwargs)
#         self.conv = nn.Linear(in_size, out_channels, bias=False)

#     def forward(self, x):
#         ret = self.addcoords(x)
#         ret = self.conv(ret)
#         return ret

'''
PyTorch 的另一种实现方式，可自动插入 x-y 尺寸/维度。
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''
class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1).transpose(1, 2)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1)
        #print(xx_channel)

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

        xx_channel_01 = xx_channel.float() / (x_dim - 1)
        yy_channel_01 = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel_01 * 2 - 1
        yy_channel = yy_channel_01 * 2 - 1
        #xx_channel = rearrange(xx_channel, 'b c h w -> b c (h w)') 
        #yy_channel = rearrange(yy_channel, 'b c h w -> b c (h w)')
        
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        return ret


class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size = 1, stride=1)
        
        

    def forward(self, x):
        ret = self.addcoords(x) # B K+2 64 48
        ret = self.conv(ret)
        ret = rearrange(ret, 'b c h w -> b c (h w)')
        return ret

class CoordConv2(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=False)
        in_size = in_channels+2
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret