import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        return x

class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation if enable_padding else 0
        super(CausalConv1d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        return result[:, :, :-self.__padding] if self.__padding != 0 else result

class CausalConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))] if enable_padding else 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        return super(CausalConv2d, self).forward(input)

class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out if act_func in ['glu', 'gtu'] else c_out, kernel_size=(Kt, 1))
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):   
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == 'glu':
            x_p, x_q = x_causal_conv[:, :self.c_out], x_causal_conv[:, -self.c_out:]
            x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
        elif self.act_func == 'gtu':
            x_p, x_q = x_causal_conv[:, :self.c_out], x_causal_conv[:, -self.c_out:]
            x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))
        elif self.act_func == 'relu':
            x = self.relu(x_causal_conv + x_in)
        elif self.act_func == 'silu':
            x = self.silu(x_causal_conv + x_in)
        else:
            raise NotImplementedError(f'ERROR: The activation function {self.act_func} is not implemented.')
        return x

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        x_list = [x]
        if self.Ks > 1:
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list.append(x_1)
        for k in range(2, self.Ks):
            x_k = torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[-1]) - x_list[-2]
            x_list.append(x_k)
        x = torch.stack(x_list, dim=2)
        x = torch.einsum('btkhi,kij->bthj', x, self.weight)
        return x + self.bias if self.bias is not None else x

class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out)) if bias else None
        self.gso = gso
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.einsum('hi,btij->bthj', self.gso, x)
        x = torch.einsum('bthi,ij->bthj', x, self.weight)
        return x + self.bias if self.bias is not None else x

class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.align = Align(c_in, c_out)
        if graph_conv_type == 'cheb_graph_conv':
            self.gconv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        else:
            self.gconv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x):
        x = self.align(x)
        x_gconv = self.gconv(x)
        x = x_gconv.permute(0, 3, 1, 2) + x
        return x

class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso, bias, droprate):
        super(STConvBlock, self).__init__()
        self.n_vertex = n_vertex
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([self.n_vertex, channels[2]])
        print(f"[DEBUG] LayerNorm shape = {[n_vertex, channels[2]]}")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)

class OutputBlock(nn.Module):
    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(channels[0], channels[1], bias=bias)
        self.fc2 = nn.Linear(channels[1], end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x).permute(0, 3, 1, 2)
