import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DNN(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        #[210] - >[200]
        self.emb_out_linear = nn.Linear(self.in_dims[0] + self.time_emb_dim, self.in_dims[0])

        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        if x.dim() ==3:
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1) 
        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        h = self.emb_out_linear(h)
        return h


class TransformerDNN(nn.Module):
    """
    A Transformer-based neural network for the reverse diffusion process.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=True, dropout=0.5, nhead=4, num_layers=1):
        super(TransformerDNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # Embedding layer for concatenating time embedding with input features
        if self.time_type == "cat":
            self.input_projection = nn.Linear(self.in_dims[0] + self.time_emb_dim, self.in_dims[0])
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        
        # Define Transformer Encoder Layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.in_dims[0], nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
        # Output Layer
        self.output_projection = nn.Linear(self.in_dims[0], self.out_dims[-1])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        # Initialize the weights of the embedding and output layers
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
        
        size = self.input_projection.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.input_projection.weight.data.normal_(0.0, std)
        self.input_projection.bias.data.normal_(0.0, 0.001)
        
        size = self.output_projection.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.output_projection.weight.data.normal_(0.0, std)
        self.output_projection.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        
        # Normalize input if specified
        if self.norm:
            x = F.normalize(x)
        
        # Apply dropout to input
        x = self.drop(x)

        if x.dim() == 3: #([64, 19, 200])
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1) #
        elif x.dim() == 2: 
            emb = emb #torch.Size([64, 10])


        h = torch.cat([x, emb], dim=-1)# torch.Size([64, 19, 210])
        
        h = self.input_projection(h)
        if h.dim() == 3:
            h = self.transformer_encoder(h)
        elif h.dim() == 2:
            h = h.unsqueeze(1)  # Add sequence dimension
            h = self.transformer_encoder(h)
            h = h.squeeze(1)  # Remove sequence dimension

        h = self.transformer_encoder(h) #torch.Size([64, 200])

        h = self.output_projection(h) #torch.Size([64, 200])

        return h


class UNet(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(UNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dim = self.in_dims[0] + self.time_emb_dim
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)

        # Encoder (downsampling)
        self.enc1 = nn.Linear(in_dim, 64)
        self.enc2 = nn.Linear(64, 128)
        self.enc3 = nn.Linear(128, 256)

        # Decoder (upsampling)
        self.dec3 = nn.Linear(256, 128)
        self.dec2 = nn.Linear(256, 64)  # 256 because of skip connection
        self.dec1 = nn.Linear(128, self.out_dims[0])  # 128 because of skip connection

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        if x.dim() == 3:
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        h = torch.cat([x, emb], dim=-1)

        # Encoder
        e1 = F.relu(self.enc1(h))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))

        # Decoder with skip connections
        d3 = F.relu(self.dec3(e3))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=-1)))
        d1 = self.dec1(torch.cat([d2, e1], dim=-1))

        return d1


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.lin1(x))
        out = self.lin2(out)
        return F.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(ResNet, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dim = self.in_dims[0] + self.time_emb_dim
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)

        self.input_layer = nn.Linear(in_dim, 64)
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64)
        )
        self.output_layer = nn.Linear(64, self.out_dims[0])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        if x.dim() == 3:
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        h = torch.cat([x, emb], dim=-1)

        h = self.input_layer(h)
        h = self.res_blocks(h)
        h = self.output_layer(h)

        return h
