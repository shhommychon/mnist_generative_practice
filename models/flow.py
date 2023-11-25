import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanarFlow(nn.Module):
    """Planar Flow

    Normalizing Flow transformation introduced in "Variational Inference with Normalizing Flows"
    by Rezende, Mohamed. (2015, doi: 10.48550/arXiv.1505.05770) 

    * MS Bing Chat과 함께 작성.
    """
    def __init__(self, latent_dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, latent_dim))
        self.scale = nn.Parameter(torch.Tensor(1, latent_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(-0.01, 0.01)
        self.scale.data.uniform_(-0.01, 0.01)
        self.bias.data.uniform_(-0.01, 0.01)

    def forward(self, z):
        psi = torch.tanh(F.linear(z, self.weight, self.bias))
        return z + self.scale * psi

    def inverse(self, z):
        # This is an approximation to the inverse of the tanh function
        psi = torch.atanh((z - self.bias) / self.weight)
        return z - self.scale * psi


class MNISTFlow(nn.Module):
    """Generative Flow model for MNIST dataset

    * MS Bing Chat과 함께 작성.
    """
    def __init__(self, input_dim=28*28, output_dim=28*28, hidden_dim=128, n_classes=10):
        super(MNISTFlow, self).__init__()

        self.hidden_dim = hidden_dim

        # label embedding as condition for generative flow model
        if n_classes > 0:
            self.label_emb = nn.Embedding(n_classes, hidden_dim)
        else:
            hidden_dim = 0
        self.n_classes = n_classes

        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, input_dim)

        # Flow
        self.flow = PlanarFlow(hidden_dim)

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x_enc = self.enc(x)
        return x_enc

    def decode(self, x_enc):
        x = self.dec(x_enc)
        return x

    def generate_embedding(self, c=None, batch_num=1):
        # class embedding as condition
        if self.n_classes > 0:
            batch_num = c.size(0)
            c = self.label_emb(c)
        else:
            c = torch.zeros(batch_num, self.hidden_dim, requires_grad=False)

        # random
        z = torch.randn_like(c)
        z_enc = z + c
        return z_enc

    def transformation(self, input, reverse=False):
        if not reverse:
            x_enc = self.flow(input)
            return x_enc
        else:
            z_enc = self.flow.inverse(input)
            return z_enc

    def conversion(self, z_enc, c0, c1):
        c0 = self.label_emb(c0)
        c1 = self.label_emb(c1)

        z = z_enc - c0
        z_enc = z + c1
        return z_enc

    def forward(self, z_enc):
        x = self.decode(z_enc)
        x = x.view(x.size(0), 1, 28, 28)

        x = self.conv_1(x)
        x = self.relu(x)

        x_ = self.conv_2(x)
        x_ = self.relu(x_)
        x = x + x_

        x = self.conv_3(x)

        x = self.tanh(x)

        return x #.type(torch.float16)
