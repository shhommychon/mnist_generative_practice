import torch
import torch.nn as nn


class MNISTVae(nn.Module):
    """Variational Autoencoder for MNIST dataset

    * MS Bing Chat과 함께 작성.
    """
    def __init__(self, latent_dim=96, image_dim=28*28, label_dim=16, n_classes=10):
        super(MNISTVae, self).__init__()

        # label embedding as condition for VAE
        if n_classes > 0:
            self.label_emb = nn.Embedding(n_classes, label_dim)
        else:
            label_dim = 0

        # Encoder
        self.fc_mu = nn.Linear(image_dim, latent_dim)  # mu layer
        self.fc_logvar = nn.Linear(image_dim, latent_dim)  # logvariance layer

        # Decoder
        self.linear = nn.Linear(latent_dim+label_dim, image_dim)

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def encode(self, x):
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, labels=None):
        x = x.view(x.size(0), -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def decode(self, z, labels=None):
        if labels is not None:
            condition = self.label_emb(labels)
            z = torch.cat([z, condition], 1)

        x = self.linear(z)

        x = x.view(x.size(0), 1, 28, 28)

        x = self.conv_1(x)
        x = self.relu(x)

        x_ = self.conv_2(x)
        x_ = self.relu(x_)
        x = x + x_

        x = self.conv_3(x)

        x = self.tanh(x)

        return x #.type(torch.float16)