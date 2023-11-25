import torch
import torch.nn as nn


class MNISTGan(nn.Module):
    """Generative Adversarial Network for MNIST dataset

    * MS Bing Chat과 함께 작성.
    """
    def __init__(self, input_dim=128, output_dim=28*28, label_dim=4, n_classes=10):
        super(MNISTGan, self).__init__()

        # label embedding as condition for GAN
        if n_classes > 0:
            self.label_emb = nn.Embedding(n_classes, label_dim)
        else:
            label_dim = 0

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.hidden_layer = nn.Linear(input_dim+label_dim, output_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, labels=None):
        if labels is not None:
            condition = self.label_emb(labels)
            x = torch.cat([x, condition], 1)

        x = self.hidden_layer(x)

        x = x.view(x.size(0), 1, 28, 28)

        x = self.conv_1(x)
        x = self.relu(x)

        x_ = self.conv_2(x)
        x_ = self.relu(x_)
        x = x + x_

        x = self.conv_3(x)

        x = self.tanh(x)

        return x #.type(torch.float16)


class Discriminator(nn.Module):
    """GAN Discriminator for MNIST dataset

    * MS Bing Chat과 함께 작성.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.hidden_layer = nn.Linear(28*28, 256)
        self.output_layer = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y):
        y = y.view(y.size(0), -1)
        y = self.hidden_layer(y)
        y = self.relu(y)
        y = self.output_layer(y)
        real_fake = self.sigmoid(y)

        return real_fake
