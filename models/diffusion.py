import torch
import torch.nn as nn


class MNISTDiffusion(nn.Module):
    """Diffusion model for MNIST dataset

    * MS Bing Chat과 함께 작성.
    """
    def __init__(self, img_height=28, img_width=28, label_dim=32, n_classes=10, num_steps=10, noise_std=0.25): 
        super(MNISTDiffusion, self).__init__()
        self.img_height = img_height
        self.img_width = img_width

        if n_classes > 0:
            self.label_emb = nn.Embedding(n_classes, label_dim)
        else:
            label_dim = 0
        self.n_classes = n_classes

        self.num_steps = num_steps
        self.noise_std = noise_std

        self.linear = nn.Linear(img_height*img_width+label_dim, img_height*img_width)

        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x, labels=None, backward=True):
        if backward:
            # Backward diffusion process
            x = x.view(x.size(0), -1)
            if self.n_classes > 0:
                c = self.label_emb(labels)

            x = self.linear(torch.cat([x, c], dim=1) if self.n_classes > 0 else x)
            x = x.view(x.size(0), 1, self.img_height, self.img_width)

            x = self.conv_1(x)
            x = self.relu(x)

            x_ = self.conv_2(x)
            x_ = self.relu(x_)
            x = x + x_

            x = self.conv_3(x)

            x = x.view(x.size(0), -1)
            
            return x.view(x.size(0), 1, self.img_height, self.img_width)
        
        else:
            # Forward diffusion process
            x = x.view(x.size(0), -1)
 
            noise = torch.randn_like(x) * self.noise_std
            x = x * (1 - self.noise_std) + noise

            return x.view(x.size(0), 1, self.img_height, self.img_width)
