import torch
import torch.nn as nn


class ConvFeatExtractor(nn.Module):
    """Convolutional Feature Extractor for MNIST dataset

    Sushan Bastola(@devnson)의 MNIST 분류 모델 앞단의 변형

    원본: https://github.com/devnson/mnist_pytorch?tab=readme-ov-file#creating-convolutional-neural-network-model 
    """
    def __init__(self):
        super(ConvFeatExtractor, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(64 * 7 * 7, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape -> [b, 1, 28, 28]

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        # x.shape -> [b, 32, 14, 14]

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        # x.shape -> [b, 64, 7, 7]

        x = x.reshape(x.size(0), -1)
        # x.shape -> [b, 64 * 7 * 7]

        x = self.linear(x)
        feat = self.batch_norm(x)
        # x.shape -> [b, 128]

        return feat


class SoftDropout(nn.Module):
    """Softly drops out the nodes in neural networks

    ※ 네이버 부스트캠프에서 배웠던 건데 기법 이름이 정확히 뭐였는지 기억이 안남
    """
    def __init__(self, p:float=0.5, n:int=5):
        super(SoftDropout, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for _ in range(n)])

    def forward(self, x):
        outputs = torch.stack([dropout(x) for dropout in self.dropouts])
        return torch.sum(outputs, 0) / torch.numel(outputs)

class ClassifierHead(nn.Module):
    """Classifier head for MNIST dataset

    Sushan Bastola(@devnson)의 MNIST 분류 모델 뒷단의 변형

    원본: https://github.com/devnson/mnist_pytorch?tab=readme-ov-file#creating-convolutional-neural-network-model 
    """
    def __init__(self):
        super(ClassifierHead, self).__init__()
        self.dropout = SoftDropout(p=0.5, n=5)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.linear(x)

        return pred


class MNISTClassifier(nn.Module):
    """Classifier for MNIST dataset

    Sushan Bastola(@devnson)의 MNIST 분류 모델 전체의 변형된 버젼

    원본: https://github.com/devnson/mnist_pytorch?tab=readme-ov-file#creating-convolutional-neural-network-model 
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.feature_extractor = ConvFeatExtractor()
        self.classifier = ClassifierHead()

    def forward(self, x):
        x = self.feature_extractor(x)
        pred = self.classifier(x)

        return pred