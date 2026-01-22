import random
import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class Conv_layer(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 optimiser=None,
                 scheduler=None,
                 num_classes=10,
                 stride=1,
                 padding=0,
                 dropout=0,
                 pooling=False,
                 ):
        super(Conv_layer,self).__init__()


        self.out_channels=out_channels

        self.num_classes=num_classes

        self.optimiser = optimiser

        self.scheduler = scheduler

        self.layer_class_fn =nn.CrossEntropyLoss()

        self.layer_back_fn = nn.MSELoss()

        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)

        self.relu  = nn.ReLU(inplace=False)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = nn.Dropout(dropout)

        self.pooling = pooling

        self.AvgPool = nn.AvgPool2d(2)

        self.MaxPool = nn.MaxPool2d(2)

        self.temp = 1

        self.loss_batch = []

        self.loss_epoch = []

        self.acc_epoch = []

        self.valid_batch = []

        self.valid_epoch = []

        self.acc_batch = []

        self.num_features = {
            64: 4096,
            128: 2048,
            256: 1024,
            512: 512,
        }

        R1 = torch.randn(self.num_features[self.out_channels],num_classes) / (num_classes ** 0.5)

        b = torch.randn(num_classes) / math.sqrt(num_classes)

        self.register_buffer('R1', R1)

        self.register_buffer('b', b)



    @torch.no_grad()
    def _layer_norm(self, x):
        tiny = torch.exp(torch.tensor(-50.0, device=self.device))
        rms = torch.sqrt(torch.mean(x ** 2, dim=[1, 2, 3], keepdim=True)) + tiny
        return x / rms


    def channel_spatial_goodness(self,x):

        B, C, H, W = x.shape

        output_size = {
            64 : 8,
            128 : 4,
            256 : 2,
            512 : 1,
        }

        x_square = x ** 2

        pooled = F.adaptive_avg_pool2d(x_square, output_size=output_size[self.out_channels])


        goodness = pooled.view(B,-1)

        return goodness

    def forward(self,x,train=False,labels=None):

        x = self.conv(x)

        x = self.relu(x)


        channel_goodness = self.channel_spatial_goodness(x)

        projected_1 = channel_goodness @ self.R1 + self.b

        if train:


            layer_loss = self.layer_class_fn(projected_1,labels)

            self.optimiser.zero_grad()

            layer_loss.backward()

            self.optimiser.step()

        with torch.no_grad():

            preds = torch.argmax(projected_1, dim=1)

            correct = (preds == labels).float().sum().item()

            acc = correct / labels.size(0)

            if train:
                self.acc_batch.append(acc)
            else:
                self.valid_batch.append(acc)

        if self.pooling:

            x = self.AvgPool(x)

        x = self.dropout(x)

        x = self._layer_norm(x)

        x = x.detach()

        return x