import torch
import torch.nn as nn
from Layer import Conv_layer



class VGG8(torch.nn.Module):
    def __init__(self,device,epochs,num_classes=10, input_channel=1):
        super(VGG8, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.small_sample_dict = {}
        self.masks = None
        self.blocks = nn.ModuleList()
        self.schedulers = []
        self.optimisers = []
        self.lr = 2e-4
        self.wc = 1e-3
        self.lr_min = 1e-5
        self.T_max = epochs
        self.dropout = 0.1
        self.c = input_channel

        # 128*32*32
        self.blocks.append(self.create_layer(self.c,128,3,1,1,False))

        # 128*16*16
        self.blocks.append(self.create_layer(128, 256, 3, 1, 1, True))

        # 256*16*16
        self.blocks.append(self.create_layer(256, 256, 3, 1, 1, False))

        # 256*8*8
        self.blocks.append(self.create_layer(256, 512, 3, 1, 1, True))

        # 512*4*4
        self.blocks.append(self.create_layer(512, 512, 3, 1, 1, True))

        # 512*2*2
        self.blocks.append(self.create_layer(512, 512, 3, 1, 1, True))

        # 512*2*2
        # self.blocks.append(self.create_linear(2048,1024, False, True))
        self.blocks.append(self.create_layer(512, 512, 3, 1, 1, False))

        # FC_layer
        self.fc_layers = nn.ModuleList([
            nn.Linear(256, num_classes, bias=True),
            nn.Linear(256, num_classes, bias=True),
            nn.Linear(512, num_classes, bias=True),
            nn.Linear(512, num_classes, bias=True),
            nn.Linear(512, num_classes, bias=True),
            nn.Linear(512, num_classes, bias=True),
        ])


        #Classifier
        self.classifier_optimiser = torch.optim.AdamW(self.fc_layers.parameters(), lr=self.lr, weight_decay=self.wc)
        self.classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.classifier_optimiser, T_max=self.T_max,eta_min=self.lr_min)
        self.optimisers.append(self.classifier_optimiser)
        self.schedulers.append(self.classifier_scheduler)
        self.classifier_loss_fn = nn.CrossEntropyLoss()


        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    def create_layer(self,in_c,out_c, kernel_size, stride, padding, pooling):
        layer = Conv_layer(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size,
                      stride=stride, padding=padding,dropout=self.dropout,pooling=pooling)
        optimiser = torch.optim.AdamW(layer.parameters(), lr=self.lr, weight_decay=self.wc)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.T_max,eta_min=self.lr_min)
        layer.optimiser = optimiser
        layer.scheduler = scheduler
        self.schedulers.append(scheduler)
        self.optimisers.append(optimiser)
        return layer


    @torch.no_grad()
    def _layer_norm(self, x):
        tiny = torch.exp(torch.tensor(-50.0, device=self.device))
        rms = torch.sqrt(torch.mean(x ** 2, dim=[1, 2, 3], keepdim=True)) + tiny
        return x / rms


    def forward_pass(self, imgs, labs):

        normalised_activations=[]

        z = self._layer_norm(imgs)

        for id, block in enumerate(self.blocks):

            z = block(z, True, labs)

            if id > 0:
                if isinstance(block, Conv_layer):
                    normalised_activations.append(torch.mean(z,dim=[2,3]))
                else:
                    normalised_activations.append(z)

        return normalised_activations


    def classifier_pass(self, features, labs):

        batch_size = features[0].shape[0]

        logits = torch.zeros(batch_size, self.num_classes, device=self.device)

        for id, layer in enumerate(self.fc_layers):
            logits = logits + layer(features[id])

        logits = logits - logits.max(dim=1, keepdim=True)[0]

        classifier_loss = self.classifier_loss_fn(logits, labs)

        self.classifier_optimiser.zero_grad()

        classifier_loss.backward()

        self.classifier_optimiser.step()

        return logits.detach(), classifier_loss.detach().item()


    def forward(self, imgs, labs):

        normalised_activations = self.forward_pass(imgs,labs)

        logits, loss = self.classifier_pass(normalised_activations, labs)

        _, predictions = torch.max(logits, dim=1)

        train_errors = torch.sum(predictions != labs).item()


        return loss, train_errors


    def test(self, imgs, labs):

        batch_size = imgs.shape[0]

        normalised_activations = []

        z = self._layer_norm(imgs)

        for id, block in enumerate(self.blocks):

            z = block(z ,False, labs)

            if id > 0:
                if isinstance(block, Conv_layer):
                    normalised_activations.append(torch.mean(z, dim=[2, 3]))

                else:
                    normalised_activations.append(z)

        logits = torch.zeros(batch_size, self.num_classes, device=self.device)

        for id, layer in enumerate(self.fc_layers):
            logits = logits + layer(normalised_activations[id])

        _, predictions = torch.max(logits, dim=1)

        test_errors = torch.sum(predictions != labs).item()

        return test_errors