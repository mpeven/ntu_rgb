'''
Models

1 - Spatial
    1.1 - Single image
    1.2 - Multiple images
2 - Temporal
    2.1 - Single chunk
    2.2 - Multiple chunks
3 - Two-stream

'''

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable



####################
# Constants
NUM_CLASSES = 60
T = 10 # Number of frames for each feature
C = 3  # Number of channels
####################


####################
# Feature extractors
resnet = models.resnet18(pretrained=True)
spatial_base_model = nn.Sequential(*list(resnet.children())[:-1])

####################
# Hyper parameters
hidden_dimension_size = 64
lstm_dropout = 0.5
####################




class Model_1_1(nn.Module):
    ''' Spatial - Single image '''
    def __init__(self, frozen_weights=False):
        super(Model_1_1, self).__init__()

        # Set up base image feature extractor
        self.base_model = spatial_base_model
        base_model_fc_size = list(self.base_model.parameters())[-1].size(0)

        # Freeze weights
        if frozen_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Prediction layer
        self.preds = nn.Linear(base_model_fc_size, NUM_CLASSES)

    def forward(self, X):
        x = self.base_model(X)
        x = x.view(-1, int(np.prod(x.size()[1:]))) # Flatten
        return self.preds(x)





class Model_1_2(nn.Module):
    ''' Spatial - Multiple images '''
    def __init__(self, hidden_dimension_size, lstm_dropout, frozen_weights=False):
        super(Model_1_2, self).__init__()

        # Set up base image feature extractor
        self.base_model = spatial_base_model
        base_model_fc_size = list(self.base_model.parameters())[-1].size(0)

        # Freeze weights
        if frozen_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # LSTM Layer
        self.hidden_dimension_size = hidden_dimension_size
        self.lstm_dropout = lstm_dropout
        self.lstmlayer = nn.LSTM(
            input_size  = base_model_fc_size,
            hidden_size = hidden_dimension_size,
            dropout     = lstm_dropout
        )

        # Final layer
        self.preds = nn.Linear(hidden_dimension_size, NUM_CLASSES)

    def forward(self, X):
        # Stack individual image features
        base_model_out = [self.base_model(X[:,i]) for i in range(X.size()[1])]
        x = torch.squeeze(torch.stack(base_model_out))

        # LSTM & final layer
        x, _ = self.lstmlayer(x)
        return self.preds(x[-1])





class Model_2_1(nn.Module):
    ''' Temporal - Single chunk '''
    def __init__(self, batch_size):
        super(Model_2_1, self).__init__()
        self.convlayer1 = nn.Sequential(
            nn.Conv3d(C * T, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.convlayer2 = nn.Sequential(
            nn.Conv3d(96, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            # nn.Dropout3d(.2),
            nn.MaxPool3d(3))
        self.convlayer3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # nn.Dropout3d(.2),
            nn.MaxPool3d(3))
        self.convlayer4 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # nn.Dropout3d(.2),
            nn.MaxPool3d(3))
        self.timelayer1 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=0),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            # nn.Dropout3d(.2),
            )
        self.preds = nn.Linear(512, NUM_CLASSES)

    def forward(self, X):
        x = self.convlayer1(X)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = x.view(-1, int(np.prod(x.size()[1:]))) # Flatten
        return self.preds(x)
