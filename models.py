'''
Models

1 - Spatial
2 - Temporal (3D optical flow)
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
# Hyper parameters
hidden_dimension_size = 64
lstm_dropout = 0.5
####################




class Model_1(nn.Module):
    ''' Spatial '''
    def __init__(self, hidden_dimension_size, lstm_dropout, frozen_weights=False,
                 single_feature=False):
        super(Model_1, self).__init__()

        # Single feature or multiple
        self.single_feature = single_feature

        # Set up base image feature extractor
        self.base_model = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        base_model_fc_size = list(self.base_model.parameters())[-1].size(0)

        # Freeze weights
        if frozen_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # TEST ############
        ###################
        for param in list(self.base_model.parameters())[:30]:
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
        self.preds_single_feature = nn.Linear(base_model_fc_size, NUM_CLASSES)

    def forward(self, X):
        # Single feature
        if self.single_feature:
            return self.preds_single_feature(torch.squeeze(self.base_model(X)))

        # Stack individual image features
        base_model_out = [self.base_model(X[:,i]) for i in range(X.size()[1])]
        x = torch.squeeze(torch.stack(base_model_out))

        # LSTM & final layer
        x, _ = self.lstmlayer(x)
        return self.preds(x[-1])





class Model_2(nn.Module):
    ''' Temporal (3D optical flow) '''
    def __init__(self, hidden_dimension_size, lstm_dropout, single_feature=False):
        super(Model_2, self).__init__()

        # Single feature or multiple
        self.single_feature = single_feature

        # Conv layers
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

        # LSTM Layer
        self.hidden_dimension_size = hidden_dimension_size
        self.lstm_dropout = lstm_dropout
        self.lstmlayer = nn.LSTM(
            input_size  = 512,
            hidden_size = hidden_dimension_size,
            dropout     = lstm_dropout
        )

        # Final layer
        self.preds = nn.Linear(hidden_dimension_size, NUM_CLASSES)
        self.preds_single_feature = nn.Linear(512, NUM_CLASSES)

    def forward(self, X):
        # No temporal if single feature
        if self.single_feature:
            x = torch.squeeze(self.convlayer4(self.convlayer3(self.convlayer2(self.convlayer1(X)))))
            return self.preds_single_feature(x)

        # Stack individual image features
        conv_layers_out = []
        for chunk in range(X.size(1)):
            x = self.convlayer1(X[:, chunk])
            x = self.convlayer2(x)
            x = self.convlayer3(x)
            x = self.convlayer4(x)
            conv_layers_out.append(x)
        x = torch.squeeze(torch.stack(conv_layers_out))

        # LSTM & final layer
        x, _ = self.lstmlayer(x)
        return self.preds(x[-1])





class Model_3(nn.Module):
    ''' 2 Stream '''
    def __init__(self, hidden_dimension_size, lstm_dropout):
        super(Model_3, self).__init__()

        # Set up base image feature extractor
        self.base_model = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        base_model_fc_size = list(self.base_model.parameters())[-1].size(0)

        # Freeze weights
        # if frozen_weights:
        #     for param in self.base_model.parameters():
        #         param.requires_grad = False

        # TEST ############
        ###################
        for param in list(self.base_model.parameters())[:51]:
            param.requires_grad = False

        # Conv layers
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

        # LSTM Layer
        self.lstmlayer_temporal = nn.LSTM(
            input_size  = 512,
            hidden_size = hidden_dimension_size,
            dropout     = lstm_dropout
        )
        self.lstmlayer_spatial = nn.LSTM(
            input_size  = base_model_fc_size,
            hidden_size = hidden_dimension_size,
            dropout     = lstm_dropout
        )

        # Final layer
        # self.preds_temporal = nn.Linear(hidden_dimension_size, NUM_CLASSES)
        # self.preds_spatial  = nn.Linear(hidden_dimension_size, NUM_CLASSES)
        self.preds_two_stream = nn.Linear(hidden_dimension_size + hidden_dimension_size, NUM_CLASSES)

    def forward(self, images, op_flow):
        # Temporal - optical flow
        conv_layers_out = []
        for chunk in range(op_flow.size(1)):
            x1 = self.convlayer1(op_flow[:, chunk])
            x1 = self.convlayer2(x1)
            x1 = self.convlayer3(x1)
            x1 = self.convlayer4(x1)
            conv_layers_out.append(x1)
        x1 = torch.squeeze(torch.stack(conv_layers_out))
        x1, _ = self.lstmlayer_temporal(x1)
        # out_temporal = self.preds_temporal(x1[-1])

        # Spatial - rgb images
        base_model_out = [self.base_model(images[:,i]) for i in range(images.size()[1])]
        x2 = torch.squeeze(torch.stack(base_model_out))
        x2, _ = self.lstmlayer_spatial(x2)
        # out_spatial = self.preds_spatial(x2[-1])

        out_cat = torch.cat([x1[-1], x2[-1]])

        return self.preds_two_stream(out_cat)



class Model_4(nn.Module):
    ''' 3D Spatial (3D images) '''
    def __init__(self, hidden_dimension_size, lstm_dropout, single_feature=False):
        super(Model_4, self).__init__()
        pass

    def forward(self, X):
        pass


class Model_5(nn.Module):
    ''' 2D Temporal (2D optical flow) '''
    def __init__(self, hidden_dimension_size, lstm_dropout, single_feature=False):
        super(Model_5, self).__init__()

        # Single feature or multiple
        self.single_feature = single_feature

        # Conv layers
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(2 * T, 96, kernel_size=7, padding=1, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3))
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3))
        self.convlayer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(3))
        self.convlayer4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(3))

        # LSTM Layer
        self.hidden_dimension_size = hidden_dimension_size
        self.lstm_dropout = lstm_dropout
        self.lstmlayer = nn.LSTM(
            input_size  = 512,
            hidden_size = hidden_dimension_size,
            dropout     = lstm_dropout
        )

        # Final layer
        self.preds = nn.Linear(hidden_dimension_size, NUM_CLASSES)
        self.preds_single_feature = nn.Linear(512, NUM_CLASSES)

    def forward(self, X):
        # Stack individual image features
        conv_layers_out = []
        for chunk in range(X.size(1)):
            x = self.convlayer1(X[:, chunk])
            x = self.convlayer2(x)
            x = self.convlayer3(x)
            x = self.convlayer4(x)
            conv_layers_out.append(x)
        x = torch.squeeze(torch.stack(conv_layers_out))

        # LSTM & final layer
        x, _ = self.lstmlayer(x)
        return self.preds(x[-1])
