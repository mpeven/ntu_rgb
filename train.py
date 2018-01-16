import numpy as np
from ntu_rgb import NTU
from feature_manager import FeatureManager
import datetime as dt
import line_profiler
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



####################
# Constants

EPOCHS = 10
NUM_VOXELS = 100*100*100
NUM_CLASSES = 60
K = 5 # Number of features per video
T = 10 # Number of frames for each feature
C = 3  # Number of channels

#
####################


####################
# Dataset

dataset = NTU()
features = FeatureManager()

#
####################


class Net(nn.Module):
    """
    Network architecture
    """
    def __init__(self):
        super(Net, self).__init__()
        # Conv3D: in_channels, out_channels, kernel_size, stride, padding
        self.layer1 = nn.Sequential(
            nn.Conv3d(C * T, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.layer2 = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.layer3 = nn.Sequential(
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.layer4 = nn.Sequential(
            nn.Conv3d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.fc1 = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU())
        self.preds = nn.Linear(2048 * K, NUM_CLASSES)

    # TODO
    # Get rid of fully connected layers
    # Concat them into a 5x512 tensor
    # Do 2 1D convolutions on those
    # Predict on 1xNumFilters ouput of the 1D convolutions
    def forward(self, X):
        all_x = []
        for k in range(K):
            x = self.layer1(X[:,k])
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = x.view(-1, int(np.prod(x.size()[1:]))) # Flatten
            x = self.fc1(x)
            x = self.fc2(x)
            all_x.append(x)
        y_out = self.preds(torch.cat(all_x, 1))
        return y_out




def train_batch_generator(test=False):
    """
    Dataset can't fit in memory - generator needed
    """

    # Get training or test split and shuffle
    if test == False:
        train_vid_ids = dataset.train_split_subject
    else:
        train_vid_ids = dataset.test_split_subject
    np.random.shuffle(train_vid_ids)

    # Yield the features and label
    for vid_id in train_vid_ids:
        x = torch.from_numpy(features.load_feature(vid_id))
        x = x.type(torch.FloatTensor)
        x = x.unsqueeze(0) # Add a fake batch dimension

        y = torch.from_numpy(np.array([vid_id % NUM_CLASSES])) #dataset.get_action(vid_id) ## TODO: implement this function

        yield x, y




def main():
    net = Net().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    all_losses = []
    print_every = 100
    for epoch in range(EPOCHS):
        start = dt.datetime.now()
        stat_dict = {"Epoch": epoch, "Loss": "0"}
        iterator = tqdm(
            iterable = enumerate(train_batch_generator()),
            total    = len(dataset.train_split_subject),
            postfix  = stat_dict
        )
        for i, (inputs, labels) in iterator:
            # Send to gpu
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            # Forward + backward pass
            optimizer.zero_grad()
            output = net(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Print stats
            if i % print_every == 0:
                all_losses.append(loss.data[0])
                stat_dict['Loss'] = "{:.5f}".format(np.mean(all_losses))
                iterator.set_postfix(stat_dict)

        # Testing
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_batch_generator(test=True)):
            inputs = Variable(inputs.cuda())
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()
            del inputs, outputs
        print('Accuracy: {:.2f}%'.format(100 * correct / total))

        np.save(all_losses, 'loss_progression_{}'.format(epoch))
        torch.save(net.state_dict(), 'torch_model_{}'.format(epoch))



def test():
    net = Net().cuda()
    net.load_state_dict(torch.load('torch_model'))




if __name__ == '__main__':
    main()
