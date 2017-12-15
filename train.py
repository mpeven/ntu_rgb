import numpy as np
from ntu_rgb import NTU
import datetime as dt
import line_profiler

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


########
# TODO
# Decrease voxel size until batch size can be > 1
# Make sure smaller voxel size still looks good
# Create validation set
# Train and test on everything

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




def get_input_features(vox_flow):
    """
    Create the input features for a single video

    Parameters
    ----------
    vox_flow : ndarray
        The voxel flow for a video, an ndarray with shape [frames,100,100,100,4]

    Returns
    -------
    torch Variable
        The list of features, each one an ndarray with shape [K*T*C,100,100,100].
        K = number of features in each video
        T = number of frames in each feature
        C = number of channels (dx, dy, dz)
    """

    # Split up video into K equal parts of size T
    frames = vox_flow.shape[0]
    skip_amount = (frames - T) / (K - 1)
    features = []
    for feature_idx in range(K):
        start = int(skip_amount * feature_idx)
        end = int(start + T)
        feature = np.vstack(vox_flow[start:end,1:,:,:,:]) # Stack frames
        features.append(feature)

    # Combine all chunks into one tensor
    stacked_feature = np.stack(features)

    return torch.from_numpy(stacked_feature)




def train_batch_generator():
    """
    Dataset can't fit in memory - generator needed
    """
    dataset = NTU()

    # Get training split
    train_vid_ids = [x for x in range(10000)]# TODO: figure out train split
    np.random.shuffle(train_vid_ids)

    for vid_id in train_vid_ids:
        vox_flow = dataset.get_voxel_flow(vid_id)
        x = get_input_features(vox_flow)
        x = x.unsqueeze(0) # Add a fake batch dimension
        y = torch.from_numpy(np.array([vid_id % NUM_CLASSES])) #dataset.get_action(vid_id) ## TODO: implement this function
        yield x.type(torch.FloatTensor), y




def main():
    net = Net().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    running_loss = 0.0
    print_every = 100
    for epoch in range(EPOCHS):
        start = dt.datetime.now()
        for i, (inputs, labels) in enumerate(train_batch_generator()):
            # Get data
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())

            # Forward + backward pass
            optimizer.zero_grad()
            output = net(inputs)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            # Print stats
            running_loss += loss.data[0]
            if (i+1) % print_every == 0:
                print('[Epoch {:02d}, Batch {:05d}] loss: {:.5f}, sec/iter: {}'.format(
                    epoch + 1, i + 1, running_loss / print_every, (dt.datetime.now() - start)/(i+1)))
                running_loss = 0.0

        # Testing
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_batch_generator()):
            inputs = Variable(inputs.cuda())
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()
            del inputs, outputs
            if i == 100:
                break
        print('Accuracy: {:.2f}%'.format(100 * correct / total))

        torch.save(net.state_dict(), 'torch_model_{}'.format(epoch))



def test():
    net = Net().cuda()
    net.load_state_dict(torch.load('torch_model'))




if __name__ == '__main__':
    main()
