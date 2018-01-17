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
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler



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
        self.convlayer1 = nn.Sequential(
            nn.Conv3d(C * T, 96, kernel_size=3, padding=1),
            nn.BatchNorm3d(96),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.convlayer2 = nn.Sequential(
            nn.Conv3d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.convlayer3 = nn.Sequential(
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.convlayer4 = nn.Sequential(
            nn.Conv3d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(3))
        self.timelayer1 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=0),
            nn.ReLU())
        self.fc1 = nn.Linear(512, NUM_CLASSES)

    # TODO: Add dropout
    def forward(self, X):
        all_x = []
        for k in range(K):
            x = self.convlayer1(X[:,k])
            x = self.convlayer2(x)
            x = self.convlayer3(x)
            x = self.convlayer4(x)
            x = x.view(-1, int(np.prod(x.size()[1:]))) # Flatten
            all_x.append(x)
        x = torch.stack(all_x, 2)
        x = self.timelayer1(x)
        x = self.timelayer1(x)
        x = x.view(-1, int(np.prod(x.size()[1:]))) # Flatten
        x = self.fc1(x)
        return x



class NTURGBDataset(TensorDataset):
    """ NTURGB+D Dataset """

    def __init__(self, test=False):
        """ test should be True if testing set """
        self.vid_ids = dataset.test_split_subject if test else dataset.train_split_subject

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        x = torch.from_numpy(features.load_feature(vid_id))
        x = x.type(torch.FloatTensor)
        y = vid_id % NUM_CLASSES #dataset.get_action(vid_id) ## TODO: implement this function
        return {"Features": x, "Label": y}




def get_train_valid_loader(batch_size, valid_size=0.1):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # Create the dataset
    train_dataset = NTURGBDataset()
    valid_dataset = NTURGBDataset()

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # Shuffle the dataset
    np.random.shuffle(indices)

    # Get the train and validation splits
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, sampler=train_sampler,
                    num_workers=4)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, sampler=valid_sampler,
                    num_workers=4)

    return (train_loader, valid_loader)




def test():
    net = Net().cuda()
    net.load_state_dict(torch.load('torch_model'))
    correct = 0
    total = 0
    stat_dict = {"Epoch": epoch, "Accuracy": "0"}
    iterator = tqdm(
        iterable = enumerate(train_batch_generator(test=True)),
        total    = len(dataset.test_split_subject),
        postfix  = stat_dict
    )
    for i, (inputs, labels) in iterator:
        inputs = Variable(inputs.cuda())
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()
        stat_dict['Accuracy'] = "{:.4f}".format(100 * correct / total)
        iterator.set_postfix(stat_dict)
        del inputs, outputs
    print('Final Accuracy: {:.2f}%'.format(100 * correct / total))



def train(num_epochs, batch_size):
    net = Net().cuda()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loader, valid_loader = get_train_valid_loader(batch_size)

    for epoch in range(num_epochs):
        ##########
        # Training
        correct = 0
        total=0
        all_losses = []
        stat_dict = {"Epoch": epoch, "Loss": "0", "Acc": "0"}
        iterator = tqdm(train_loader, postfix=stat_dict)
        for i, train_data in enumerate(iterator):
            # Send to gpu
            inputs = Variable(train_data['Features'].cuda())
            labels = Variable(train_data['Label'].cuda())

            # Forward + backward pass
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += train_data['Label'].size(0)
            correct += (predicted.type(torch.LongTensor) == train_data['Label'].type(torch.LongTensor)).sum()
            all_losses.append(loss.data[0])
            stat_dict['Loss'] = "{:.5f}".format(np.mean(all_losses))
            stat_dict['Acc'] = "{:.4f}".format(100 * correct / total)
            iterator.set_postfix(stat_dict)

        ##########
        # Validation
        correct = 0
        total = 0
        stat_dict = {"Epoch": epoch, "Accuracy": "0"}
        iterator = tqdm(valid_loader, postfix=stat_dict)
        for i, valid_data in enumerate(iterator):
            inputs = Variable(valid_data['Features'].cuda())
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += valid_data['Label'].size(0)
            correct += (predicted.type(torch.LongTensor) == valid_data['Label'].type(torch.LongTensor)).sum()
            stat_dict['Accuracy'] = "{:.4f}".format(100 * correct / total)
            iterator.set_postfix(stat_dict)
            del outputs
        print('Final Accuracy: {:.2f}%'.format(100 * correct / total))

        ##########
        # Saving results
        np.save('loss_progression_{}'.format(epoch), all_losses)
        torch.save(net.state_dict(), 'torch_model_{}'.format(epoch))




def main():
    train(batch_size=2, num_epochs=10)
    test()


if __name__ == '__main__':
    main()
