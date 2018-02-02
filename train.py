'''
Experiments to run:

- Soft max on the 5 features
- Adam optimizer
- More overfit prevention
- Check accuracy on cross-view split
- Deeper network
- Fully connected layer at the end
- Random rotations
- Smaller/Larger voxel grid
- Learning-rate reduction every few epochs
- BatchNorm, Dropout, BatchNorm + Dropout
- RNN/LSTM cell as the temporal layer

- Find another dataset?

'''


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
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler



####################
# Constants

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
    def __init__(self):
        super(Net, self).__init__()
        # C - num channels (3 - dx,dy,dz)
        # T - num frames per feature (10)
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
        all_x = []
        # K - num features per video (5)
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
        return self.preds(x)






class NTURGBDataset(Dataset):
    """ NTURGB+D Dataset """

    def __init__(self, test=False, validation=False, full_train=False):
        """ test should be True if testing set """
        if test:
            self.vid_ids = dataset.test_split_subject
        elif validation:
            self.vid_ids = dataset.validation_split_subject
        elif full_train:
            self.vid_ids = dataset.train_split_subject
        else:
            self.vid_ids = dataset.train_split_subject_with_validation

    def __len__(self):
        return len(self.vid_ids)

    def __getitem__(self, idx):
        vid_id = self.vid_ids[idx]
        x = torch.from_numpy(features.load_feature(vid_id))
        x = x.type(torch.FloatTensor)
        y = dataset.id_to_action[vid_id]
        return {"Features": x, "Label": y}




def get_train_valid_loader(batch_size):
    # Create the dataset
    train_dataset = NTURGBDataset()
    valid_dataset = NTURGBDataset(validation=True)

    # Seed the shuffler
    np.random.seed(149)
    torch.manual_seed(149)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                    batch_size=batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                    batch_size=batch_size, shuffle=True,
                    num_workers=4, pin_memory=True)

    return (train_loader, valid_loader)



def get_test_loader(batch_size):
    test_dataset = NTURGBDataset(test=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                    batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=True)
    return test_loader




def test(batch_size):
    net = Net().cuda()
    net.eval()
    net.load_state_dict(torch.load('torch_model_19'))
    test_loader = get_test_loader(batch_size)
    correct = 0
    total = 0
    stat_dict = {"Accuracy": "0"}
    iterator = tqdm(test_loader, postfix=stat_dict)
    for i, test_data in enumerate(iterator):
        inputs = Variable(test_data['Features'].cuda(async=True))
        labels = Variable(test_data['Label'].cuda(async=True))
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.type(torch.LongTensor) == test_data['Label'].type(torch.LongTensor)).sum()
        stat_dict['Accuracy'] = "{:.4f}".format(100 * correct / total)
        iterator.set_postfix(stat_dict)
        del inputs, outputs
    print('Final Accuracy: {:.2f}%'.format(100 * correct / total))



def train(num_epochs, batch_size):
    net = Net().cuda()
    loss_func = nn.CrossEntropyLoss()

    # Set up optimizer with auto-adjusting learning rate
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Get the data loaders
    train_loader, valid_loader = get_train_valid_loader(batch_size)

    for epoch in range(num_epochs):
        # Update learning rate
        scheduler.step()

        ##########
        # Training
        net.train()
        correct = 0
        total=0
        all_losses = []
        stat_dict = {"Epoch": epoch, "Loss": "0", "Acc": "0"}
        iterator = tqdm(train_loader, postfix=stat_dict)
        for i, train_data in enumerate(iterator):
            # Send to gpu
            inputs = Variable(train_data['Features'].cuda(async=True))
            labels = Variable(train_data['Label'].cuda(async=True))

            # Forward + backward pass
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            if i%10 == 0:
                _, predicted = torch.max(outputs.data, 1)
                total += train_data['Label'].size(0)
                correct += (predicted.type(torch.LongTensor) == train_data['Label'].type(torch.LongTensor)).sum()
                all_losses.append(loss.data[0])
                stat_dict['Loss'] = "{:.5f}".format(np.mean(all_losses))
                stat_dict['Acc'] = "{:.4f}".format(100 * correct / total)
                iterator.set_postfix(stat_dict)

        ##########
        # Validation
        net.eval()
        correct = 0
        total = 0
        stat_dict = {"Epoch": epoch, "Accuracy": "0"}
        iterator = tqdm(valid_loader, postfix=stat_dict)
        for i, valid_data in enumerate(iterator):
            inputs = Variable(valid_data['Features'].cuda(async=True))
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += valid_data['Label'].size(0)
            correct += (predicted.type(torch.LongTensor) == valid_data['Label'].type(torch.LongTensor)).sum()
            stat_dict['Accuracy'] = "{:.4f}".format(100 * correct / total)
            iterator.set_postfix(stat_dict)
            del outputs
        print('Epoch {:02} top-1 validation accuracy: {:.1f}%'.format(epoch, 100 * correct / total))

        ##########
        # Saving results
        np.save('loss_progression_{}'.format(epoch), all_losses)
        torch.save(net.state_dict(), 'torch_model_{}'.format(epoch))




def main():
    train(batch_size=8, num_epochs=20)
    test(batch_size=8)


if __name__ == '__main__':
    main()
