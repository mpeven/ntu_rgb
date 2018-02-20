'''
Experiments to run:

- Check accuracy on cross-view split
- Deeper network
- Fully connected layer at the end
- Random rotations

- Find another dataset?

'''

import numpy as np
import datetime as dt
import line_profiler
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from models import Model_1_2 as Net
from datasets import NTURGBDataset



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




def test(num_epochs, batch_size, hidden_dimension_size, lstm_dropout):
    net = Net(hidden_dimension_size, lstm_dropout).cuda()
    net.eval()
    net.load_state_dict(torch.load('torch_models/torch_model_{}'.format(epoch)))
    test_loader = get_test_loader(batch_size)
    correct = 0
    total = 0
    stat_dict = {"Epoch": "{:02}".format(epoch), "Accuracy": "0"}
    iterator = tqdm(test_loader, postfix=stat_dict, ncols=100)
    for i, test_data in enumerate(iterator):
        inputs = Variable(test_data[0].cuda(async=True))
        labels = Variable(test_data[1].cuda(async=True))
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.type(torch.LongTensor) == test_data[1].type(torch.LongTensor)).sum()
        stat_dict['Accuracy'] = "{:.4f}".format(100 * correct / total)
        iterator.set_postfix(stat_dict)
        del inputs, outputs


def train(num_epochs, batch_size, hidden_dimension_size, lstm_dropout):
    net = Net(hidden_dimension_size, lstm_dropout).cuda()
    loss_func = nn.CrossEntropyLoss()

    # Set up optimizer with auto-adjusting learning rate
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

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
        iterator = tqdm(train_loader, postfix=stat_dict, ncols=100)
        for i, train_data in enumerate(iterator):
            # Send to gpu
            inputs = Variable(train_data[0].cuda(async=True))
            labels = Variable(train_data[1].cuda(async=True))

            # Forward + backward pass
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update loss and accuracy
            if (i+1)%10 == 0:
                _, predicted = torch.max(outputs.data, 1)
                total += train_data[1].size(0)
                correct += (predicted.type(torch.LongTensor) == train_data[1].type(torch.LongTensor)).sum()
                all_losses.append(loss.data[0])
                stat_dict['Loss'] = "{:.5f}".format(np.mean(all_losses))
                stat_dict['Acc'] = "{:.4f}".format(100 * correct / total)
                iterator.set_postfix(stat_dict)

        ##########
        # Validation
        net.eval()
        val_correct = 0
        val_total = 0
        for i, valid_data in enumerate(valid_loader):
            inputs = Variable(valid_data[0].cuda(async=True))
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += valid_data[1].size(0)
            val_correct += (predicted.type(torch.LongTensor) == valid_data[1].type(torch.LongTensor)).sum()
            del outputs
        print('Epoch {:02} top-1 validation accuracy: {:.1f}%'.format(epoch, 100 * val_correct / val_total))

        ##########
        # Saving results
        # np.save('torch_models/loss_progression_{}'.format(epoch), all_losses)
        # torch.save(net.state_dict(), 'torch_models/torch_model_{}'.format(epoch))

        ##########
        # Early stopping
        if (100 * correct / total) > 99.8:
            break

    return epoch



def main():
    import sys
    from itertools import product
    combos = list(product([1024, 512,256,128,64,32,16,8,4,2],[0, .5, .9, .95, .99]))
    h,d = combos[int(sys.argv[1])]
    print("Hidden size: {}, Dropout: {}".format(h,d))
    train(batch_size=32, num_epochs=50, hidden_dimension_size=h, lstm_dropout=d)
    # test(batch_size=32, num_epochs=last_epoch, hidden_dimension_size=h, lstm_dropout=d)


if __name__ == '__main__':
    main()
