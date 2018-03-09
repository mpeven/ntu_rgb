from __future__ import division
import sys
from itertools import product
import cv2
import numpy as np
import datetime as dt
import line_profiler
from tqdm import tqdm
tqdm.monitor_interval = 0

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from datasets import get_train_valid_loader, get_test_loader, get_train_loader

from config import *


def get_accuracy(outputs, labels):
    ''' Calculate the classification accuracy '''
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted.type(torch.LongTensor) == labels.type(torch.LongTensor)).sum()
    return correct / total





def test_epoch(net, test_loader, desc):
    ''' Validation or test epoch '''
    # Turn off dropout, batch norm, etc..
    net.eval()

    # Data to save
    to_save_output = []
    to_save_labels = []

    # Single pass through testation data
    accuracies = []
    iterator = tqdm(test_loader, desc=desc, ncols=100, leave=False)
    for i, test_data in enumerate(iterator):
        # Get input data and labels
        if len(test_data) == 3:
            inputs = (Variable(test_data[0], volatile=True).cuda(),
                      Variable(test_data[1], volatile=True).cuda())
        else:
            inputs = Variable(test_data[0], volatile=True).cuda()
        labels = Variable(test_data[-1], volatile=True).cuda()

        # Forward pass
        outputs = net(inputs)

        # Save data
        if desc == "Testing":
            to_save_output.append(outputs.data.cpu().numpy().copy())
            to_save_labels.append(test_data[-1].numpy().copy())

        # Calculate accuracy
        accuracies.append(get_accuracy(outputs, test_data[-1]))
        iterator.set_postfix({"Accuracy": "{:.4f}".format(100 * np.mean(accuracies))})

    # Save data
    if desc == "Testing":
        all_output = np.concatenate(to_save_output)
        all_labels = np.concatenate(to_save_labels)
        np.save('_output_spatial_{:.4f}'.format(100 * np.mean(accuracies)), all_output)
        np.save('_labels_spatial_{:.4f}'.format(100 * np.mean(accuracies)), all_labels)

    return 100 * np.mean(accuracies)





def training_epoch(net, optimizer, epoch, train_loader):
    ''' Training epoch '''
    # Set the network to training mode
    net.train()
    loss_func = nn.CrossEntropyLoss().cuda()

    # Single pass through training data
    accuracies = []
    losses = []
    stat_dict = {"Epoch": epoch}
    iterator = tqdm(train_loader, postfix=stat_dict, ncols=100)
    for i, train_data in enumerate(iterator):
        # Get input data and labels
        if len(train_data) == 3:
            inputs = (Variable(train_data[0].cuda(async=True)),
                      Variable(train_data[1].cuda(async=True)))
        else:
            inputs = Variable(train_data[0].cuda(async=True))
        labels = Variable(train_data[-1].cuda(async=True))

        # Forward pass, calculate loss, backward pass
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update loss and accuracy
        if (i+1)%10 == 0:
            accuracies.append(get_accuracy(outputs, train_data[-1]))
            losses.append(loss.data[0])
            stat_dict['Loss'] = "{:.5f}".format(np.mean(losses))
            stat_dict['Acc'] = "{:.4f}".format(100 * np.mean(accuracies))
            iterator.set_postfix(stat_dict)

    # Return the training accuracy
    return 100 * np.mean(accuracies)





def main():
    '''
    Finished - 1, 2, 5, 7, 8, 9, 10
    Running -
    TODO - 3, 4, 6
    01, 02 - 2D spatial (images)
    03, 04 - 3D geometric (3D images)
    05, 06 - 3D temporal (3D optical flow)
    07, 08 - 3D temporal (3D optical flow - no augmentation)
    09, 10 - 2D temporal (2D optical flow)

    ------ If time:
    - 2-stream concatenate lstm output
    - 2-stream svm classifier
    '''
    print_config()

    # Get network
    net = torch.nn.DataParallel(NEURAL_NET).cuda()

    # Get dataloaders
    train_loader = get_train_loader()
    test_loader = get_test_loader()

    # Set up optimizer with auto-adjusting learning rate
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train
    for epoch in range(NUM_EPOCHS):
        scheduler.step()
        train_acc = training_epoch(net, optimizer, epoch, train_loader)

        # valid_acc = test_epoch(net, valid_loader, desc="Validation")
        # print('Epoch {:02} top-1 validation accuracy: {:.1f}%'.format(epoch, valid_acc))

        # Checkpoint results
        model_file = 'torch_models/torch_model_experiment_{:02}_epoch_{:02}'.format(EXPERIMENT_NUM, epoch)
        torch.save(net.state_dict(), model_file)

        # Early stopping
        if train_acc > 99.9:
            break

    # Save results
    model_file = 'torch_models/torch_model_experiment_{:02}'.format(EXPERIMENT_NUM)
    torch.save(net.state_dict(), model_file)

    # Test
    # net.load_state_dict(torch.load('torch_models/torch_model_experiment_{:02}'.format(EXPERIMENT_NUM)))
    test_acc = test_epoch(net, test_loader, desc="Testing")
    print('Experiment {:02} test-set accuracy: {:.2f}%'.format(EXPERIMENT_NUM, test_acc))


if __name__ == '__main__':
    main()
