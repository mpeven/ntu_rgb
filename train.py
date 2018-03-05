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
from models import Model_1, Model_2, Model_3, Model_4, Model_5



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





def run_experiment(experiment_id):
    '''
    01 - 2D spatial CS
    02 - 2D spatial CV
    03 - 3D spatial CS
    04 - 3D spatial CV
    05 - 3D temporal CS
    06 - 3D temporal CV
    07 - 3D temporal (no augmentation) CS
    08 - 3D temporal (no augmentation) CV
    09 - 2D temporal CS
    10 - 2D temporal CV
    ------ If time:
    11 - 2-stream concatenate lstm output
    12 - 2-stream svm classifier
    '''
    print("Running experiment {:02}".format(experiment_id))

    num_epochs = 50
    hidden_dimension_size = 256
    lstm_dropout = 0

    # Init parameters
    images = False
    images_3D = False
    op_flow = False
    op_flow_2D = False

    # Set experiment parameters
    if experiment_id == 1 or experiment_id == 2:
        batch_size = 32
        images = True
        net = Model_1(hidden_dimension_size, lstm_dropout).cuda()
        cross_view = False if experiment_id == 1 else True
    if experiment_id == 3 or experiment_id == 4:
        batch_size = 32
        images_3D = True
        net = Model_4(hidden_dimension_size, lstm_dropout).cuda()
        cross_view = False if experiment_id == 3 else True
    if experiment_id == 5 or experiment_id == 6:
        batch_size = 8
        op_flow = True
        net = Model_2(hidden_dimension_size, lstm_dropout).cuda()
        cross_view = False if experiment_id == 5 else True
    if experiment_id == 9 or experiment_id == 10:
        batch_size = 16
        op_flow_2D = True
        net = Model_5(hidden_dimension_size, lstm_dropout).cuda()
        cross_view = False if experiment_id == 5 else True


    # Get dataloaders
    train_loader = get_train_loader(batch_size, images=images, images_3D=images_3D,
                                    op_flow=op_flow, op_flow_2D=op_flow_2D,
                                    cross_view=cross_view)
    test_loader = get_test_loader(batch_size, images=images, images_3D=images_3D,
                                  op_flow=op_flow, op_flow_2D=op_flow_2D,
                                  cross_view=cross_view)

    # Set up optimizer with auto-adjusting learning rate
    parameters = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train
    for epoch in range(num_epochs):
        scheduler.step()
        training_epoch(net, optimizer, epoch, train_loader)

        # valid_acc = test_epoch(net, valid_loader, desc="Validation")
        # print('Epoch {:02} top-1 validation accuracy: {:.1f}%'.format(epoch, valid_acc))

    # Save results
    model_file = 'torch_models/torch_model_experiment_{:02}'.format(experiment_id)
    torch.save(net.state_dict(), model_file)


    # Test
    # net.load_state_dict(torch.load('torch_models/torch_model_experiment_{:02}'.format(experiment_id)))
    test_acc = test_epoch(net, test_loader, desc="Testing")
    print('Experiment {:02} test-set accuracy: {:.2f}%'.format(experiment_id, test_acc))


if __name__ == '__main__':
    # main()
    run_experiment(int(sys.argv[1]))
