import copy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from update import LocalUpdate, test_inference
from utils import average_weights, concat_dataset


def federated(global_model, opts, train_dataset, test_dataset, noised_dataset, user_groups, logger, print_every=2):
    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []

    plt_avg_train_acc = []
    plt_avg_test_acc = []
    plt_avg_test_ls = []

    for epoch in tqdm(range(opts['epochs'])):
        client_vectors = []

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(opts['frac'] * opts['num_users']), 1)
        idxs_users = np.random.choice(range(opts['num_users']), m, replace=False)

        if opts['noise_frac'] == 0:
            for idx in idxs_users:
                local_model = LocalUpdate(opts=opts, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
                lm, w, loss, params = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

                _, _, model_vector = test_inference(opts, lm, test_dataset)
                client_vectors.append(model_vector.tolist())
        else:
            # Get parameters
            local_params = []

            # Split users
            split_idx = int(len(idxs_users) * (1 - opts['noise_frac']))

            for idx in idxs_users[:split_idx]:
                local_model = LocalUpdate(opts=opts, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
                lm, w, loss, params = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                local_params.append(copy.deepcopy(params))
                _, _, model_vector = test_inference(opts, lm, test_dataset)
                client_vectors.append(model_vector.tolist())

            for i in idxs_users[split_idx:]:
                local_model = LocalUpdate(opts=opts, dataset=noised_dataset, idxs=user_groups[i], logger=logger)
                lm, w, loss, params = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                local_params.append(copy.deepcopy(params))
                _, _, model_vector = test_inference(opts, lm, test_dataset)
                client_vectors.append(model_vector.tolist())

        if opts['clean']:
            client_vectors = np.array(client_vectors)
            mean_vector = client_vectors.mean(axis=0)
            if opts['dist'] == 'euclidean':
                distance = np.sqrt(np.power(client_vectors - mean_vector, 2).sum(axis=1))
            elif opts['dist'] == 'mahalanobis':
                distance = []
                for cv in client_vectors:
                    E = cv - mean_vector
                    V = np.vstack([cv, mean_vector])
                    V = np.cov(V.T)
                    VI = np.linalg.inv(V)
                    D = np.sqrt(np.sum(np.dot(E, VI) * E, axis=1))
                    distance.append(D)
            far_idxs = np.argsort(distance)[-opts['clean']:]
            local_weights = [local_weights[i] for i, user in enumerate(idxs_users) if user not in far_idxs]

        # Update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()

        for c in range(opts['num_users']):
            local_model = LocalUpdate(opts=opts, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)

        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        plt_avg_train_acc.append(100 * train_accuracy[-1])

        # Test inference after completion of training
        test_acc, test_loss, _ = test_inference(opts, global_model, test_dataset)
        plt_avg_test_acc.append(test_acc)
        plt_avg_test_ls.append(test_loss)

    print(f' \n Results after {opts["epochs"]} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print(">> Number of users: ", len(idxs_users))
    if opts['clean']:
        print(">> Deleted users: ", far_idxs)

    return train_accuracy, train_loss, plt_avg_test_acc, plt_avg_test_ls


def solo(model, opts, train_dataset, test_dataset, noised_dataset):
    device = torch.device('cuda:{}'.format(opts['gpu']) if torch.cuda.is_available() else 'cpu')

    if opts['noise_frac']:
        train_dataset = concat_dataset(opts, train_dataset, noised_dataset)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device)

    # Set optimizer for the local updates
    if opts['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opts['lr'], momentum=opts['momentum'])
    elif opts['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opts['lr'], weight_decay=1e-4)

    epoch_loss = []

    for epoch in tqdm(range(opts['epochs'])):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # testing
    test_acc, test_loss, _ = test_inference(opts, model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100 * test_acc))

    return test_acc, test_loss
