import copy

import torch
from tqdm import tqdm

from utils import average_weights


def test(global_model, config, testloader):
    device = torch.device('cuda:{}'.format(config['gpu']) if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.NLLLoss().to(device)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    loss, correct, total = 0, 0, 0

    global_model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = global_model(images)
        batch_ls = criterion(outputs, labels)
        loss += batch_ls.item()

        _, preds = torch.max(outputs, 1)
        preds = preds.view(-1)
        correct += torch.sum(torch.eq(preds, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss


def train(global_model, config, train_dataset, parties, testloader=None, test_every=0):
    # copy weights
    global_weights = global_model.state_dict()
    losses = []

    if config['staging'] == 1:
        for p in parties:
            p.stage1(model=copy.deepcopy(global_model), train_dataset=train_dataset)
            p.split(train_dataset)

        print('Commencing Federated Learning')

        for r in tqdm(range(config['rounds'])):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {r + 1} |\n')

            global_model.train()

            for p in parties:
                if r < config['stg_shift']:
                    w, ls = p.stage2(model=copy.deepcopy(global_model), train_dataset=train_dataset, r=r)
                else:
                    w, ls = p.stage3(model=copy.deepcopy(global_model), train_dataset=train_dataset, r=r)
                local_weights.append(copy.deepcopy(w))
                # local_losses.append(copy.deepcopy(ls))

            # Update global weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)

            # loss_avg = sum(local_losses) / len(local_losses)
            # losses.append(loss_avg)

            if test_every is not False:
                if (r + 1) % test_every == 0:
                    test_acc, test_ls = test(global_model, config, testloader)
                    print(f'\n Results after {r + 1} global rounds of training:')
                    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    elif config['staging'] == 0:
        for r in tqdm(range(config['rounds'])):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {r + 1} |\n')

            global_model.train()

            for p in parties:
                w, ls = p.nostage(model=copy.deepcopy(global_model), train_dataset=train_dataset, r=r)
            local_weights.append(copy.deepcopy(w))
            # local_losses.append(copy.deepcopy(ls))

            # Update global weights
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)

            # loss_avg = sum(local_losses) / len(local_losses)
            # losses.append(loss_avg)

            if test_every is True:
                if (r + 1) % test_every == 0:
                    test_acc, test_ls = test(global_model, config, testloader)
                    print(f'\n Results after {r + 1} global rounds of training:')
                    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    return global_model, losses
