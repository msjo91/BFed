import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DatasetSplit


class Party:
    def __init__(self, name, config, indices):
        self.name = name
        self.config = config
        self.indices = indices
        self.device = torch.device('cuda:{}'.format(self.config['gpu']) if torch.cuda.is_available() else 'cpu')
        self.criterion = torch.nn.NLLLoss().to(self.device)
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.forgotten = None
        self.unforgotten = None
        self.mixed = None

    def split_stg1(self, train_dataset):
        trainloader = DataLoader(DatasetSplit(train_dataset, self.indices),
                                 batch_size=self.config['local_bs'], shuffle=True)
        testloader = DataLoader(DatasetSplit(train_dataset, self.indices), batch_size=len(self.indices), shuffle=False)
        return trainloader, testloader

    def stage1(self, model, train_dataset):
        """
        Count forgetting events
        """
        print('Party {} commencing Stage 1'.format(self.name))
        trainloader, testloader = self.split_stg1(train_dataset)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])
        history = {}

        for ep in range(self.config['stg1_ep']):
            for batch_idx, (images, labels, idxs, _) in enumerate(trainloader):
                model.train()
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if ep == self.config['stg1_ep'] - 1:
                    history[batch_idx] = [images, labels, idxs.detach().cpu().numpy().astype(np.int), None]

                    model.eval()
                    for k, v in list(history.items())[-3:]:
                        outputs = model(v[0])
                        _, preds = torch.max(outputs, 1)
                        preds = preds.view(-1)
                        events = ~torch.eq(preds, v[1]).detach().cpu().numpy()
                        events = events.astype(np.int)
                        if v[3] is None:
                            v[3] = events
                        else:
                            v[3] = [a + b for a, b in zip(v[3], events)]

        for k, v in history.items():
            train_dataset.forgotten[v[2]] = v[3]

        num_forgotten = len(train_dataset.forgotten[self.indices][
                                np.where(train_dataset.forgotten[self.indices] >= self.config['event_threshold'])
                            ])
        print('Party {} Stage 1 complete: {}/{} samples forgotten'.format(self.name, num_forgotten, len(self.indices)))

    def split(self, train_dataset):
        loader = DataLoader(DatasetSplit(train_dataset, self.indices), batch_size=len(self.indices), shuffle=False)
        for _, _, _, events in loader:
            self.forgotten = self.indices[np.where(events > 0)]
            self.unforgotten = self.indices[np.where(events == 0)]
        ratio = self.config['unforgotten_ratio'] / (1 - self.config['unforgotten_ratio'])
        if len(self.unforgotten) <= int(ratio * len(self.forgotten)):
            num_items = 0
        else:
            num_items = int(ratio * len(self.forgotten))
        rnd_unf = np.random.choice(self.unforgotten, num_items, replace=False)
        self.mixed = np.concatenate([self.forgotten, rnd_unf])

    def stage2(self, model, train_dataset, r):
        """
        Train with only unforgotten samples to build initial global model
        """
        trainloader = DataLoader(DatasetSplit(train_dataset, self.unforgotten),
                                 batch_size=self.config['local_bs'], shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])

        losses = []

        model.train()

        for ep in range(self.config['stg2_ep']):
            batch_ls = []
            for batch_idx, (images, labels, _, _) in enumerate(trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_ls.append(loss.item())

                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        r + 1, ep, batch_idx * len(images),
                        len(trainloader.dataset), 100 * batch_idx / len(trainloader), loss.item()
                    ))

            loss_avg = sum(batch_ls) / len(batch_ls)
            losses.append(loss_avg)
        return model.state_dict(), losses

    def stage3(self, model, train_dataset, r):
        """
        Train with both forgotten and unforgotten samples
        """
        trainloader = DataLoader(DatasetSplit(train_dataset, self.mixed),
                                 batch_size=self.config['local_bs'], shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])

        losses = []

        model.train()

        for ep in range(self.config['stg3_ep']):
            batch_ls = []
            for batch_idx, (images, labels, _, _) in enumerate(trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_ls.append(loss.item())

                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        r + 1, ep, batch_idx * len(images),
                        len(trainloader.dataset), 100 * batch_idx / len(trainloader), loss.item()
                    ))
            loss_avg = sum(batch_ls) / len(batch_ls)
            losses.append(loss_avg)
        return model.state_dict(), losses

    def nostage(self, model, train_dataset, r):
        """
        Train without separation
        """
        trainloader = DataLoader(train_dataset, batch_size=self.config['local_bs'], shuffle=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'])

        losses = []

        model.train()

        for ep in range(self.config['stg2_ep'] + self.config['stg3_ep']):
            batch_ls = []
            for batch_idx, (images, labels, _, _) in enumerate(trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_ls.append(loss.item())

                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        r + 1, ep, batch_idx * len(images),
                        len(trainloader.dataset), 100 * batch_idx / len(trainloader), loss.item()
                    ))

            loss_avg = sum(batch_ls) / len(batch_ls)
            losses.append(loss_avg)
        return model.state_dict(), losses

    def test(self, model, testloader):
        model.eval()
        loss, correct, total = 0, 0, 0

        for images, labels in testloader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = model(images)
            batch_ls = self.criterion(outputs, labels)
            loss += batch_ls.item()

            _, preds = torch.max(outputs, 1)
            preds = preds.view(-1)
            correct += torch.sum(torch.eq(preds, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss
