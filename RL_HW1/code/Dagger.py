import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torchvision.transforms import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DaggerAgent:
    def __init__(self,):
        pass

    @abstractmethod
    def select_action(self, ob):
        pass


# here is an example of creating your own Agent
class ExampleAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        self.model = None

    # train your model with labeled data
    def update(self, data_batch, label_batch):
        self.model.train(data_batch, label_batch)

    # select actions by your model
    def select_action(self, data_batch):
        label_predict = self.model.predict(data_batch)
        return label_predict


class MyAgent(DaggerAgent):
    def __init__(self, necessary_parameters=None):
        super(DaggerAgent, self).__init__()
        # init your model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.num_classes = 8

        self.learning_rate = 0.001

        self.loss_fn = nn.CrossEntropyLoss()

        self.model = resnet18(num_classes=self.num_classes).to(device)

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    # train your model with labeled data
    def update(self, data_batch, label_batch):

        data_batch = np.array(data_batch)
        label_batch = np.array(label_batch)
        data_batch = torch.from_numpy(data_batch)
        label_batch = torch.from_numpy(label_batch)
        data_batch = data_batch.permute(0, 3, 1, 2)
        data_batch = data_batch.to(device)
        label_batch = label_batch.to(device)

        self.model.train()

        self.optimizer.zero_grad()
        # self.model.train(data_batch, label_batch)
        outputs = self.model(data_batch)
        loss = self.loss_fn(outputs, label_batch)
        loss.backward()  # Backpropagation
        self.optimizer.step()

    # select actions by your model
    def select_action(self, data_batch):
        label_predict = self.model.predict(data_batch)
        return label_predict
