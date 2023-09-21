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


class MyAgent1(DaggerAgent):
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

        self.num_classes = 18

        self.learning_rate = 0.001

        self.loss = nn.CrossEntropyLoss()

        self.model = resnet18(num_classes=self.num_classes).to(device)

        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    # train your model with labeled data
    def update(self, data_batch, label_batch):
        self.optimizer.zero_grad()
        self.model.train(data_batch, label_batch)

    # select actions by your model
    def select_action(self, data_batch):
        label_predict = self.model.predict(data_batch)
        return label_predict
