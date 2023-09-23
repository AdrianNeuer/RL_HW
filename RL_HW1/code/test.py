from torchvision.models import resnet18
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

data_batch = np.random.randn(250, 210, 160, 3)
data_batch = data_batch.astype(np.float32)
label_batch = np.random.randint(0, 8, (250,))
# print(data_batch.shape, label_batch.shape)

data_batch = torch.from_numpy(data_batch)
label_batch = torch.from_numpy(label_batch)
data_batch = data_batch.permute(0, 3, 1, 2)
label_batch = label_batch.long()  # can be commented out
data_batch = data_batch.to(device)
label_batch = label_batch.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

num_classes = 8

learning_rate = 0.001

loss_fn = nn.CrossEntropyLoss()

model = resnet18(num_classes=8).to(device)

optimizer = optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.9)

model.train()
optimizer.zero_grad()  # Zero the gradient buffers
outputs = model(data_batch)

# Calculate the loss
loss = loss_fn(outputs, label_batch)
loss.backward()  # Backpropagation
optimizer.step()

pred = model(data_batch)
