from inceptionmod import *
from datagenerator import DatasetGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn
import torch.optim as optim
import time

X_train = []
y_train = []
X_val = []
y_val = []
X_test = []
y_test = []

labels = [i for i in range(952)]

# Dataset for train
for label in labels:
    label_path = os.path.join(train_path, str(label))
    for image in os.listdir(label_path):
        img_path = os.path.join(label_path, image)
        img = cv2.imread(img_path)
        X_train.append(img)
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Dataset for val
for label in labels:
    label_path = os.path.join(val_path, str(label))
    for image in os.listdir(label_path):
        img_path = os.path.join(label_path, image)
        img = cv2.imread(img_path)
        X_val.append(img)
        y_val.append(label)

X_val = np.array(X_val)
y_val = np.array(y_val)

# Dataset for test
for label in labels:
    label_path = os.path.join(test_path, str(label))
    for image in os.listdir(label_path):
        img_path = os.path.join(label_path, image)
        img = cv2.imread(img_path)
        X_test.append(img)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Dataset loader
trainloader = torch.utils.data.DataLoader(DatasetGenerator(X_train, y_train), batch_size=256,
                                          shuffle=True)

valloader = torch.utils.data.DataLoader(DatasetGenerator(X_val, y_val), batch_size=64,
                                         shuffle=False)

testloader = torch.utils.data.DataLoader(DatasetGenerator(X_test, y_test), batch_size=32,
                                         shuffle=False)

# Loss history
loss_train = []
loss_val = []

# Model training
model = googlenetmod()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
                        
def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (torch.argmax(output, axis=1)==label).float().mean()

for epoch in range(5):  # loop over the dataset multiple times
    total_loss = 0.0
    total_val_loss = 0.0
    tic = time.time()
    tic_step = time.time()
    train_acc = 0.0
    valid_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        train_acc += acc(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # loss update
        total_loss += loss.item()
        
    # calculate validation accuracy
    for i, data in enumerate(valloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        valid_acc += acc(outputs, labels)
        total_val_loss += criterion(outputs, labels).item()

    print("Epoch %d: loss %.3f, train acc %.3f, valid acc %.3f, in %.1f sec" % (
            epoch, total_loss/len(trainloader), train_acc/len(trainloader),
            valid_acc / len(valloader), time.time()-tic))
    
    loss_train.append(total_loss)
    loss_val.append(total_val_loss)
    
print('Finished Training')

# Model evaluation
model.eval()

test_acc = 0.0
test_loss = 0.0

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    test_acc += acc(outputs, labels)
    test_loss += criterion(outputs, labels).item()

print(f"Test accuracy: {test_acc / len(testloader)}")
print(f"Test loss: {test_loss / len(testloader)}")