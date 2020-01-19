import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm#just for progressbars!

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    device = torch.device("cpu")
    print("Running on the CPU.")
    
DATADIR = "Machine_Learning"
CATEGORIES = ["No-Poleron","Poleron"]
DATASIZE = 12800
NEIGHBOURS = 8
MAX_DISTANCE = 3.5

REBUILD_DATA = False
VAL_PCT = 0.2 #percentage of validation DATA
BATCH_SIZE = 100
EPOCHS = 10
  
training_data = []
X = []
Y = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        path += '.dat'
        class_num = CATEGORIES.index(category)
        with open(path, "r") as dataset:
            for line in dataset:
                new_array = [float(distance.strip()) for distance in line.split()]
                training_data.append([np.array(new_array), np.eye(2)[class_num]])
    
    np.random.shuffle(training_data)
    np.save("training_data.npy", training_data)

if REBUILD_DATA:            
    create_training_data()
else:
    training_data = np.load("training_data.npy", allow_pickle=True)

#print(len(training_data))

class Net(nn.Module): #fully connected layers
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NEIGHBOURS,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,16)
        self.fc4 = nn.Linear(16,2)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)
   
net = Net().to(device)

X = torch.Tensor([i[0] for i in training_data]).view(-1,8)
X = X/MAX_DISTANCE
Y = torch.Tensor([i[1] for i in training_data])

val_size = int(len(X)*VAL_PCT)
#print(val_size)

train_X = X[:-val_size]
train_Y = Y[:-val_size]

test_X = X[-val_size:]
test_Y = Y[-val_size:]

#print(len(train_X))
#print(len(test_X))

def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,8)
            batch_Y = train_Y[i:i+BATCH_SIZE]
            
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            net.zero_grad()
            
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch: {epoch}. Loss: {loss}")

train(net)

def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_Y[i]).to(device)
            net_out = net(test_X[i].view(-1,8).to(device))[0]
            predicted_class = torch.argmax(net_out)
            
            if predicted_class == real_class:
                correct += 1
            total += 1
            
    print("Accuracy:",round(correct/total,3))
    
test(net)