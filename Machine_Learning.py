import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm #just for progressbars!
import matplotlib.pyplot as plt
import time

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    device = torch.device("cpu")
    print("Running on the CPU.")
    
TRAINING_ON_NEAR_POLARONS = True
    
DATADIR = "Machine_Learning"
TRAINDIR = "TRAINING_DATA"
TESTDIR = "TESTING_DATA"


    
TRAINSIZE = 9100
TESTSIZE = 1860
NEIGHBOURS = 2
MAX_DISTANCE = 3.5

REBUILD_DATA = True
BATCH_SIZE = 100
EPOCH = 1
EPOCHS = 101
if TRAINING_ON_NEAR_POLARONS:
    OUTPUTSIZE = 3
else:
    OUTPUTSIZE = 2
    
if TRAINING_ON_NEAR_POLARONS:
    CATEGORIES = ["No-Polaron","Near-Polaron","Polaron"]
    MATRIX_FOLDER = f"Confusion-Matrix_{round(time.time(),0)}"
    PNGNAME = f"Polaron-Near_Polaron-No_Polaron_{round(time.time(),0)}"
    label_predict = ["Predicted: No Polaron","Predicted: Near Polaron","Predicted: Polaron"]
    label_Real = ["Actual: No Polaron","Actual: Near Polaron","Actual: Polaron"]
else:
    CATEGORIES = ["No-Polaron","Polaron"]
    MATRIX_FOLDER = f"Confusion-Matrix_{round(time.time(),0)}"
    PNGNAME = f"Polaron-No_Polaron_{round(time.time(),0)}.png"
    label_predict = ["Predicted: No Polaron","Predicted: Polaron"]
    label_Real = ["Actual: No Polaron","Actual: Polaron"]
    
training_data = []
testing_data = []
train_X = []
train_Y = []
test_X = []
test_Y = []

train_accuracies = []
accuracies = []
timestamp = []




confusion_matrix = torch.zeros(OUTPUTSIZE,OUTPUTSIZE)

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, TRAINDIR, category)
        path += '.dat'
        class_num = CATEGORIES.index(category)
        with open(path, "r") as dataset:
            for line in dataset:
                new_array = [float(distance.strip()) for distance in line.split()]
                oxygen_1array = new_array[0:4]
                oxygen_2array = new_array[4:6]
                titanium_array = new_array[6:]
                if NEIGHBOURS == 2:
                    training_array = oxygen_1array[0:2]
                elif NEIGHBOURS == 4:
                    training_array = oxygen_1array
                elif NEIGHBOURS == 6:
                    training_array = oxygen_1array + oxygen_2array
                elif NEIGHBOURS == 8:
                    training_array = oxygen_1array + oxygen_2array + titanium_array
                training_data.append([np.array(training_array), np.eye(OUTPUTSIZE)[class_num]])
    
    np.random.shuffle(training_data)
    np.save("training_data.npy", training_data)
    
def create_testing_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, TESTDIR, category)
        path += '.dat'
        class_num = CATEGORIES.index(category)
        with open(path, "r") as dataset:
            for line in dataset:
                new_array = [float(distance.strip()) for distance in line.split()]
                oxygen_1array = new_array[0:4]
                oxygen_2array = new_array[4:6]
                titanium_array = new_array[6:]
                if NEIGHBOURS == 2:
                    testing_array = oxygen_1array[0:2]
                elif NEIGHBOURS == 4:
                    testing_array = oxygen_1array
                elif NEIGHBOURS == 6:
                    testing_array = oxygen_1array + oxygen_2array
                elif NEIGHBOURS == 8:
                    testing_array = oxygen_1array + oxygen_2array + titanium_array
                testing_data.append([np.array(testing_array), np.eye(OUTPUTSIZE)[class_num]])
    
    np.random.shuffle(testing_data)
    np.save("testing_data.npy", testing_data)

if REBUILD_DATA:            
    create_training_data()
    create_testing_data()
else:
    training_data = np.load("training_data.npy", allow_pickle=True)
    testing_data = np.load("testing_data.npy", allow_pickle=True)

#print(len(training_data))
#print(len(testing_data))

class Net(nn.Module): #fully connected layers
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NEIGHBOURS,16) 
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,16)
        self.fc4 = nn.Linear(16,OUTPUTSIZE)
        '''
        NN for overfitting
        self.fc1 = nn.Linear(NEIGHBOURS,128) 
        self.fc2 = nn.Linear(128,512)
        self.fc3 = nn.Linear(512,128)
        self.fc4 = nn.Linear(128,OUTPUTSIZE)
        
        '''
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net().to(device)

train_X = torch.Tensor([i[0] for i in training_data]).view(-1,NEIGHBOURS)
train_X = train_X/MAX_DISTANCE
train_Y = torch.Tensor([i[1] for i in training_data])

test_X = torch.Tensor([i[0] for i in testing_data]).view(-1,NEIGHBOURS)
test_X = test_X/MAX_DISTANCE
test_Y = torch.Tensor([i[1] for i in testing_data])

os.mkdir(MATRIX_FOLDER)

def train(net):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    for epoch in range(EPOCH):
        #for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        for i in range(0, len(train_X), BATCH_SIZE):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1,NEIGHBOURS)
            batch_Y = train_Y[i:i+BATCH_SIZE]
            
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            net.zero_grad()
            outputs = net(batch_X)
            
            matches = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs,batch_Y)]
            in_sample_acc = matches.count(True)/len(matches)
            
            loss = loss_function(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            
        train_accuracies.append(float(in_sample_acc))
            
        #print(f"Epoch: {epoch}. Loss: {loss}, Acc: {in_sample_acc}")
        
def test(net, time):
    correct = 0
    total = 0
    
    with torch.no_grad():
        #for i in tqdm(range(len(test_X))):
        for i in range(len(test_X)):
            real_class = torch.argmax(test_Y[i]).to(device)
            net_out = net(test_X[i].view(-1,NEIGHBOURS).to(device))[0]
            predicted_class = torch.argmax(net_out)
            
            if predicted_class == real_class:
                correct += 1
            total += 1
            
            acc = round(correct/total,3)
            
            for real, predict in zip(real_class.view(-1), predicted_class.view(-1)):
                confusion_matrix[real.long(),predict.long()] +=1
                
    accuracies.append(float(acc))
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    norm_confusion_matrix = confusion_matrix/confusion_matrix.sum(1)
    
    path = MATRIX_FOLDER + f"_{time}.png"
    path = os.path.join(MATRIX_FOLDER, path)
    
    fig, ax = plt.subplots()
    im = ax.imshow(norm_confusion_matrix,cmap="Greens")
    
    ax.set_xticks(np.arange(len(label_predict)))
    ax.set_yticks(np.arange(len(label_Real)))
    
    ax.set_xticklabels(label_predict)
    ax.set_yticklabels(label_Real)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    for i in range(len(label_Real)):
        for j in range(len(label_predict)):
            text = ax.text(j, i, round(norm_confusion_matrix[i, j].item(),3), ha="center", va="center", color="k")
            
    ax.set_title(f"Confusion Matrix after {time+1} Epochs")
    fig.tight_layout()
    plt.savefig(path)
    
    #print("Accuracy:",acc)

for i in tqdm(range(EPOCHS)):
    train(net)
    test(net,i)
    print(f"Train Accuracy: {train_accuracies[i]}   Accuracy: {accuracies[i]}")
    timestamp.append(int(i))
    
    if i in [25,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(timestamp, train_accuracies, label="Training")
        ax.plot(timestamp, accuracies, label="Validating")
        ax.legend(loc=4)
        ax.set_title('Classification Accuracy')
        pngname = PNGNAME + f"_{i}.png"
        path = os.path.join(MATRIX_FOLDER, pngname)
        plt.savefig(path)

path = os.path.join(MATRIX_FOLDER, "log")
with open(path,"w") as log:
    for i in range(len(timestamp)):
        logging = str(timestamp[i]) + "    " + str(train_accuracies[i]) + "    " + str(accuracies[i]) + "\n"
        log.write(logging)


    