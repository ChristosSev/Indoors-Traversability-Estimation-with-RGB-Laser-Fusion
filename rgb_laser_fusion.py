import os
import csv
from glob import glob
import pathlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset,  ConcatDataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from transformers import SegformerForImageClassification
import requests
import pickle
from PIL import Image
import torch
import glob
import torchvision
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable
import numpy as np
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import pathlib
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import os
from sklearn.manifold import TSNE
import warnings
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

batch_size = 64
    
path_to_laser = '/data/zak/robot/extracted/mocap/8_26/front_laser/'

path_to_train = '/home/christos/data/liga/erb'
path_to_test = '/home/christos/data/liga/nh/train'

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Transforms
transformer=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])



train_loader=DataLoader(
    torchvision.datasets.ImageFolder(path_to_train,transform=transformer),
    batch_size=1, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(path_to_test,transform=transformer),
    batch_size=1, shuffle=False
)

#calculating the size of training and testing image
train_count=len(glob.glob(path_to_train+'/**/*.jpg'))
test_count=len(glob.glob(path_to_test+'/**/*.jpg'))


lasers = glob.glob(path_to_laser + '*pkl')
lasers.sort()

for l in lasers:
    #print(l)
    file_to_read = open(l, "rb")
    loaded_dictionary_lasers = pickle.load(file_to_read)


    ranges = loaded_dictionary_lasers["ranges"]
    ranges_reshaped = np.array(ranges).reshape(1, 1081)
   

    ranges_200_400 = ranges_reshaped[0][200:400]

    #if np.any(ranges_200_400 <= 1.2):
        #print("")


laser = ranges_reshaped





class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=2):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.classifier = nn.Linear(256,256)
        self.modelB.classifier = nn.Linear(768,768)
        
        # Create new classifier
        self.classifier = nn.Linear(768+256, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.logits
        #print(x1.size(), "x1 size")
        #x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.logits
       # print(x2.size(), "x2 size")
        #x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        #print(x.size(), "after concatenated")
        #x = self.classifier(F.relu(x))
        #print(x.size(), "after class")
        return x


    
    
class Projectorr(nn.Module):
    def __init__(self):
        super(Projectorr, self).__init__()
        #self.nf = 768

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1081, 512),
            # nn.BatchNorm1d(self.nf),
            nn.ReLU(True),
        )

        self.mha = nn.MultiheadAttention(512, 2)

   

    def forward(self, x1, x2):
        
        
       # print(x1.shape)
        x1 = self.fc1(x1)
        
       # x2 = x2.view(batch_size, -1)
        x2 = self.fc2(x2)
        #print(x2)
          
        x, _ = self.mha(x2, x1, x1)
        
        #print(x.shape,"after mha")
        # x = self.fc(x)
        return x    
    
        

class Lin(nn.Module):
    def __init__(self):
        super(Lin, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(512, 2),
        )

    def forward(self, x):
        
        #print(x.shape,"fc before")
        x = self.fc(x)
        
       # print(x.shape,"fc after")
        return x    
    
model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
model2 = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
i = 0
    
    
    
    
    # Freeze these models
for name, param in model.named_parameters():
    #print(i, name)
    #print(param)
    if i < 192:
        param.requires_grad = False
    i += 1

for name, param in model2.named_parameters():
    #print(i, name)
    if i < 197:
        param.requires_grad = False
    i += 1       
    
    
    
model3 = MyEnsemble(model, model2)    
#print(model3)

model3.eval()

model3.to(device)


proj2 = Projectorr().to(device)
lin = Lin().to(device)


    
allWeights = list(model3.parameters()) +  list(proj2.parameters()) + list(lin.parameters())
optimizer= SGD(allWeights, lr=0.01, weight_decay=0.0001)
#optimizer = Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)

criterion=nn.CrossEntropyLoss()

#Model training and saving best model

#print(model)


trnLosses, tstLosses = [], []
accs = []
best_accuracy = 0.0

for epoch in range(50):
    trnLabel, trnPred = [], []
    tstLabel, tstPred = [], []
    # Evaluation and training on training dataset
#     model.train()
#     model2.train()
    model3.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for images, labels in train_loader:
        #print(i)
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            #print(images.shape, "images shape")
           
           
            labels = Variable(labels.cuda())
            
        optimizer.zero_grad()
        
        
        lasers = laser 
        laser = torch.tensor(lasers)
        laser = laser.float()
        laser = Variable(laser.cuda())
    
        outputs1 = model3(images)
        #print(outputs1.size())
        
       # outputs2 = model2(images).logits
        
        
        #outputs_proj = proj(outputs1, outputs2)
      
    
        outputs_proj2 = proj2(outputs1,laser)
    
    

        outputs = lin(outputs_proj2)
       
        
        
        err = criterion(outputs, labels)
        err.backward()
        optimizer.step()

       

        #train_loss += loss.device().data * images.size(0)
        _, prediction = torch.max(outputs, 1)
        trnLosses.append(err.item())
        trnLabel.append(labels)
        trnPred.append(prediction)

        
       # print(prediction.size(),"biaaa")

       # train_accuracy += float(torch.sum(prediction == labels.data))
        
   # train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    train_accuracy += float(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count

    # Evaluation on testing dataset

    model3.eval()
    test_accuracy = 0.0
    labels_list = []
    prediction_list = []
    x_list = []
    y_list = []

    for i, (images, labels) in enumerate(test_loader):
        
        if torch.cuda.is_available():
            images=torch.tensor(images)
            images = Variable(images.cuda()) 
            #labels = torch.tensor(labels)
            labels = Variable(labels.cuda())
            
        optimizer.zero_grad()
        
        
        lasers = laser 
        laser = torch.tensor(lasers)
        laser = laser.float()
        laser = Variable(laser.cuda())
    
        outputs1 = model3(images)
        # print(outputs.size())
        
        #outputs2 = model2(images).logits
        
        
        outputs_proj = proj2(outputs1, laser)
      
    
    
    
        #outputs_proj2 = proj2(outputs_proj,laser)
    
    

        outputs = lin(outputs_proj)
        _, prediction = torch.max(outputs, 1)
        
        
        
        err = criterion(outputs, labels)

        tstLosses.append(err.item())
        tstLabel.append(labels)
        tstPred.append(prediction)
        
        
        
        
        #prediction_list.extend(prediction.cpu())
        # print(prediction_list)
        #print(prediction.size(), "predictions size")
        #print(labels.size(), "labels size")
        #print(prediction, "prediction is")
        #print(labels.data, "data")
        #print(torch.sum(prediction == labels.data), "prosthesi")
        test_accuracy += int(torch.sum(prediction == labels.data))

    test_accuracy = test_accuracy / test_count
    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Test Accuracy: ' + str(test_accuracy))







