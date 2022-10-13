# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:50:24 2022

@author: 

"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset , DataLoader

#Declaration of file path （File directory)
test_data_file = 'data/test_data.npz'
train_data_file = 'data/train_data.npz'
val_data_file = 'data/valid_data.npz'
batchsize = 50
epochs = 99 #Number of training cycles
lr = 0.001 #Default 0.001

#Create an instance of SummaryWriter for tensorboard 
writer = SummaryWriter()

transformation = T.Compose([T.ToTensor(), #Converts images to Tensor
                            #T.ConvertImageDtype(dtype= torch.long),
                            T.Normalize(0, 0.5),  #Normalise image using Range Rule where std = (Max -Min)/4
                            T.RandomHorizontalFlip(), #Random Horizontal Flip 
                            #T.RandomVerticalFlip(), #Random Vertical Flip (Not useful in our test)
                            #T.RandomRotation(degrees = (0,180)), #Random Rotation (Not useful in our test)
                            T.Pad(4), #Pad the image by 4
                            T.RandomCrop(20*20) #Randomly crop
                            ])

#Create a CustomDataSet using the Dataset module
class CustomDataSet(Dataset):
    #Initialise 
    def __init__(self, data_file,transform=None,target_transform=None):
        #Load data to img_labels and img_dir
        #Might want to consider modifying the cr2 files to raw files and do all
        #sorts of manipulation here before loading the files.
        #Best to check if the files given is already in right format before running ML
        data = np.load(data_file)['arr_0']
        self.img_labels = list(zip(*data))[0]
        self.img_dir = list(zip(*data))[1]
        self.transform = transform
        self.target_transform = target_transform
    
    #Determine the length
    def __len__(self):
        return len(self.img_labels)
    
    #Use for obtaining data from the dataset
    def __getitem__(self,idx):
        #Extract labels and image based on index
        image = self.img_dir[idx]
        label = self.img_labels[idx]
        #Apply transformation if required
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return label,image

#Create instances of the class CustomDataSet
train_set = CustomDataSet(train_data_file,transformation)
test_set = CustomDataSet(test_data_file,transformation)
val_set = CustomDataSet(val_data_file,transformation)

#Load Data using Dataloader, with batchsize and shuffle=True
trainloader = DataLoader(train_set,batch_size=batchsize,shuffle =True)
testloader = DataLoader(test_set,batch_size=batchsize,shuffle =True)
valloader = DataLoader(val_set,batch_size=batchsize,shuffle =True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    #Use for defining layers
    def __init__(self):
        super(Model,self).__init__()
        # 5×5 Convolutional Layer with 32 filters, stride 1 and padding 2.
        self.conv1 = nn.Conv2d(3, 32, 5,stride=1,padding=2)
        # 2×2 Max Pooling Layer with a stride of 2.
        self.maxpool1 = nn.MaxPool2d(2,stride=2)
        # Batch Normalisation of 32 inputs
        self.batchnorm1 = nn.BatchNorm2d(32) 
        # 3×3 Convolutional Layer with 64 filters, stride 1 and padding 1.
        self.conv2 = nn.Conv2d(32, 64, 3,stride=1,padding=1)
        # 2×2 Max Pooling Layer with a stride of 2.
        self.maxpool2 = nn.MaxPool2d(2,stride=2)
        #Batch Normalisation of 64 inputs
        self.batchnorm2 = nn.BatchNorm2d(64) 
        # 3×3 Convolutional Layer with 128 filters, stride 1 and padding 1.
        self.conv3 = nn.Conv2d(64, 128, 3,stride=1,padding=1)
        # 2×2 Max Pooling Layer with a stride of 2.
        self.maxpool3 = nn.MaxPool2d(2,stride=2)
        # Batch Normalisation of 128 inputs
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        # self.conv4 = nn.Conv2d(128, 128, 3,stride=1,padding=1)
        # # 2×2 Max Pooling Layer with a stride of 2.
        # self.maxpool4 = nn.MaxPool2d(2,stride=2)
        # # Batch Normalisation of 128 inputs
        # self.batchnorm4 = nn.BatchNorm2d(128)
        
        
        
        
        #Flatten for Fully-connected layer
        self.flatten1 = nn.Flatten() 
        # Fully-connected layer with 1024 output units
        self.fc1      = nn.Linear(320000, 1024) 
        #Flatten for Fully-connected layer
        self.flatten2 = nn.Flatten()
        # Fully-connected layer with 10 output units        
        self.fc2      = nn.Linear(1024, 1)
        
    #Use to connect the layers
    def forward(self,x):
        #Apply layers in the following order:
        # x -> conv1 -> Relu -> maxpool1 -> batchnorm1 -> 
        # conv2 -> Relu -> maxpool2 -> batchnorm2 ->
        # conv3 -> Relu -> maxpool3 -> batchnorm3 ->
        # flatten1 -> fc1 -> flatten2 -> fc2 
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool3(x)
        x = self.batchnorm3(x)
        
        # x = F.relu(self.conv4(x))
        # x = self.maxpool4(x)
        # x = self.batchnorm4(x)
        
        x = self.flatten1(x)
        x = F.relu(self.fc1(x))
        x = self.flatten2(x)
        x = self.fc2(x)
        return x
def trainNValidate():
#Create instance of Model()
    #model = torchvision.models.resnet18(pretrained=True)
    model = Model()
    model.to(device)
    #Create instance of CrossEntropyLoss()
    #ce_loss_func = nn.CrossEntropyLoss()
    loss_func = nn.MSELoss()
    #Create instance of Adam optimizer
    optimizer = optim.Adam(model.parameters(),lr = lr,betas = (0.9,0.999))
    
    #Cycle through the determined epochs
    for epoch in range(epochs):
        print("Running")
        running_loss = 0.0
        #Training Process
        for i,data in enumerate(trainloader,0):
            #Read data
            labels,inputs = data
          
            #Set optimizer to zero gradient
            optimizer.zero_grad()
            model.train()
            output = model(inputs.float())
            #print(f'Output of Model:{output}')
            loss = loss_func(output.float(),labels.float().view(-1, 1))
            
            #backward propagation, compute the parameters to reach the output
            loss.backward()
            #Update of parameters
            optimizer.step()
            
            #Adding to tensorboard
            writer.add_scalar("Loss/Train", loss,epoch)
            
            #Print statistics
            running_loss += loss.item()
            
            print(f'[{epoch+1},{i+1:5d}] loss: {running_loss:.3f}')
            running_loss = 0.0
            
        #Declare training directory PATH to save checkpoints
        PATH = './checkpoints/PCPC_net_{:02d}.pth'.format(epoch)
        #Save model state dictionary to PATH
        torch.save(model.state_dict(),PATH)
        
        validation_loss = 0.0
        running_loss = 0.0
        print("Starting Validation for epoch {}".format(epoch+1))
        #Deactivate gradient
        with torch.no_grad():
            #Validation Process
            for j,data in enumerate(valloader,0):
                #Read data
                labels,inputs = data
                
                #Set optimizer to zero gradient
                optimizer.zero_grad()
                model.eval()
                output = model(inputs.float())
                #print(f'Output of Model:{output}')
                loss = loss_func(output.float(),labels.float().view(-1, 1))
                
                validation_loss += loss
                    
            #Average validation loss    
            validation_loss /= len(valloader)
            print('Validation loss for epoch {:2d} : {:5f}'.format(epoch+1,validation_loss))
        
        #Declare validation directory PATH to save checkpoints
        PATH = './checkpoints/PCPC_val_{:02d}.pth'.format(epoch)
        #Save model state dictionary to PATH
        torch.save(model.state_dict(),PATH)
        #Adding to tensorboard
        writer.add_scalar("Loss/Validation", validation_loss,epoch)
    
    print("Finish Training")

def accuracy(): 
    
    for epoch in range(epochs):
        #Trained Model Against Test
        #New instance of Model
        #model = torchvision.models.resnet18(pretrained=True)
        model = Model()
        model.to(device)
        #Manipulate string such that it can read up to 100 epoch
        if (epoch<10):
            TRAIN_PATH = './checkpoints/PCPC_net_0'+str(epoch)+'.pth'
        else:
            TRAIN_PATH = './checkpoints/PCPC_net_'+str(epoch)+'.pth'
        
        #Load saved checkpoints
        model.load_state_dict(torch.load(TRAIN_PATH))
        
        correct =0 
        total = 0
        #Deactivate gradient
        with torch.no_grad():
            for data in testloader:
                #Read Data
                labels,images = data
                
                outputs = model(images.float())
                
                #Choose class with the highest accuracy
                predicted,_ = torch.max(outputs.data, 1)
                total += labels.size(0)
                #print(f'Predicted: {predicted} VS Actual: {labels}')
                correct += (abs(predicted - labels)<0.1*labels).sum().item()
                
        print(f'Accuracy of the network(Train){epoch} on the test images:{100 * correct//total} %')
        
        #Adding to tensorboard
        writer.add_scalar("Accuracy/Training",100*correct//total,epoch)
        
        #Validation Model Against Test
        #New instance of Model
        model = Model()
        #model = torchvision.models.resnet18(pretrained=True)
        #Manipulate string such that it can read up to 100 epoch
        if (epoch<10):
            VAL_PATH = './checkpoints/PCPC_val_0'+str(epoch)+'.pth'
        else:
            VAL_PATH = './checkpoints/PCPC_val_'+str(epoch)+'.pth'
        
        #Load saved checkpoints
        model.load_state_dict(torch.load(VAL_PATH))
        
        correct =0 
        total = 0
        #Deactivate gradient
        with torch.no_grad():
            for data in testloader:
                #Read Data
                labels,images = data
                
                outputs = model(images.float())
                
                #Choose class with the highest accuracy
                predicted,_ = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                #print(f'Predicted: {predicted} VS Actual: {labels}')
                correct += (abs(predicted - labels)<0.1*labels).sum().item()
                
        print(f'Accuracy of the network(Validation){epoch} on the test images:{100 * correct//total} %')
        
        #Adding to tensorboard
        writer.add_scalar("Accuracy/Validation",100*correct//total,epoch)


def convertBinary(v,TOL=0.2):
    '''
    

    Parameters
    ----------
    v : Tensor array of Labels
        This parameter contains all 

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    lower = v > (80*(1-TOL))
    #print(f'Lower:{lower}')
    upper = v < (160*(1+TOL))
    #print(f'Upper:{upper}')
    
    #print(lower&upper)
    return (lower&upper)

def countMean(v):
    result = 0
    for value in v:
        result += value
    result = result/len(v)
    return result


def check_debug(epoch = 39,TOL = 0.20):
    combined_accuracy = 0
    model = Model()
    model.to(device)
    if (epoch<10):
        TRAIN_PATH = './checkpoints/PCPC_net_0'+str(epoch)+'.pth'
    else:
        TRAIN_PATH = './checkpoints/PCPC_net_'+str(epoch)+'.pth'
    model.load_state_dict(torch.load(TRAIN_PATH))
    correct =0 
    total = 0
    #Deactivate gradient
    with torch.no_grad():
        for data in testloader:
            #Read Data
            labels,images = data
            outputs = model(images.float())
            
            #Choose class with the highest accuracy
            predicted,_ = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            #print(f'Predicted: {predicted} VS Actual: {labels}')
            print(f'Epoch:{epoch}')
            print(f'Predicted: {predicted}')
            print(f'Actual: {labels}')
            correct += (abs(predicted - labels)<TOL*labels).sum().item()
    
    print(f'Accuracy of the network(Train) on the test images:{100 * correct//total} %')
    combined_accuracy = 100 * correct//total
    #Adding to tensorboard
    writer.add_scalar("Accuracy/Training",100*correct//total,epoch)
    
    #Validation Model Against Test
    #New instance of Model
    model = Model()
    #Manipulate string such that it can read up to 100 epoch
    if (epoch<10):
        VAL_PATH = './checkpoints/PCPC_val_0'+str(epoch)+'.pth'
    else:
        VAL_PATH = './checkpoints/PCPC_val_'+str(epoch)+'.pth'
    
    #Load saved checkpoints
    model.load_state_dict(torch.load(VAL_PATH))
    
    correct =0 
    total = 0
    #Deactivate gradient
    with torch.no_grad():
        for data in testloader:
            #Read Data
            labels,images = data

            outputs = model(images.float())
            
            #Choose class with the highest accuracy
            predicted,_ = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            #print(f'Predicted: {predicted} VS Actual: {labels}')
            print(f'Predicted: {predicted}')
            print(f'Actual: {labels}')
            
            correct += (abs(predicted - labels)<TOL*labels).sum().item()
    
    print(f'Accuracy of the network(Validation) on the test images:{100 * correct//total} %')
    combined_accuracy += 100 * correct//total
    combined_accuracy /= 2
    print(f'Average Accuracy = {combined_accuracy}%')
    #Adding to tensorboard
    writer.add_scalar("Accuracy/Validation",100*correct//total,epoch)

def decide(demo_data_file,epoch = 39,TOL=0.2):
    
    demo_set = CustomDataSet(demo_data_file,transformation)
    demoloader = DataLoader(demo_set,batch_size=batchsize,shuffle =False)
    model = Model()
    model.to(device)
    if (epoch<10):
        TRAIN_PATH = './checkpoints/PCPC_net_0'+str(epoch)+'.pth'
    else:
        TRAIN_PATH = './checkpoints/PCPC_net_'+str(epoch)+'.pth'
    model.load_state_dict(torch.load(TRAIN_PATH))
    with torch.no_grad():
        for data in demoloader:
            #Read Data
            labels,images = data
            outputs = model(images.float())
            
            #Choose class with the highest accuracy
            predicted,_ = torch.max(outputs.data, 1)
        print(f'Epoch:{epoch}')
        #print(f'Predicted: {predicted}')
        binary = convertBinary(predicted,TOL=TOL)
        average_thickness = countMean(predicted)
        confidence = sum(binary)/len(binary)*100
        #print(f'Final Decision: {binary}')
        if sum(binary)>=4:
            print(f'Final Decision: Pass with {(confidence):0.2f}% confidence')
            print(f'Average thickness:{average_thickness}\n')
        else:
            print(f'Final Decision: Fail with {(100-confidence):0.2f}% confidence')
            print(f'Average thickness:{average_thickness}\n')


if __name__ == '__main__':
    #trainNValidate()
    #accuracy()
    #check_debug()
