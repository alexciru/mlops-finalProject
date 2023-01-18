import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Classifier
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
#import wandb
import random

class MyDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = torch.load(images_path)
        self.labels = torch.load(labels_path)
        self.labels = self.labels.type(torch.LongTensor)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

#Hyperparameters
num_epochs = 3
learning_rate = 0.001   

# my own version of dataset loading
dataset = MyDataset('data/processed/images.pt', 'data/processed/labels.pt')
print(len(dataset))
trainloader = DataLoader(dataset, batch_size=128, shuffle=True)

#Creating a instance of the model
model = Classifier(num_classes=43)

#init the wandb logging
#wandb.watch(model, log_freq=100)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# random guess NN
images, labels = next(iter(trainloader))
# Get the class probabilities
#ps = torch.exp(model(images))
# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
#print(f"Shape of the pred: {ps.shape}")
#top_p, top_class = ps.topk(1, dim=1)
#equals = top_class == labels.view(*top_class.shape)
#accuracy = torch.mean(equals.type(torch.FloatTensor))
#print(f'Accuracy with random guess: {accuracy.item()*100}%')

# Train the model
train_losses = []
losses = []
steps = 0

train_losses, test_losses = [], []
for e in range(num_epochs):
    tot_train_loss = 0
    optimizer.zero_grad()
    for images, labels in trainloader:
        #print(f"img size: {images.shape}")
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        tot_train_loss += loss.item()

        loss.backward()
        optimizer.step()
    else:
        tot_test_loss = 0
        test_correct = 0  # Number of correct predictions on the test set
        
        # Turn off gradients for validation, saves memory and computations
        
        # Get mean loss to enable comparison between train and test sets
        train_loss = tot_train_loss / len(trainloader.dataset)
        #test_loss = tot_test_loss / len(testloader.dataset)

        # At completion of epoch
        train_losses.append(train_loss)
        #test_losses.append(test_loss)
        steps += 1
        print("Epoch: {}/{}.. ".format(e+1, num_epochs),
              "Training Loss: {:.5f}.. ".format(train_loss))


# Use the plot function to draw a line plot
#plt.plot(range(steps), losses)

# Add a title and axis labels
#plt.title("Training Loss vs Training Steps")
#plt.xlabel("Training Steps")
#plt.ylabel("Training Loss")
# Save the plot
#plt.savefig("reports/figures/lossV4.png")

# Save the model weights.
torch.save(model.state_dict(), 'models/trained_modelV9.pt')

