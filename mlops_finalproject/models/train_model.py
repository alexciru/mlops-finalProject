# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from model import ModifiedMobileNetV3
import matplotlib.pyplot as plt

#Hyperparameters
num_epochs = 20

# # Define the data transforms
# data_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# Load the dataset
images = torch.load("data/processed/images.pt")
labels = torch.load("data/processed/labels.pt")
labels = labels.type(torch.LongTensor)
print(images.shape)
print(labels[0:12].shape)
train_dataset = TensorDataset(images, labels[0:12])
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True) 

#Creating a instance of the model
model = ModifiedMobileNetV3(num_classes=43)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
losses = []
steps = 0
model.train
for epoch in range(num_epochs):
    running_loss = 0
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        losses.append(running_loss/len(trainloader))
        steps += 1
        print(f"Training loss: {running_loss/len(trainloader)}")


# Use the plot function to draw a line plot
plt.plot(range(steps), losses)

# Add a title and axis labels
plt.title("Training Loss vs Training Steps")
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")

# Save the plot
plt.savefig("reports/figures/lossV1.png")

torch.save(model.state_dict(), 'models/trained_modelV1.pt')

# # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for i, (inputs, labels) in enumerate(dataloader):
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# # Print the accuracy
# print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
