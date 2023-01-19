import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import MobileNetV3Lightning
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import wandb
import random
import os
from tqdm import tqdm
from google.cloud import storage
import os
from datetime import datetime
import hydra

class MyDataset(Dataset):
        def __init__(self, train, path):

            if train:
                self.images_path = os.path.join(path,"data","processed","train","images.pt")
                self.labels_path = os.path.join(path,"data","processed","train","labels.pt")

            else:
                self.images_path = os.path.join(path,"data","processed","test","images.pt")
                self.labels_path = os.path.join(path,"data","processed","test","labels.pt")

            self.images = torch.load(self.images_path)
            self.labels = torch.load(self.labels_path)
            self.labels = self.labels.type(torch.LongTensor)
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

@hydra.main(config_path=os.path.join(os.getcwd(),'models/'), config_name='config.yaml')
def main(cfg): 

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    wandb.init(
        # set the wandb project where this run will be logged
        project="mlops_finalProject",
        name=hydra.utils.get_original_cwd(),
        # track hyperparameters and run metadata
        config={
        "learning_rate": cfg.hyperparameters.learning_rate,
        "architecture": "CNN",
        "dataset": "GTSRB",
        "epochs": cfg.hyperparameters.num_epochs,
        }
    )

    # my own version of dataset loading
    train_dataset = MyDataset(True, root_dir)
    print(len(train_dataset))
    trainloader = DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)

    # test loader
    test_dataset = MyDataset(False, root_dir)
    print(len(test_dataset))
    testloader = DataLoader(test_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    #Creating a instance of the model
    model = MobileNetV3Lightning(cfg.hyperparameters.learning_rate, num_classes=43)

    #init the wandb logging
    wandb.watch(model, log_freq=100)

    # Train the model
    losses = []
    steps = 0
    accu = 0
    for epoch in range(cfg.hyperparameters.num_epochs):
        print(f"epoch: {epoch+1}/{cfg.hyperparameters.num_epochs}")
        running_loss = 0
        for (inputs, labels) in tqdm(trainloader):
            loss_, preds_ = model.training_step(inputs, labels)
            running_loss += loss_.item()

        # Optional: Decrease lr
        #model.optimizer.param_groups[0]['lr'] *= 60

        wandb.log({"Train loss": running_loss/len(trainloader)})
        losses.append(running_loss/len(trainloader))
        steps += 1
        print(f"Training loss: {running_loss/len(trainloader)}")   
        # # Print the accuracy
        train_acc = model.test_model(trainloader)
        test_acc = model.test_model(testloader)
        wandb.log({"Train Acc": train_acc})
        wandb.log({"Test Acc": test_acc})
        print('Accuracy of the model on the train images: {} %'.format(train_acc))
        print('Accuracy of the model on the test images: {} %'.format(test_acc))
        if accu < test_acc:
            accu = test_acc

            # save weights locally
            timestamp = datetime.today().strftime('%Y%m%d_%H%M')
            name = f"trained_model_32img_{timestamp}.pt"
            torch.save(model.state_dict(), os.path.join(root_dir,"models",name))

            # Save to google storage
            # storage_client = storage.Client()
            # buckets = list(storage_client.list_buckets())
            # bucket = storage_client.get_bucket("training-bucket-mlops") 
            # blob = bucket.blob("weights/" + name)
            # blob.upload_from_filename('models/' + name )
            # print(f"Succesfully push the weights {name} into: {bucket}")

    # Use the plot function to draw a line plot
    plt.plot(range(steps), losses)

    # Add a title and axis labels
    plt.title("Training Loss vs Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")

    # Save the plot
    plt.savefig("reports/figures/loss_32img.png")

if __name__ == "__main__":
    main()