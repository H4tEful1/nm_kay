import os
from config.settings import path_routing
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import Dataset
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr



fnames = ["kay_labels.npy", "kay_labels_val.npy", "kay_images.npz"]

with np.load(os.path.join(path_routing.database_path, fnames[2])) as dobj:
  dat = dict(**dobj)
labels = np.load(os.path.join(path_routing.database_path, 'kay_labels.npy'))
val_labels = np.load(os.path.join(path_routing.database_path, 'kay_labels_val.npy'))

# Converting stimulus to RGB and changing the scale to 0-255 (Specific to Kay dataset images)
stimuli_tr = dat["stimuli"]
stimuli_ts = dat["stimuli_test"]
stimuli_tr_xformed = np.zeros((1750, 3, 128, 128))
stimuli_ts_xformed = np.zeros((120, 3, 128, 128))
for i in range(1750):
  img = stimuli_tr[i, :, :]
  img = ((img - np.min(img))*255/(np.max(img) - np.min(img))).astype(int)
  stimuli_tr_xformed[i, :, :, :] = [img,img,img]

for i in range(120):
  img = stimuli_ts[i, :, :]
  img = ((img - np.min(img))*255/(np.max(img) - np.min(img))).astype(int)
  stimuli_ts_xformed[i, :, :, :] = [img, img, img]

# @title Setting up training and test data for LOC region
loc_id_tr = np.where(dat['roi'] == 7)
loc_id_ts = np.where(dat['roi'] == 7)

response_tr = np.squeeze(dat["responses"][:, loc_id_tr])
response_ts = np.squeeze(dat["responses_test"][:, loc_id_ts])

# @title Custom dataloader for loading images in numpy array
# Custom dataloader for loading images in numpy array
class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.FloatTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1, 2, 0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dataset = {
    'train': MyDataset(stimuli_tr_xformed, response_tr, transform=transform['train']),
    'val': MyDataset(stimuli_ts_xformed, response_ts, transform=transform['val'])
}
dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(dataset[x], batch_size=50, shuffle=True) for x in ['train', 'val']}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to train model
def train_model(dataloaders, dataset_sizes, num_epochs=5, learning_rate=0.2):
    net = models.alexnet(pretrained=True)
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, np.shape(response_ts)[1])

    net.to(device)

    criterion = nn.MSELoss()  # We can change it.
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs.float(), labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f"{phase} Loss: {epoch_loss:.4f}")

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())

        print()

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net


def boost_model(dataloaders, dataset_sizes, num_models=7, num_epochs=5, learning_rate=0.2):
    models_list = []
    individual_mse = []

    for i in range(num_models):
        dataloader = copy.deepcopy(dataloaders)
        dataset_size = copy.deepcopy(dataset_sizes)
        print(f"Training model {i+1}/{num_models}")
        net = train_model(dataloader, dataset_size, num_epochs, learning_rate)
        models_list.append(net)

        # Predict with the current model and calculate MSE
        predictions, true_labels = predict([net], dataloaders, phase='val')
        mse = np.mean((predictions - true_labels) ** 2)
        individual_mse.append(mse)
        print(f"Model {i+1} MSE: {mse:.4f}")

    return models_list, individual_mse


def predict(models_list, dataloaders, phase='val'):
    for model in models_list:
        model.eval()

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            outputs = torch.zeros((inputs.size(0), np.shape(response_ts)[1])).to(device)

            for model in models_list:
                outputs += model(inputs)

            outputs /= len(models_list)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_outputs, all_labels

def evaluate_pretrained_alexnet(dataloaders, dataset_sizes):
    net = models.alexnet(pretrained=True)
    num_ftrs = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ftrs, np.shape(response_ts)[1])

    net.to(device)
    net.eval()

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            outputs = net(inputs)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mse = np.mean((all_outputs - all_labels) ** 2)
    print(f"Pretrained AlexNet MSE: {mse:.4f}")

    return mse


# Assume dataloaders and dataset_sizes are defined
pretrained_mse = evaluate_pretrained_alexnet(dataloaders, dataset_sizes)
models_list, individual_mse = boost_model(dataloaders, dataset_sizes, num_models=5, num_epochs=10, learning_rate=0.1)
predictions, true_labels = predict(models_list, dataloaders, phase='val')

# Evaluate the combined predictions
ensemble_mse = np.mean((predictions - true_labels) ** 2)
print(f"Boosted Model MSE: {ensemble_mse:.4f}")

# Plotting MSE for individual models, pretrained AlexNet, and boosted ensemble using a line plot
plt.figure(figsize=(10, 6))
model_indices = list(range(1, len(individual_mse) + 1))
plt.plot(model_indices, individual_mse, marker='o', linestyle='-', label='Individual Models')
plt.axhline(y=pretrained_mse, color='g', linestyle='--', label='Pretrained AlexNet')
plt.axhline(y=ensemble_mse, color='r', linestyle='--', label='Boosted Ensemble')

plt.xlabel('Model Number')
plt.ylabel('MSE')
plt.title('MSE of Individual Models, Pretrained AlexNet, and Boosted Ensemble')
plt.legend()
plt.show()



