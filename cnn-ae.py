import gzip
import pickle
import os
import random

import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA, PCA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(0)


"""
This is an mnist auto-encoder.  The purpose is to visualize how ordinary auto-encoders have a latent
space that is not regular.  This is why variational auto-encoders are needed.  

"""

# CONSTANTS #

# This is the dimension of the bottle neck.  It is the most important parameter in AE

MNIST_PATH = './data/'
MNIST_FILE = MNIST_PATH + "mnist.pkl.gz"


def load_mnist():
    with gzip.open(MNIST_FILE, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid)) = pickle.load(f, encoding='latin-1')

    x_train, y_train, x_valid, y_valid = [tensor(data) for data in (x_train, y_train, x_valid, y_valid)]

    # convert pixel values to be between 0 and 1
    x_train, x_valid = x_train / 255.0, x_valid / 255.0

    return x_train, y_train, x_valid, y_valid



def set_data_filters(f, train_data, target_data):
    train_filter = np.isin(target_data, f)

    return train_data[train_filter], target_data[train_filter]

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size, stride, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels * 2),
            nn.MaxPool2d(2, 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size, stride, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding=1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.clamp(x, min=0, max=1)
        return x




class Dataset:
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


from sklearn.decomposition import PCA
import torch

class LatentSpaceVisualizer:
    def __init__(self, encoder, dataloader):
        self.encoder = encoder
        self.dataloader = dataloader
        self.pca = PCA(n_components=2)

    def run(self):
        self.encode_data()
        self.apply_pca()
        self.plot()

    def encode_data(self):
        self.encoder.eval()
        encoded_samples = []
        with torch.no_grad():
            for batch in self.dataloader:
                x, y = batch
                # x = x.view(x.size(0), -1)  # We don't need to flatten anymore
                latent_vectors = self.encoder(x)
                encoded_samples.append((latent_vectors.view(latent_vectors.size(0), -1).cpu().numpy(), y.cpu().numpy()))  # Flatten the latent vectors
        self.encoded_samples = encoded_samples
        print(f"Encoded samples: {len(self.encoded_samples)}")



    def apply_pca(self):
        X = np.concatenate([x for x, y in self.encoded_samples])
        self.labels = np.concatenate([y for x, y in self.encoded_samples])
        self.latent_vectors = self.pca.fit_transform(X)
        print(f"Shape of PCA applied data: {self.latent_vectors.shape}")

    def plot(self):
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(self.latent_vectors[:, 0], self.latent_vectors[:, 1], c=self.labels)
        plt.legend(*scatter.legend_elements())
        plt.show()


def get_random_image(img_data):
    return img_data[random.randint(0, len(img_data))]


def display_random_training_image(x_train, model):
    # Lets pick a random image from the training set and see how untrained model reproduces it
    random_image = get_random_image(x_train)
    random_model_image = model(random_image)

    random_image_for_display = random_image.reshape(28, 28)

    random_model_image_for_display = random_model_image.reshape(28, 28).detach().numpy()

    plt.imshow(random_image_for_display)
    plt.show()
    plt.imshow(random_model_image_for_display)
    plt.show()


def row_col_loss(x, x_hat, bs):
    bs = x.shape[0]

    a = x.reshape(bs, 28, 28)
    b = x_hat.reshape(bs, 28, 28)

    c = (a - b).sum(axis=1)
    c = c.pow(2)
    d = (a - b).sum(axis=0)
    d = d.pow(2)
    c = c.sum()
    d = d.sum()

    return c + d


def train(model, opt, dl, num_epochs, bs):
    model.train()
    print(f"training for {num_epochs}")
    loss_func = nn.MSELoss()
    counter = 0
    running_loss = 0.0
    for epoch in range(num_epochs):
        for xb, yb in dl:
            opt.zero_grad()
            output = model(xb)
            loss = loss_func(output, xb)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            counter += 1

            if counter % 1000 == 0:
                average_loss = running_loss / 1000
                print(f'Batch {counter}, Average Loss {average_loss}')
                running_loss = 0.0



def convert_model_output_to_img(output):
    output = output.reshape(28, 28).detach().numpy()
    return output


def display_img(img):
    plt.imshow(img)
    plt.show()

def plot_input_output_pairs(model, input_data, num_pairs=5):
    # Select random indices from the input data
    random_indices = torch.randint(low=0, high=len(input_data), size=(num_pairs,))

    # Plot input-output pairs
    fig, axes = plt.subplots(nrows=num_pairs, ncols=2, figsize=(8, 2 * num_pairs))
    for i, idx in enumerate(random_indices):
        input_image = input_data[idx]
        output_image = model(input_image.unsqueeze(0)).squeeze(0)

        # Move channel dimension to the end for imshow
        input_image = input_image.permute(1, 2, 0)
        output_image = output_image.permute(1, 2, 0)

        # Squeeze the last dimension if it's 1 (for grayscale images)
        if input_image.shape[2] == 1:
            input_image = input_image.squeeze(2)
        if output_image.shape[2] == 1:
            output_image = output_image.squeeze(2)

        # Detach and move to CPU for plotting
        input_image = input_image.detach().cpu().numpy()
        output_image = output_image.detach().cpu().numpy()

        axes[i, 0].imshow(input_image, cmap='gray')
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(output_image, cmap='gray')
        axes[i, 1].set_title("Output")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def save_model(model, filename):
    """
    Saves the model to the specified filename.
    """
    torch.save(model.state_dict(), filename)

def main():

    x_train, y_train, x_valid, y_valid = load_mnist()

    # Set here what digits you want to plot.  Plotting more than
    # 3-4 digits will result in a mess
    train_filter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    x_train, y_train = set_data_filters(train_filter, x_train, y_train)

    print(x_train.shape)
    print(y_train.shape)
    print(x_train.dtype)
    in_channels = 1  # grayscale images
    out_channels = 16  # You can adjust this number
    kernel_size = 3  # You can adjust this number
    stride = 1  # You can adjust this number
    model = ConvAutoencoder(in_channels, out_channels, kernel_size, stride)


    print(model)

    # Convert the training tensors to the appropriate datatypes
    x_train = x_train.to(torch.float)
    y_train = y_train.to(torch.long)
    x_train = x_train.unsqueeze(1).to(torch.float)
    train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)

    # Batch size
    bs = 32

    # Dataloader
    train_dl = DataLoader(train_ds, bs, sampler=RandomSampler(train_ds))

    # Optimizer
    learning_rate = 1.e-3
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Epochs
    epochs = 20

    train(model, opt, train_dl, epochs, bs)
    save_model(model, "cnn-ae-model.pkl")
    model.eval()
    # Plot a random image from the training data and show how the
    # decoder reproduces it
    random_img = get_random_image(x_train)
    random_img = random_img.squeeze(0)
    display_img(random_img)
    random_img = random_img.unsqueeze(0)
    random_img = random_img.unsqueeze(0)
    random_model_image = model(random_img)

    random_model_image = random_model_image.squeeze().detach().numpy()
    display_img(random_model_image)

    # Create a DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)

    visualizer = LatentSpaceVisualizer(model.encoder, train_dataloader)
    visualizer.run()

    num_pairs = 5
    plot_input_output_pairs(model, x_train, num_pairs)

main()






