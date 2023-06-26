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

"""
This is an mnist auto-encoder.  The purpose is to visualize how ordinary auto-encoders have a latent
space that is not regular.  This is why variational auto-encoders are needed.  

"""

# CONSTANTS #

# This is the dimension of the bottle neck.  It is the most important parameter in AE
bottle_neck_size = 36
MNIST_PATH = './data/'
MNIST_FILE = MNIST_PATH + "mnist.pkl.gz"


def load_mnist():
    with gzip.open(MNIST_FILE, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid)) = pickle.load(f, encoding='latin-1')

    x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid))

    # convert pixel values to be between 0 and 1
    x_train, x_valid = x_train / 255.0, x_valid / 255.0

    return x_train, y_train, x_valid, y_valid


def set_data_filters(f, train_data, target_data):
    train_filter = np.isin(target_data, f)

    return train_data[train_filter], target_data[train_filter]


class Model(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size, decoder_in_size, decoder_out_size):
        super(Model, self).__init__()

        self.input_shape = None  # Store the input shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in_size**2, 400),
            nn.BatchNorm1d(400),  # Add BatchNorm layer
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),  # Add Dropout layer
            nn.Linear(400, 300),  # Add another Linear layer
            nn.BatchNorm1d(300),  # Add BatchNorm layer
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),  # Add Dropout layer
            nn.Linear(300, encoder_out_size)  # Connect to bottleneck
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(decoder_in_size, 300),  # Connect from bottleneck
            nn.BatchNorm1d(300),  # Add BatchNorm layer
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),  # Add Dropout layer
            nn.Linear(300, 400),  # Add another Linear layer
            nn.BatchNorm1d(400),  # Add BatchNorm layer
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),  # Add Dropout layer
            nn.Linear(400, decoder_out_size**2),
            nn.Sigmoid()
        )

    def forward(self, x):
        self.input_shape = x.shape
        if x.size(0) == 1:  # Handle the single sample case
            x = x.view(-1).unsqueeze(0).to(torch.float)  # Flatten the input data and add a new dimension
        else:
            x = x.view(x.size(0), -1).to(torch.float)  # Flatten the input data
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(self.input_shape)
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
                x = x.view(x.size(0), -1)  # Flattening the images
                latent_vectors = self.encoder(x)
                encoded_samples.append((latent_vectors.cpu().numpy(), y.cpu().numpy()))
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
    random_image.unsqueeze(0)
    print(random_image.shape)
    plt.imshow(random_image_for_display)
    plt.show()
    plt.imshow(random_model_image_for_display)
    plt.show()


def train(model, opt, dl, num_epochs):
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

        axes[i, 0].imshow(input_image.cpu().numpy(), cmap='gray')
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(output_image.detach().cpu().numpy(), cmap='gray')
        axes[i, 1].set_title("Output")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def main():

    x_train, y_train, x_valid, y_valid = load_mnist()

    # Set here what digits you want to plot.  Plotting more than
    # 3-4 digits will result in a mess
    train_filter = [0, 1, 2, 3]
    x_train, y_train = set_data_filters(train_filter, x_train, y_train)

    print(x_train.shape)
    print(y_train.shape)

    # Set the dimensions of the encoder / decoder
    encoder_in_size = x_train.shape[1]
    print("encoder input size", encoder_in_size)
    encoder_out_size = bottle_neck_size
    decoder_in_size = bottle_neck_size
    # decoder_out_size must be same as the encoder_in_size since we are trying to reconstruct the input
    decoder_out_size = encoder_in_size

    model = Model(encoder_in_size, encoder_out_size, decoder_in_size, decoder_out_size)

    print(model)

    # Convert the training tensors to the appropriate datatypes
    x_train = x_train.to(torch.float)
    y_train = y_train.to(torch.long)

    train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)

    # Batch size
    bs = 8

    # Dataloader
    train_dl = DataLoader(train_ds, bs, sampler=RandomSampler(train_ds))

    # Optimizer
    learning_rate = 1.e-3
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Epochs
    epochs = 1

    # Plot a random image from the training data and show how the
    # decoder reproduces it
    random_img = get_random_image(x_train)
    random_img = random_img.unsqueeze(0)
    print("random image shape", random_img.shape)
    model.eval()
    random_model_image = model(random_img)
    random_model_image = convert_model_output_to_img(random_model_image)
    random_img = random_img.reshape(28, 28).numpy()

    display_img(random_img)
    display_img(random_model_image)
    model.train()
    train(model, opt, train_dl, epochs)
    model.eval()
    # Plot a random image from the training data and show how the
    # decoder reproduces it
    random_img = get_random_image(x_train)
    random_img = random_img.unsqueeze(0)
    print("random image shape", random_img.shape)

    random_model_image = model(random_img)
    random_model_image = convert_model_output_to_img(random_model_image)
    random_img = random_img.reshape(28, 28).numpy()

    display_img(random_img)
    display_img(random_model_image)

    # Create a DataLoader
    train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)

    visualizer = LatentSpaceVisualizer(model.encoder, train_dataloader)
    visualizer.run()

    num_pairs = 5
    plot_input_output_pairs(model, x_train, num_pairs)

main()






