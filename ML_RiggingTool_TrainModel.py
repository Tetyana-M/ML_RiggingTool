import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Network(nn.Module):
    """
    Model class.
    """
    def __init__(self, input_size, output_size, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

class ML_RiggingTool_TrainModel():
    """
    Train the model to calculate X, Y and Z coordinates given X, Y and Z rotations
    of two joints.
    """
    def __init__(self, theta_combinations, end_xyz):
        batch_size = 128
        hidden_size = 100
        learning_rate = .0005
        num_epochs = 50

        normalized_input_data = self.normalize(end_xyz)
        normalized_output_data = self.normalize(theta_combinations)

        # Convert numpy arrays to PyTorch tensors.
        input_tensor = torch.tensor(normalized_input_data, dtype=torch.float32)
        output_tensor = torch.tensor(normalized_output_data, dtype=torch.float32)

        # Create a dataset and dataloader.
        dataset = TensorDataset(input_tensor, output_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Instantiate the model, loss function, and optimizer.
        model = Network(input_size=input_tensor.shape[1], output_size=output_tensor.shape[1], hidden_size=hidden_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # TEMPORARY HERE
        LOCAL_PATH = os.environ.get('LOCAL_MAYA')
        save_dir = os.path.realpath(f'{LOCAL_PATH}/ML_RiggingTool/data/')
         
        # Train the model.
        self.train(model, dataloader, criterion, optimizer, save_dir, num_epochs=num_epochs)

        # TEMPORARY TEST
        test = model(torch.Tensor([24.739283077086036, 22.835945425106374, 40.46364814022808]))
        print(test)

    def normalize(self, X):
        """
        Normalize values to be between 0 and 1 for model processing.

        @param X: numpy array of shape (n, m)
        @return X: normalized param array
        """
        epsilon = 1e-10
        Xmean = np.mean(X, axis=1, keepdims=True)
        Xstd = np.std(X, axis=1, keepdims=True)
        # Set standard deviation to 1 for very small values to avoid division
        # by near-zero numbers.
        Xstd[Xstd < epsilon] = 1
        X = (X - Xmean) / Xstd
        
        return X

    def train(self, model, dataloader, criterion, optimizer, save_dir, num_epochs=100):
        """
        Model training function.
        """
        model.train()
        for epoch in range(num_epochs):
            for batch_inputs, batch_outputs in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                loss.backward()
                optimizer.step()

            #print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            if (epoch + 1) % 10 == 0:
                self.save_model_params(model, save_dir)

    def save_model_params(self, model, save_dir):
        """
        Save the model weights and biases as .npy files.
        """
        os.makedirs(save_dir, exist_ok=True)

        def save_layer_params(layer, name):
            if isinstance(layer, nn.Linear):
                np.save(os.path.join(save_dir, f'{name}_weights.npy'), layer.weight.detach().cpu().numpy())
                np.save(os.path.join(save_dir, f'{name}_biases.npy'), layer.bias.detach().cpu().numpy())

        save_layer_params(model.fc1, 'layer1')
        save_layer_params(model.fc2, 'layer2')
        save_layer_params(model.fc3, 'layer3')
        save_layer_params(model.fc4, 'layer4')

