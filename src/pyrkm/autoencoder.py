import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import pickle

class Autoencoder(nn.Module):
    def __init__(self, model_name, layer_sizes, max_epochs, batch_size, lr, optimizer, loss):
        super(Autoencoder, self).__init__()

        self.name = model_name
        print('*** Initializing {}'.format(self.name))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float64)
        print('The model is working on the following device: {}'.format(self.device))        
        self.max_epochs = max_epochs
        self.lr = lr
        self.epoch = 0
        self.batch_size = batch_size
        self.bottleneck_size = layer_sizes[-1]
        # Epochs at which to store the model
        num_points = 50 
        self.t_to_save = sorted(list(set(np.round(np.logspace(np.log10(1), np.log10(self.max_epochs), num_points)).astype(int).tolist())))

        if loss=='MSE':
            self.criterion = nn.MSELoss()
        else:
            print('Loss {} not implemented'.format(loss))

        encoder_layers = []
        for i in range(len(layer_sizes) - 1):
            encoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            encoder_layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for i in range(len(layer_sizes) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i-1]))
            decoder_layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(*decoder_layers)

        if optimizer=='Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer=='SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=lr)
        else:
            print('Optimizer {} not implemented'.format(optimizer))
        
    
    def forward(self, x, k=1):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train(self, train_data, test_data, print_error=True):
        while self.epoch < self.max_epochs:
            for _, v_data in enumerate(train_data):
                start_time = time.time()
                self.optimizer.zero_grad()
                outputs = self(v_data)
                loss = self.criterion(outputs, v_data)
                loss.backward()
                self.optimizer.step()

                self.epoch += 1

                # Store the model state
                if self.epoch in self.t_to_save:
                    with open("model_states/{}_t{}.pkl".format(self.name,self.epoch), "wb") as file:
                        pickle.dump(self, file)

                if self.epoch % 100 ==0:
                    t = time.time()-start_time
                    if print_error:
                        test_out = self(test_data)
                        test_loss = self.criterion(test_out, test_data)
                        print ("Epoch: %d , Test-loss %.5g , train-loss %.5g , time: %f"%(self.epoch, test_loss, loss, t))
                    else:
                        print ("Epoch: %d , train-loss %.5g , time: %f"%(self.epoch, loss, t))

        print('*** Training finished')

    def reconstruct(self, data, k):
        data = torch.Tensor(data).to(self.device).to(torch.double)
        v_model = self(data)
        return data.detach().cpu().numpy(), v_model.detach().cpu().numpy()

    def generate(self, n_samples, k, h_binarized=False, from_visible=False):
        # How does the VAE generates?
        # what is the noise distribution we have to use?
        h = torch.rand(n_samples, self.bottleneck_size).to(self.device).to(torch.double)
        v_model = self.decoder(h)
        return v_model.detach().cpu().numpy()
