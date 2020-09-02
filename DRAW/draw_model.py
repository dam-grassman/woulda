import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as f

import numpy as np

"""
The equation numbers on the comments corresponding
to the relevant equation given in the paper:
DRAW: A Recurrent Neural Network For Image Generation.
"""

# Define some constants
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

H, W, channel = 7, 7, 7
# Dictionary storing network parameters.
params = {
    'T' : 50,# Number of glimpses.
    'batch_size': 4,# Batch size.
    'A' : W,# Image width
    'B': H,# Image height
    'z_size' : 8,# Dimension of latent space.
    'read_N' : W,# N x N dimension of reading glimpse.
    'write_N' : H,# N x N dimension ofmy writing glimpse.
    'dec_size': 32,# Hidden dimension for decoder.
    'enc_size' : 32,# Hidden dimension for encoder.
    'epoch_num': 50,# Number of epochs to train for.
    'learning_rate': 3e-4, # Learning rate.
    'beta1': 0.5,
    'clip': 5.0,
    'save_epoch' : 10,# After how many epochs to save checkpoints and generate test output.
    'channel' : channel, # Number of channels for image.(3 for RGB, etc.)
    'conv':True}


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        #print('COnv')
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.Gates = nn.Conv2d(input_size , hidden_size, kernel_size, padding=self.padding)
        #print('INPUT', input_size, hidden_size)
    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        #print('BACTH and Spacial',batch_size,  spatial_size)
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )
            #print('STATE SIZE', state_size)
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        #print('Shape', input_.shape, prev_hidden.shape)
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        #print('Gates', gates.shape)
        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        #print('Gates', in_gate.shape, remember_gate.shape, out_gate.shape, cell_gate.shape )
        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)
        #print(remember_gate.size(), prev_cell.size(), in_gate.size(), cell_gate.size() )
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell

class DRAWModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.T = params['T']
        self.Z = 8
        self.A = params['A']
        self.B = params['B']
        self.z_size = params['z_size']
        self.read_N = params['read_N']
        self.write_N = params['write_N']
        self.enc_size = params['enc_size']
        self.dec_size = params['dec_size']
        self.device = params['device']
        self.channel = params['channel']
        self.conv = params['conv']

        # Stores the generated image for each time step.
        self.cs = [0] * self.Z
        
        # To store appropriate values used for calculating the latent loss (KL-Divergence loss)
        self.logsigmas = [0] * self.Z
        self.sigmas = [0] * self.Z
        self.mus = [0] * self.Z
        #print(self.read_N,self.read_N,self.channel, self.dec_size , self.enc_size)
        if self.conv == True:
            print('Conv LSTM as encoder')
            self.encoder = ConvLSTMCell(self.channel+self.enc_size, 4 *self.enc_size, 3)
            self.fc_mu = nn.Linear(self.enc_size*self.A*self.B, self.z_size)
            self.fc_sigma = nn.Linear(self.enc_size*self.A*self.B, self.z_size)

        else : 
            self.encoder = nn.LSTMCell(2*self.read_N*self.read_N*self.channel + self.dec_size, self.enc_size)

        # To get the mean and standard deviation for the distribution of z.
            self.fc_mu = nn.Linear(self.enc_size, self.z_size)
            self.fc_sigma = nn.Linear(self.enc_size, self.z_size)

        self.decoder = nn.LSTMCell(self.z_size, self.dec_size)

        self.fc_write = nn.Linear(self.dec_size, self.write_N*self.write_N*self.channel)

        # To get the attention parameters. 5 in total.
        self.fc_attention = nn.Linear(self.dec_size, 5)

    def forward(self, x):
        self.batch_size = x.size(0)

        # (batch_size, T, channel, widgth, heigth)
        #x = x.permute(0,1,4,2,3)

        # requires_grad should be set True to allow backpropagation of the gradients for training.
        h_enc_prev = torch.zeros((self.batch_size,self.enc_size, self.A, self.B), requires_grad=True, device=self.device)
        enc_state = torch.zeros((self.batch_size,self.enc_size, self.A, self.B), requires_grad=True, device=self.device)
        

        for t in range(self.T):

            # Equation 4.
            r_t = x[:,(self.T-1)-t,:]
            
            # Equation 5.
            h_enc, enc_state = self.encoder(r_t, (h_enc_prev, enc_state))
                        
            h_enc_prev = h_enc
        #print(h_enc.shape)

        h_dec_prev = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)
        dec_state = torch.zeros(self.batch_size, self.dec_size, requires_grad=True, device=self.device)

        for t in range(self.Z):

            c_prev = torch.zeros((self.batch_size,self.channel, self.B,self.A), requires_grad=True, device=self.device) if t == 0 else self.cs[t-1]
            
            # Equation 6.
            z, self.mus[t], self.logsigmas[t], self.sigmas[t] = self.sampleQ(h_enc.view(self.batch_size, -1))
            
            # Equation 7.
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))

            # Equation 8.
            self.cs[t] = c_prev + self.write(h_dec).view((self.batch_size,self.channel, self.B,self.A))
            #print(t, self.cs[t].shape)

            h_enc_prev = h_enc
            h_dec_prev = h_dec

    def write(self, h_dec):
        # No attention
        return self.fc_write(h_dec)

    def sampleQ(self, h_enc):
        e = torch.randn(self.batch_size, self.z_size, device=self.device)

        # Equation 1.
        mu = self.fc_mu(h_enc)
        # Equation 2.
        log_sigma = self.fc_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        
        z = mu + e * sigma

        return z, mu, log_sigma, sigma

    def filterbank(self, gx, gy, sigma_2, delta, N, epsilon=1e-8):
        grid_i = torch.arange(start=0.0, end=N, device=self.device, requires_grad=True,).view(1, -1)
        
        # Equation 19.
        mu_x = gx + (grid_i - N / 2 - 0.5) * delta
        # Equation 20.
        mu_y = gy + (grid_i - N / 2 - 0.5) * delta

        a = torch.arange(0.0, self.A, device=self.device, requires_grad=True).view(1, 1, -1)
        b = torch.arange(0.0, self.B, device=self.device, requires_grad=True).view(1, 1, -1)

        mu_x = mu_x.view(-1, N, 1)
        mu_y = mu_y.view(-1, N, 1)
        sigma_2 = sigma_2.view(-1, 1, 1)

        # Equations 25 and 26.
        Fx = torch.exp(-torch.pow(a - mu_x, 2) / (2 * sigma_2))
        Fy = torch.exp(-torch.pow(b - mu_y, 2) / (2 * sigma_2))

        Fx = Fx / (Fx.sum(2, True).expand_as(Fx) + epsilon)
        Fy = Fy / (Fy.sum(2, True).expand_as(Fy) + epsilon)

        if self.channel == 3:
            Fx = Fx.view(Fx.size(0), 1, Fx.size(1), Fx.size(2))
            Fx = Fx.repeat(1, 3, 1, 1)
            
            Fy = Fy.view(Fy.size(0), 1, Fy.size(1), Fy.size(2))
            Fy = Fy.repeat(1, 3, 1, 1)

        return Fx, Fy

    def loss(self, x, x_true):
        self.forward(x)

        criterion = nn.BCELoss()
        x_recon = torch.sigmoid(self.cs[-1])
        #x_recon = torch.softmax(self.cs[-1], dim=1)
        # Reconstruction loss.
        # Only want to average across the mini-batch, hence, multiply by the image dimensions.
        Lx = criterion(x_recon, x_true) * self.A * self.B * self.channel
        # Latent loss.
        Lz = 0

        """
        for t in range(self.Z-1, self.Z):

            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]

            kl_loss = torch.sum(mu_2 + sigma_2 - 2*logsigma, 1)*0.5 - 0.5*self.Z
            Lz += kl_loss
        """
        #Lz = torch.mean(Lz)
        net_loss = Lx# + Lz

        return net_loss

    def generate(self, num_output):
        self.batch_size = num_output
        h_dec_prev = torch.zeros(num_output, self.dec_size, device=self.device)
        dec_state = torch.zeros(num_output, self.dec_size  , device=self.device)

        for t in range(self.Z):
            c_prev = torch.zeros(self.batch_size, self.B*self.A*self.channel, device=self.device) if t == 0 else self.cs[t-1]
            z = torch.randn(self.batch_size, self.z_size, device=self.device)
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

        imgs = []

        for img in self.cs:
            # The image dimesnion is B x A (According to the DRAW paper).
            img = img.view(-1, self.B, self.A,  self.channel)
            imgs.append(vutils.make_grid(torch.sigmoid(img).detach().cpu(), nrow=int(np.sqrt(int(num_output))), padding=1, normalize=True, pad_value=1))

        return imgs


import time
import os

def train_draw(model, optimizer, params, dataloader_train, dataloader_test, folder_checkpoint = 'DRAW/checkpoint'):

    losses = []
    iters = 0
    avg_loss = 0
    device = params['device']

    folder = '{0}/{1}'.format(folder_checkpoint, params['T'])

    if not os.path.exists(folder):
        os.makedirs(folder)

    print("-"*25)
    print("Starting Training Loop...\n")
    print("-"*25)

    start_time = time.time()

    for epoch in range(0, params['epoch_num']):
        epoch_start_time = time.time()
        
        for i, data_batch in enumerate(dataloader_train, 0):
            data = torch.tensor(data_batch[:,1:,:,:,:], dtype = torch.float32)
            data_tr = torch.tensor(data_batch[:,0], dtype = torch.float32).to(device)
            
            # Get batch size.
            bs = data.shape[0]
            data = data.to(device)
            optimizer.zero_grad()

            # Calculate the loss.
            loss = model.loss(data, data_tr)
            loss_val = loss.cpu().data.numpy()
            avg_loss += loss_val

            # Calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])

            # Update parameters.
            optimizer.step()
            # Check progress of training.
            if i != 0 and i%(min(100, len(dataloader_train)-1)) == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                    % (epoch+1, params['epoch_num'], i, len(dataloader_train), avg_loss / min(100, len(dataloader_train))))
                avg_loss = 0
            losses.append(loss_val)
            iters += 1

        if dataloader_test : 
            avg_loss_test = 0
            model.eval()
            for i, data_batch in enumerate(dataloader_test, 0):
                data = torch.tensor(data_batch[:,1:,:,:,:], dtype = torch.float32).to(device) 
                data_tr = torch.tensor(data_batch[:,0], dtype = torch.float32).to(device)        
                loss = model.loss(data, data_tr)
                loss_val = loss.cpu().data.numpy()
                avg_loss_test += loss_val
            model.train()
            print('Test Loss : epoch %d\tLoss: %.4f'
                        % (epoch+1, avg_loss_test / len(dataloader_test)))
        avg_loss = 0
        epoch_time = time.time() - epoch_start_time

        print("Time Taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
        # Save checkpoint and generate test output.
        if (epoch+1) % params['save_epoch'] == 0:
            torch.save({
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'params' : params
                }, '{0}/model_epoch_{1}'.format(folder, epoch+1))

    training_time = time.time() - start_time
    print("-"*50)
    print('Training finished!\nTotal Time for Training: %.2fm' %(training_time / 60))
    print("-"*50)