#!/usr/bin/env python
# coding: utf-8

# In[17]:


# Inside tacotron2.py

import torch
import torch.nn as nn

class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        #Define your Tacotron-2 model architecture
        self.encoder = nn.LSTM(input_size=80, hidden_size=256, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=256, hidden_size=80, num_layers=1, batch_first=True)
        self.fc = nn.Linear(80, 1)

    def forward(self, x):
        # Define the forward pass of your Tacotron-2 model
        encoder_output, _ = self.encoder(x)
        decoder_input = torch.zeros_like(x[:, :1, :])

        for _ in range(x.size(1)):
            decoder_output, _ = self.decoder(decoder_input)
            output = self.fc(decoder_output)
            decoder_input = output

        return output

    def save_model(self, checkpoint_path):
        # Save only the state dictionary
        torch.save({'state_dict': self.state_dict()}, checkpoint_path)

    def load_model(self, checkpoint_path):
        # Load only the state dictionary
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        self.load_state_dict(state_dict)
        self.eval()


# In[ ]:




