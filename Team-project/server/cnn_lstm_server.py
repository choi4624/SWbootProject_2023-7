import datetime
import io
import fastapi
from fastapi import FastAPI, File, UploadFile
import multipart
import torch
import requests
import uvicorn
import tempfile

import base64
import os
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.gaussian_process.kernels import DotProduct, RBF, RationalQuadratic


app = FastAPI()



@app.post("/predict")
async def predict(file: UploadFile):
    # Read the file contents
    file_path = save_uploaded_file(file)

    # Load the data from the file
    sample_data = load_data_from_pfile(file_path)

    # Load the PyTorch model
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Declaring a variable "device" which will hold the device(i.e. either GPU or CPU) we are 
                                                                      #training the model on
    model = CNN_LSTM_FP().to(device)
    model.load_state_dict(torch.load('/home/i4624/vscode/gitclone/SWbootProject_2023-7/Team-project/model/CNN-LSTM/inout/cnn_lstm_model3.pth'))

    # Do the inference
    results = infer_model(model, file_path, device)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[10,10])
    ax.scatter(range(len(results['predictions'])), results['predictions'], c='b', marker='.', label='predictions')
    ax.legend()

    # Save the PNG file
    fig_bytes = fig.savefig('temp.png', format='png')
    encoded_image = base64.b64encode(fig_bytes.read())

    # Save the PNG file to the server directory
    filename = file.filename
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join('../output', date + '.png')
    with open(file_path, 'wb') as f:
        f.write(fig_bytes)

    # Return the prediction
    return encoded_image


def save_uploaded_file(file):
    """
    Save the uploaded file to a temporary directory.

    Args:
        file: The uploaded file.

    Returns:
        The path to the saved file.
    """
    filename = file.filename
    file_path = os.path.join(tempfile.gettempdir(), filename)
    with open(file_path, 'wb') as f:
        f.write(file.file.read())
    return file_path

    
def load_data_from_pfile(file_path):  # helper function
    """
    Load the data from the pfile.

    Args:
        file: The file containing the data.

    Returns:
        The data.
    """
    with open(file_path, 'rb') as pfile:
        sample_data = pkl.load(pfile)
    return sample_data





class PHMTestDataset_Sequential(Dataset):
    """PHM data set where each item is a sequence"""
    def __init__(self, dataset='', seq_len=5):
        """
        dataset_id = index of dataset to read from
        indices = indices from the dataset that need to be included
        seq_len = length of the output sequence
        i.e., return a sequence of length `seq_len` from the dataset as `x` and fault probability of the last frame as `y`
        """
        self.data = load_data_from_pfile(dataset)
        self.seq_len = seq_len
    
    def __len__(self):
        return self.data['x'].shape[0]-self.seq_len+1
    
    def __getitem__(self, i):
        sample = {'x': torch.from_numpy(self.data['x'][i:i+self.seq_len])}
        return sample

def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', batch_norm=True):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
    nn.init.xavier_uniform_(conv.weight)
    relu = nn.ReLU()
    if batch_norm:
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            relu
        )
    else:
        return nn.Sequential(
            conv,
            relu
        )

class CNN_CWT_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_bn_relu(2, 16, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = conv_bn_relu(16, 32, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = conv_bn_relu(32, 64, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = conv_bn_relu(64, 128, 3, stride=1, padding=1, bias=True, batch_norm=True)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, x):
        # input shape = [Nx2x128x128]
        x = self.conv1(x) # [Nx16x128x128]
        x = self.pool1(x) # [Nx16x64x64]
        x = self.conv2(x) # [Nx32x64x64]
        x = self.pool2(x) # [Nx32x32x32]
        x = self.conv3(x) # [Nx64x32x32]
        x = self.pool3(x) # [Nx64x16x16]
        x = self.conv4(x) # [Nx128x16x16]
        x = self.pool4(x) # [Nx128x8x8]
        x = self.flatten(x) # [Nx8192] {128*8*8=8192} (N => batch size, 128 => no. of channels, 8*8 => height of image*width of image)
        x = self.fc1(x) # [Nx256] 
        # x = self.dropout1(x) # apply dropout (Dropout is much harder to implement in LSTM)
        x = nn.ReLU()(x) # apply ReLU activation
        x = self.fc2(x) # [Nx128]
        x = nn.ReLU()(x) # apply ReLU activation
        return x


class CNN_LSTM_FP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CNN_CWT_Encoder()
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)
    
    def forward(self, x):
        # input shape = [N x l x 2 x 128 x 128] Here, N - batch size, l - sequence length (i.e. SEQ_LEN = 5),  2 - no. of channels or no. of filters, 
                                                      # 128 * 128 - height of an image * width of an image
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size*seq_len, C, H, W) # transform input of shape [N x l x 2 x 128 x 128] into input of shape [(Nxl) x 2 x 128 x 128]. basically,
                                                # converting(= transforming) into sequences. [(Nxl) x 2 x 128 x 128] - transformed input sequence
        x = self.encoder(x) # pass transformed input sequence through CNN Encoder, CNN Encoder converts the image input data sequence of shape 
                            # [(Nxl) x 2 x 128 x 128] into linear vector sequence by flatenning, output feature vector sequence shape = [(Nxl) x 128]
        x = x.view(batch_size, seq_len, -1) # transform encoded feature vector sequence into time distributed(= shared, alloted, assigned) input as required by
                                            # LSTM unit or LSTM cell
        x, _ = self.lstm1(x) # pass transformed encoded feature vector sequence through LSTM unit or LSTM cell, _ variable contains the hidden layers or hidden
                             # states of LSTM, we don't require those hidden layers in our implementation, therefore just stored in _ variable, and if we want
                             # we can initialize(= activate, start) hidden states from _ variable, here hidden states are the array of zeroes
        x = self.fc(x[:,-1,:]) # pass last vector sequence(i.e. output vector sequence of LSTM unit at last time step) through fully connected network layer
        x = nn.Sigmoid()(x)
        return x    



def model_inference_helper(model, dataloader, device):
    results = {'predictions':[]}
    model.eval()
    for i, batch in enumerate(dataloader):
        x = batch['x'].to(device, dtype=torch.float)

        with torch.no_grad():
            y_prediction = model(x)

        if y_prediction.size(0)>1:
            results['predictions'] += y_prediction.cpu().squeeze().tolist()
        elif y_prediction.size(0)==1:
            results['predictions'].append(y_prediction.cpu().squeeze().tolist())
    return results

def infer_model(model, file, device):
    test_dataset = PHMTestDataset_Sequential(dataset=file)

    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

    results = model_inference_helper(model, test_dataloader, device)

  

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)