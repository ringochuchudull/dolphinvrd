import torch

from dataset.dolphin import DOLPHINVIDEOVRD, get_transform

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

from model.helper.utility import cpu_or_gpu
from collections import defaultdict

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=512, output_size=9):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(15360, output_size)

        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(15360, 9)
        
        #Use CE with logit
        #self.softmax = F.softmax
        self.init_weights()
    
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)

        print(lstm_out.view(len(input_seq), -1).shape)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))

        return predictions

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)




def main():
    check = DOLPHINVIDEOVRD('../DOLPHIN/', set='Train', mode='specific', transforms=get_transform(False))
    data_loader = torch.utils.data.DataLoader(check, batch_size=1, shuffle=False)

    DEVICE = cpu_or_gpu('cuda')
    MODEL = LSTM()



    # Transformer for motion detection
    for clip, motion in data_loader:
        # Each clip has a segment size = 30
        
        # Batch Size 1
        for did, blob in motion.items():
            
            if did == 5:  # Skip motions of pip as they are unlabelled
                continue

            print(did, blob['traj'].shape, blob['motion'].shape)

            MODEL(blob['traj'])

if __name__ == '__main__':
    main()