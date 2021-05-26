import torch.nn as nn
import torch
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=512, output_size=9):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(15360, output_size)
        self.drop = nn.Dropout(p=0.5)
        self.label = nn.Linear(hidden_layer_size, output_size)
        self.init_weights()
    
    def attention_net(self, lstm_output, final_state):

    	hidden = final_state.squeeze(0)
    	attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
    	soft_attn_weights = F.softmax(attn_weights, 1)
    	new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
    
    	return new_hidden_state

    def forward(self, input_seq):
        #print(input_seq.shape)
        output, (final_hidden_state, final_cell_state) = self.lstm(input_seq)
        print(final_hidden_state.size(), output.size())
        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(attn_output)
        
        print(logits.shape)
        return logits

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


