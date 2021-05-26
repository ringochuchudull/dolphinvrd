import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

from dataset.dolphin import DOLPHINVIDEOVRD, get_transform

from model.helper.utility import cpu_or_gpu
from model.helper.parser import DolphinParser
from rnn import LSTM

from tqdm import tqdm


class DOLPHINTWOSTREAMNET(nn.Module):

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

    dp = DolphinParser()
    dp_args = dp.parse()

    check = DOLPHINVIDEOVRD(dp_args.data_path, set='Train', mode='specific', transforms=get_transform(False))
    data_loader = torch.utils.data.DataLoader(check, batch_size=1, shuffle=True)

    DEVICE = cpu_or_gpu(dp_args.device)
    TEMPORAL_MODEL = LSTM().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(TEMPORAL_MODEL.parameters(), lr=0.001, momentum=0.9)
    
    running_loss = 0.0
    # Single Epoch
    for _, motion in tqdm(data_loader):
        # Each clip has a segment size = 15        
        
        optimizer.zero_grad()

        try:
            # Batch Size 1 at a time
            for did, blob in motion.items():
            
                if did == 5:  # Skip motions of pipe as they are unlabelled
                    continue
                
                dense_traj = blob['traj'].to(DEVICE)
                video_clip = blob['imgsnapshot'].to(DEVICE)
                gt_class = blob['motion'].to(DEVICE)

                temporal_loss = TEMPORAL_MODEL(dense_traj)
                #loss = criterion(pred_class, torch.argmax(gt_class, dim=1))
                #loss.backward()
                #optimizer.step()

            #running_loss += loss.item()
        except Exception as e:
            print(e)

if __name__ == '__main__':
    print('Run motion detection training script')
    main()