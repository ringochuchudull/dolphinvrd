import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

from dataset.dolphin import DOLPHINVIDEOVRD, get_transform

from model.helper.utility import cpu_or_gpu
from model.helper.parser import DolphinParser
from rnn import LSTM
from s3d_resnet import s3d_resnet

from tqdm import tqdm


def make_model():
    TEMPORAL_MODEL = LSTM()
    SPATIAL_MODEL = s3d_resnet()

    return TEMPORAL_MODEL, SPATIAL_MODEL


def main():

    dp = DolphinParser()
    dp_args = dp.parse()

    dataset_train = DOLPHINVIDEOVRD(dp_args.data_path, set='Train', mode='specific', transforms=get_transform(False))
    data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)

    DEVICE = cpu_or_gpu(dp_args.device)
    TEMPORAL_MODEL, SPATIAL_MODEL = make_model()
    TEMPORAL_MODEL, SPATIAL_MODEL = TEMPORAL_MODEL.to(DEVICE), SPATIAL_MODEL.to(DEVICE)
    
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

                pred_0 = SPATIAL_MODEL(video_clip, dense_traj)
            
                loss = criterion(pred_0, torch.argmax(gt_class, dim=1))
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        except Exception as e:
            print(e)

import random
from model.helper.utility import plot_traj
import cv2

def plot():
    dp = DolphinParser()
    dp_args = dp.parse()

    dataset = DOLPHINVIDEOVRD(dp_args.data_path, set='Train', mode='specific', transforms=get_transform(False))

    n = [random.randint(1,30) for _ in range(1,10)]
    for idx in n:
        clip, traj = dataset[idx]
        frames = plot_traj(clip, traj)
        for j, f in enumerate(frames):
            cv2.imwrite(f'./imgfolder/{str(j).zfill(6)}.png', f)


if __name__ == '__main__':
    print('Run motion detection training script')
    #main()
    plot()
