import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math, os

from dataset.dolphin import DOLPHINVIDEOVRD, get_transform

from model.helper.utility import cpu_or_gpu, git_root
from model.helper.parser import DolphinParser
from rnn import LSTM
from s3d_resnet import s3d_resnet

from sklearn.metrics import classification_report

#from tensorboardX import SummaryWriter


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
    SPATIAL_MODEL = SPATIAL_MODEL.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(SPATIAL_MODEL.parameters(), lr=0.001, momentum=0.9)
    
    epoch_start = 1
    #writer = SummaryWriter('runs/')
    #with SummaryWriter(comment='SPATIAL_MODEL') as w:
    #    w.add_graph(SPATIAL_MODEL)
    try:
        modelparam = torch.load(os.path.join(git_root(),'model','param','motiondetect.pth'), map_location=DEVICE)    
        SPATIAL_MODEL.load_state_dict(modelparam['model_state_dict'])
        optimizer.load_state_dict(modelparam['optimizer_state_dict'])
        epoch_start = modelparam['epoch']
    except:
        print('Model Paramaters do not exist, starting a new model')

    running_loss = 0.0
    # Single Epoch

    dl_len = len(data_loader)

    for epoch in range(epoch_start, 12):
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
                    #writer.add_graph(SPATIAL_MODEL, (video_clip, dense_traj))

                    loss = criterion(pred_0, torch.argmax(gt_class, dim=1))
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()/dl_len
                #writer.add_scalar('Accuracy/train', running_loss, 0)    

            except Exception as e:
                print(e, '\nSaving model dictionary due to Key board interupt or else')
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': SPATIAL_MODEL.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, os.path.join(git_root(), 'model', 'param', f'motiondetect.pth'))

            # Training accuracy
            
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': SPATIAL_MODEL.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, os.path.join(git_root(), 'model', 'param', f'motiondetect.pth'))

import random
from model.helper.utility import plot_traj
import cv2

def plot():
    dp = DolphinParser()
    dp_args = dp.parse()

    dataset = DOLPHINVIDEOVRD(dp_args.data_path, set='Train', mode='specific', transforms=get_transform(False))

    #n = [random.randint(0, 1) for _ in range(1, len(dataset))]
    n = [0, 1]
    for idx in n:
        
        if not os.path.exists(os.path.join('imgfolder', f'{str(idx).zfill(6)}')):
            os.mkdir(os.path.join('imgfolder', f'{str(idx).zfill(6)}'))
        
        clip, traj = dataset[idx]
        frames = plot_traj(clip, traj)
        for j, f in tqdm(enumerate(frames)):
            #cv2.imwrite(f'./imgfolder/{str(j).zfill(6)}.png', f)
            cv2.imwrite(os.path.join('imgfolder', f'{str(idx).zfill(6)}', f'{str(j).zfill(6)}.png'), f)

if __name__ == '__main__':
    print('Run motion detection training script')
    main()
    #plot()
