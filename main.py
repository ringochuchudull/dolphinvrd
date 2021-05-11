import numpy as np
import torch, torchvision

from model.helper.parser import DolphinParser
from model.helper.dolphin_detector_train import get_transform
from model.helper.utility import git_root, cpu_or_gpu
from dataset.dolphin import DOLPHIN

import os, yaml
from tqdm import tqdm

from model.tracker.tracktor.network import FRCNN_FPN
from model.tracker.tracktor.tracktor import Tracker

def train():
    pass

def inference():
    pass

def eval():
    pass


def main():
    dp = DolphinParser()
    dp_args = dp.parse()

    try:
        dataset = DOLPHIN(data_path=dp_args.data_path,
                          set='Train',
                          mode='general',
                          transforms=get_transform(train=True))

        dataset_test = DOLPHIN(data_path=dp_args.data_path,
                               set='Test',
                               mode='general',
                               transforms=get_transform(train=False))

    except AssertionError as e:
        print('You do not have dataset in the folder')
        save_path = os.path.join(git_root(), 'dataset', 'dolphin')


        dataset = DOLPHIN(data_path=dp_args.data_path,
                          set='Train',
                          mode='general',
                          transforms=get_transform(train=True))

        dataset_test = DOLPHIN(data_path=dp_args.data_path,
                               set='Test',
                               mode='general',
                               transforms=get_transform(train=False))

    DEVICE = cpu_or_gpu(dp_args.device)

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)
    torch.backends.cudnn.deterministic = True

    print("+Initializing object detector+")

    try:
        obj_detect = FRCNN_FPN(num_classes=3)
        model_weight = os.path.join(git_root(), 'general_detector_0.pth') #'model', 'param', 'general_detector_0.pth')
        # .pth file needed

        checkpoint = torch.load(model_weight, map_location=DEVICE)
        obj_detect.load_state_dict(checkpoint['model_state_dict'])

    except (FileNotFoundError, Exception):
        obj_detect = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    obj_detect.to(DEVICE)
    obj_detect.eval()

    print("+Initializing Tracker")
    tracker = None
    with open(os.path.join(git_root(), 'model', 'tracker', 'tracktor', 'configuration.yaml'), 'r') as stream:
        try:
            configyaml = yaml.safe_load(stream)['tracktor']['tracker']
            tracker = Tracker(obj_detect, None, configyaml)

            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
            num_frames = 0
            print(f'Length of Sequence: {len(data_loader)}')
            for i, (frame, blob) in enumerate(tqdm(data_loader)):
                
                blob['img'] = frame
                with torch.no_grad():
                    tracker.step(blob, idx=i+1)

                num_frames += 1
                results = tracker.get_results()

        except yaml.YAMLError as exc:
            print(exc)

if __name__ == '__main__':
    main()