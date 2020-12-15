'''
__author__ = 'Ringo S.W. Chu'
__Description__ = 'Covert ImgNet Video to Frames
'''

from __future__ import absolute_import, division, print_function
from ours.helper.parser import GeneralParser
import os, glob, json
import cv2
from tqdm import tqdm

def read_video(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise Exception('Cannot open {}'.format(path))
    video = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            video.append(frame)
        else:
            break
    cap.release()
    return video

def main():

    parser = GeneralParser()
    parse_aug = parser.parse()

    data_path = parse_aug.data_path
    print(f'Your path to data is: {data_path}')

    train_test_set = 'test'

    #Json location
    all_json_annotation = sorted(glob.glob(os.path.join(data_path, train_test_set ,'*.json')))

    #Save destination
    save_destination = os.path.join(data_path, train_test_set)

    #Video location
    video_location = os.path.join(data_path, '')

    for js in tqdm(all_json_annotation):

        # Reading from file
        anno = json.loads(open(js, "r").read())

        height = anno['height']
        width = anno['width']
        fps = anno['fps']
        frame_count = anno['frame_count']
        trajectories = anno['trajectories']
        subject_objects = anno['subject/objects']
        relation_instances = anno['relation_instances']

        video_id = anno['video_id']

        # Video Path
        this_video_path = os.path.join(video_location, 'videos', video_id+'.mp4')
        video_frames = read_video(this_video_path)

        assert anno['frame_count'] == len(video_frames), \
            '{} : anno {} video {}'.format(anno['video_id'], anno['frame_count'], len(video_frames))
        assert anno['width'] == video_frames[0].shape[1] and anno['height'] == video_frames[0].shape[0], \
            '{} : anno ({}, {}) video {}'.format(anno['video_id'], anno['height'], anno['width'], video_frames[0].shape)

        video_destination = os.path.join(save_destination, video_id)
        if not os.path.exists(video_destination):
            print(f'Create Directory {video_destination}')
            os.mkdir(video_destination)

        for idx, f in zip(range(1, len(video_frames)+1), video_frames):
            cv2.imwrite(os.path.join(video_destination, str(idx).zfill(6)+'.jpeg'), f)

        input()

        '''
        print(height, width)
        print(fps, frame_count)
        print(video_id)
        print('\n\n\n\n\n\n\n')
        print(subject_objects)
        print('\n\n\n\n\n\n\n')
        print(len(trajectories))
        print('\n\n\n\n\n\n\n')
        print(relation_instances)
        '''

if __name__ == '__main__':
    main()