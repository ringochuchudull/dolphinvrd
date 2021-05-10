## Finding Invisible and Directional Relationship between Objeccts in Tracking Videos (Provisional, Subject to change)


### How to use it
In your root directory, enter this command
```
export PYTHONPATH=$PWD
```

You should have an anaconda distribution with Python3.6+, and install the dependencies
```
pip install Requirement.txt
```
##### Inspecting 𝔻𝕆𝕃ℙℍ𝕀ℕ/🅳🅾🅻🅿🅷🅸🅽/𝘋𝘖𝘓𝘗𝘏𝘐𝘕/𝘿𝙊𝙇𝙋𝙃𝙄𝙉

To run the visualisation, download the data and follow the structure as listed in `dataset/readme.txt`. And then run the following command.
```
python helpers/visualise_video.py --data_path dataset/DOLPHIN --play_visualise yes --save_visualise_image f
```

##### Training 𝔻𝕠𝕝𝕡𝕙𝕚𝕟𝕒𝕝/🅳🅾🅻🅿🅷🅸🅽🅰🅻/𝘋𝘰𝘭𝘱𝘩𝘪𝘯𝘢𝘭/𝘿𝙤𝙡𝙥𝙝𝙞𝙣𝙖𝙡 Detector   

python helpers/train_dolphinal_detector.py --data_path dataset/DOLPHIN --device cuda

##### Notes
