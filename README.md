# NVIDIA_Challenge
Coding challenge for 2017 NVIDIA internship

## Usage:
1. Download AlexNet weights from this link: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy. Save in `models/bvlc_alexnet.npy`.
2. Place mp4 files to be classified in the `data/` directory. Feel free to keep or delete the test videos already there.
3. Run the classifier by executing `python runner.py` with the following options

    -d/--data_dir= <directory/of/video/data/>
    -r/--range=    "start_vid end_vid" (Range of videos to classify. If not specified, defaults to all videos.)
    
4. Results of the classification will be saved in csv files in the `save/` directory.

Notes:

- Used pretrained AlexNet implementation from https://github.com/guerzh/tf_weights/blob/master/myalexnet_forward.py (attributed in myalexnet_forward.py), with some tweaks
- Used generator so wouldn't have to keep whole videos in memory at once
    - Same logic for writing CSV in batches
    - downside to this is that there would be jitter in between batches if trying to display video in real-time
        - avg around 19fps for reading in frames of 360p video.
        - problem is right now that reading in a batch and doing inference happen in series
- Given more time 
    - would explore other architectures, e.g. inception.
    - since reading frames is bottleneck, would explore pre-converting videos to TFRecord to enable parallelization and overlapped reading/processing
- Added range argument to enable resuming inference if interrupted

