# NVIDIA Challenge
Coding challenge implementation for my Spring 2017 NVIDIA deep learning internship. The task was to build a video classification pipeline using TensorFlow and output the object classification of each frame as a CSV.

## Usage:
1. Download AlexNet weights from this link: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy.
2. Place mp4 files to be classified in the `data/` directory. Feel free to keep or delete the test videos already there.
3. Run the classifier by executing `python runner.py` with the following options

        -d/--data_dir= <directory/of/video/data/>
        -r/--range=    <"start_vid_# end_vid_#"> (Range of videos to classify. If not specified, defaults to all videos.)
        -w/--weights=  <path/to/pretrained/weights.npy> (Default = ../models/bvlc_alexnet.npy).
    
4. Results of the classification will be saved in csv files in the `save/` directory.

## Design Decisions:
- Used AlexNet implementation from https://github.com/guerzh/tf_weights/blob/master/myalexnet_forward.py, pretrained on ImageNet. Made some code structure tweaks to enable easier classification.
- In case classification is interupted between videos, I included the range argument to allow the user to easily resume classification on the video where the program stopped.
- I wrote `gen_epoch()` in `utils.py` as a python generator. This reads in frames for each batch on the fly, so we wouldn't have to keep whole videos in memory at once.
    - One downside to this is that, as it's currently coded structured, reading in a batch and doing inference happen in series. Depending on the speed of reading and classifying, there could be jitter in between batches if one tried to display the video and inferences in real-time (30fps).
        - On high-end machines, this shouldn't be a problem. Running on a desktop with a NVIDIA GTX 980 Ti, I averaged 100 fps for reading in frames of 360p video and 260 fps for inference. Since both reading and inference happen so fast, the next batch could be read and classified before the first batch finishes displaying, even if they are running in series. 
        - However, it would cause jitter on lower-end machines. Running on my laptop CPU, I averaged around 19fps for reading and 28fps on inference, so we would have to wait for the next batch to be read in and classified after displaying the previous batch.
        
## Improvements:
- Given more time, I would love to explore other, more recent architectures, such as inception.
- Since reading frames with OpenCV seems to be the bottleneck, I would want to change that part of the pipeline. 
    - One thing to explore would be pre-converting videos to TFRecord format. This would enable easy parallelization for reading in batches as well as reading the next batch on the cpu while classifying the current batch on the GPU.
