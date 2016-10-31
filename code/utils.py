import cv2
import numpy as np
from time import time

import constants as c

def process_frame(frame):
    #normalize
    processed_frame = frame - np.mean(frame)

    #crop
    height, width, channels = frame.shape
    offset = (width - height) / 2.
    processed_frame = processed_frame[:, int(np.floor(offset)):-int(np.ceil(offset)), :]

    #resize
    processed_frame = cv2.resize(processed_frame, (c.FRAME_SIZE, c.FRAME_SIZE), interpolation=cv2.INTER_AREA)

    return processed_frame

def gen_epoch(vid_path, batch_size=32):
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        read_start = time()

        batch = np.zeros([batch_size, c.FRAME_SIZE, c.FRAME_SIZE, 3])
        for i in xrange(batch_size):
            ret, frame = cap.read()

            if ret:
                batch[i] = process_frame(frame)
            else:
                # clip the last batch and exit the loop
                batch = batch[:i]
                break


        yield batch

        read_time = time() - read_start
        read_fps = len(batch) / read_time
        print 'Frame read fps: %f' % read_fps

        if not ret:
            break

    cap.release()
