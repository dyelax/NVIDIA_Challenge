import tensorflow as tf
import getopt
import sys
import os
from glob import glob

from myalexnet_forward import AlexNet
import constants as c


def run(vid_range, weights_path):
    """
    Run a classification network on videos in vid_range.

    :param vid_range: The range of videos to classify. Applied to the sorted list of videos in
                      c.DATA_DIR.
    :param weights_path: The path to the pretrained network weights.
    """
    ##
    # Setup
    ##

    sess = tf.Session()

    print 'Init model...'
    model = AlexNet(weights_path)

    print 'Init variables...'
    sess.run(tf.initialize_all_variables())

    ##
    # Classify videos
    ##

    # get video paths from data_dir
    paths = sorted(glob(os.path.join(c.DATA_DIR, '*.mp4')))

    # if range not specified, make it the range of all videos
    if vid_range is None:
        vid_range = xrange(len(paths))

    # classify each video in the range
    for vid_num in vid_range:
        model.get_preds(sess, paths[vid_num])

def usage():
    print 'Options:'
    print '-d/--data_dir= <directory/of/video/data/>'
    print '-r/--range=    <"start_vid end_vid"> (Range of videos to classify. If not specified, ' \
                          'defaults to all videos.)'
    print '-w/--weights=  <path/to/pretrained/weights.npy> (Default = ../models/bvlc_alexnet.npy).'

def handle_input():
    ##
    # Handle cmd line input
    ##

    vid_range = None
    weights_path = '../models/bvlc_alexnet.npy'

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'd:r:w:', ['data_dir=', 'range=', 'weights='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-d', '--data_dir'):
            c.DATA_DIR = arg
        if opt in ('-r', '--range'):
            endpoints = arg.split()
            assert len(endpoints) == 2
            start, end = int(endpoints[0]), int(endpoints[1])
            vid_range = xrange(start, end)
        if opt in ('-w', '--weights'):
            weights_path = arg

    return vid_range, weights_path

def main():
    vid_range, weights_path = handle_input()
    run(vid_range, weights_path)

if __name__ == '__main__':
    main()
