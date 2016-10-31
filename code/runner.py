import tensorflow as tf
import getopt
import sys
import os
from glob import glob

from myalexnet_forward import AlexNet
import constants as c


def run(vid_range):
    ##
    # Setup
    ##

    sess = tf.Session()

    print 'Init model...'
    model = AlexNet()

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
    print '-r/--range=    "start_vid end_vid" (Range of videos to classify. If not specified, ' \
                          'defaults to all videos.)'

def handle_input():
    ##
    # Handle cmd line input
    ##

    vid_range = None

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'd:r:O', ['data_dir=', 'range=', 'overwrite'])
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

    return vid_range

def main():
    vid_range = handle_input()
    run(vid_range)

if __name__ == '__main__':
    main()
