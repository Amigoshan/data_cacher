# python -m Datacacher.DataCacher
import argparse
import cv2
import numpy as np
import time

from .utils import visdepth, visflow, visseg, vispcd
import os
from os.path import join

_CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

# example command: 
# python -m data_cacher.example_datacacher --data-root $DATA_ROOT --data-file data_cacher/data/data_coalmine.txt

def get_args():
    parser = argparse.ArgumentParser(description='datacacher_example')
    parser.add_argument('--data-root', required=True, 
                        help='root directory of the data')
    parser.add_argument('--data-file', required=True, 
                        help='root directory of the data')

    parser.add_argument('--buffer-framenum', type=int, default=100, 
                        help='how many frames are loaded for the buffer')
    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='shuffle the trajectory if there are multiple ones')
    parser.add_argument('--visualize-off', action='store_false', default=True,
                        help='visualize the data')
    
    args = parser.parse_args()
    return args
                        
def call_data_cacher(args):
    from .modality_type.tartanair_types import image_lcam_front, image_rcam_equirect, depth_lcam_fish, seg_lcam_fish, flow_lcam_front, lidar
    from .DataSplitter import DataSplitter
    from .datafile_editor import read_datafile
    from .DataCacher import DataCacher

    datafile = args.data_file
    dataroot = args.data_root
    buffersize = args.buffer_framenum
    shuffle = args.shuffle
    visualize = args.visualize_off

    # parse datafile
    trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)

    # split the whole dataset, with 12 frames in each buffer
    # framenum should be as big as what your RAM can hold
    # typically 1k - 10k
    dataspliter = DataSplitter(trajlist, trajlenlist, framelist, 
                                framenum= buffersize, 
                                shuffle= shuffle) # this parameter only take effect if you have multiple trajectories in the datafile

    # we can load as many modality as we want
    rgbtype = image_lcam_front([(640, 640)])
    panorgbtype = image_rcam_equirect([(320,640)])
    fishdepthtype = depth_lcam_fish([(560,360)])
    fishsegtype = seg_lcam_fish([(280,280)])
    # note that the flow is one frame less than other modalities in the trajecotry
    # the last sample in a trajectory will be filled with a blank data 
    # you can figure out which frame is blank by looking at the trajlenlist
    flowtype = flow_lcam_front([(280,280),(280,280)])
    lidartype = lidar([(57600, 3)])

    if visualize:
        o3d_cam = join(_CURRENT_PATH, 'o3d_camera.npz')
        lidarcam = np.load(o3d_cam)

    datacacher = DataCacher(modalities=[rgbtype, panorgbtype, fishdepthtype, fishsegtype, flowtype, lidartype], 
                            modkey_list=[['img0'], ['img1'],['depth0'],['seg0'], ['flow0', 'mask0'], ['lidar0']], 
                            data_splitter= dataspliter, 
                            data_root= dataroot, 
                            num_worker= 2, 
                            batch_size=1, 
                            load_traj=False, 
                            verbose=True)
     
    while not datacacher.new_buffer_available:
        print('wait for data loading...')
        time.sleep(1)
    datacacher.switch_buffer()
    iter_count = 0
    repeat_count = 0

    while True:
        for k in range(buffersize):
            iter_count += 1
            starttime = time.time()
            sample = datacacher[k]
            img0 = sample["img0"].transpose(1,2,0)
            img1 = sample["img1"].transpose(1,2,0)

            depth0 = sample['depth0']
            seg0 = sample['seg0']
            flow = sample['flow0'].transpose(1,2,0)
            flow_mask = sample['mask0']
            lidar0 = sample['lidar0']

            if visualize:
                disp0 = cv2.vconcat((img0, img1))
                depth0vis = visdepth(80./(depth0+10e-6))
                seg0vis = visseg(seg0)
                flowvis = visflow(flow, mask=flow_mask)
                disp1 = cv2.hconcat((depth0vis, cv2.vconcat((seg0vis, flowvis))))
                # this lidar visualization is slow
                lidarvis = vispcd(lidar0, vis_size = (640, 400), o3d_cam=lidarcam)
                disp = cv2.hconcat((disp0, cv2.vconcat((disp1, lidarvis))))
                cv2.imshow('img',disp)
                cv2.waitKey(1)

            print(" # %d interation, repeat %d: loss %.2f, time %.2f" % (iter_count, repeat_count, np.random.rand(), time.time()-starttime))
            
            # training code starts
            # do something fantastic
            # end of training code 

            # import ipdb;ipdb.set_trace()
            
        repeat_count += 1
        if datacacher.new_buffer_available:
            datacacher.switch_buffer()
            print(" *** Buffer switched.. ".format(k, np.random.rand(), time.time()-starttime))
            repeat_count = 0


if __name__=="__main__":
    args = get_args()
    call_data_cacher(args)
