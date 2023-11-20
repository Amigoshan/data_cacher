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
# python -m data_cacher.example_multidatasets --data-spec data_cacher/dataspec/sample_tartanair_random.yaml
def get_args():
    parser = argparse.ArgumentParser(description='datacacher_example')

    parser.add_argument('--data-spec', required=True, 
                        help='root directory of the data')

    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='shuffle the trajectory if there are multiple ones')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='shuffle the trajectory if there are multiple ones')
    parser.add_argument('--visualize-off', action='store_false', default=True,
                        help='visualize the data')
    
    args = parser.parse_args()
    return args
                        
def call_multi_datasets(args):
    from .MultiDatasets import MultiDatasets

    dataspec = args.data_spec
    shuffle = args.shuffle
    batch = args.batch_size
    visualize = args.visualize_off

    trainDataloader = MultiDatasets(dataspec, 
                       'local', 
                       batch=batch, 
                       workernum=0, # this is the worker number of loading data from RAM to batch, because it is usually fast, it doesn't needs a large number of workers
                       shuffle=shuffle,
                       verbose=True)

    if visualize:
        o3d_cam = join(_CURRENT_PATH, 'o3d_camera.npz')
        lidarcam = np.load(o3d_cam)

    tic = time.time()
    num = 1000 # repeat for some iterations                       
    for k in range(num):
        starttime = time.time()
        sample = trainDataloader.load_sample(notrepeat=False, fullbatch=False)
        img0 = sample["img0"] # batch x seq x 3 x h x w
        img1 = sample["img1"] # batch x seq x 3 x h x w

        depth0 = sample['depth0'] # batch x seq x h x w
        seg0 = sample['seg0'] # batch x seq x h x w
        flow = sample['flow0'] # batch x seq x 2 x h x w
        flow_mask = sample['mask0'] # batch x seq x h x w
        lidar0 = sample['lidar0'] # batch x seq x k x 3

        # import ipdb; ipdb.set_trace()
        if visualize:
            for w in range(img0.shape[0]): # iterate over batches
                # visualize the first frame in the sequence
                img0_0 = img0[w,0].numpy().transpose(1,2,0)
                img1_0 = img1[w,0].numpy().transpose(1,2,0)
                seg0_0 = seg0[w,0].numpy()
                depth0_0 = depth0[w,0].numpy()
                flow_0 = flow[w,0].numpy().transpose(1,2,0)
                flow_mask_0 = flow_mask[w,0].numpy()
                lidar0_0 = lidar0[w,0].numpy()

                disp0 = cv2.vconcat((img0_0, img1_0))
                depth0vis = visdepth(80./(depth0_0+10e-6))
                seg0vis = visseg(seg0_0)
                flowvis = visflow(flow_0, mask=flow_mask_0)
                disp1 = cv2.hconcat((depth0vis, cv2.vconcat((seg0vis, flowvis))))
                # this lidar visualization is slow
                lidarvis = vispcd(lidar0_0, vis_size = (640, 400), o3d_cam=lidarcam)
                disp = cv2.hconcat((disp0, cv2.vconcat((disp1, lidarvis))))
                cv2.imshow('img',disp)
                cv2.waitKey(1)

        # get meta-data from the sample 
        # these data might be useful when you want more controls over the dataloading process
        trajdir = sample['trajdir'] # the full path of the trajectory
        # the dataset_info returns a dictionary
        # datainfo = {'new_buffer' (boolean): true if it is the first batch of the new buffer 
        #             'epoch_count' (int): how many times the whole dataset is enumerated, 
        #             'batch_count_in_buffer' (int): how many batches has been sampled from the current buffer,
        #             'batch_count_in_epoch' (int): how many batches has been sampled in the current epoch, 
        #             'dataset_name' (string): the name of datafile for current batch, 
        #             'buffer_repeat' (int): how many times the current buffer is enumerated}
        metadata = sample['dataset_info']
        print(" # %d interation, repeat %d on the buffer, epoch %d, time %.2f, loading from %s"  % (k, metadata['buffer_repeat'], metadata['epoch_count'], time.time()-starttime, trajdir[0]))

    print("Training Complete in %.2f s"  % (time.time()-tic))
    trainDataloader.stop_cachers()    


if __name__=="__main__":
    args = get_args()
    call_multi_datasets(args)