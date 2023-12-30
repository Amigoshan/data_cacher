# python -m Datacacher.DataCacher
import argparse
import cv2
import numpy as np
import time

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
                       'psc', 
                       batch=batch, 
                       workernum=0, # this is the worker number of loading data from RAM to batch, because it is usually fast, it doesn't needs a large number of workers
                       shuffle=shuffle,
                       verbose=True)

    tic = time.time()
    num = 1000 # repeat for some iterations
    for k in range(num):
        starttime = time.time()
        sample = trainDataloader.load_sample(notrepeat=False, fullbatch=False)
       
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