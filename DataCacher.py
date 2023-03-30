import torch
from torch.utils.data import DataLoader
import time

import threading

from .modality_type.ModBase import FrameModBase, SimpleModBase
torch.multiprocessing.set_sharing_strategy('file_system')

from .TrajBuffer import TrajBuffer
from .CacherDataset import CacherDataset, SimpleDataloader

class DataCacher(object):

    def __init__(self, modality_dict, data_splitter, data_root, num_worker, batch_size=1, load_traj=False, verbose=False):
        '''
        modality_dict: {mod_name: modality, ..} 
                        mod_name: the key of the sample dict, e.g. img0
                        object of the class under modality_type folder, e.g. rgb_lcam_front
        data_root: the root directory of the dataset
        num_worker: the number of workers
        batch_size: the batch size, 1 is best as tested on my local machine
        The sizes defined in the modalities_sizes are are the sizes required for training, 
        if the loaded data is in different shape with what defined in modalities_sizes, the data will be resized
        '''
        self.verbose = verbose

        # self.modalities_sizes = modalities_sizes
        # self.datatypes = list(self.modalities_sizes.keys())
        self.modality_dict = modality_dict
        self.modnames = list(modality_dict.keys()) # a list of strings, which are the keys of the buffer
        self.modalities = [modality_dict[kk] for kk in self.modnames]
        self.modnum = len(self.modnames)

        self.num_worker = num_worker
        self.batch_size = batch_size
        # self.cacher_dataset = cacher_dataset
        self.data_root = data_root
        if load_traj:
            self.splitter_func = data_splitter.get_next_trajectory
        else:
            self.splitter_func = data_splitter.get_next_split
        # self.data_splitter = data_splitter # split the whole dataset into subsets

        # initialize two buffers
        self.loading_buffer = None
        self.ready_buffer = None
        self.loading_a = False
        self.loading_b = False
        self.new_buffer_available = False
        self.active_mod = -1
        self.active_modname = ""
        self.mod_ind = 0
        self.dataiter = None
        # This following lines won't allocate RAM memory yet
        self.buffer_a = TrajBuffer(self.modnames, self.modalities, verbose)
        self.buffer_b = TrajBuffer(self.modnames, self.modalities, verbose)

        # initialize a dataloader
        self.stop_flag = False
        self.loading_b = False
        self.loading_a = True
        self.loading_buffer = self.buffer_a
        self.reset_buffer() # this will allocate the memory

        # run datacacher in a sperate thread
        th = threading.Thread(target=self.run)
        th.start()

    def load_simple_mod(self, modname, modality):
        simpleloader = SimpleDataloader(modality, self.loading_buffer.trajlist, self.loading_buffer.trajlenlist, 
                                        self.loading_buffer.framelist, datarootdir=self.data_root)
        startind = 0
        for k in range(len(self.loading_buffer.trajlenlist)):
            datanp = simpleloader.load_data(k)
            self.loading_buffer.insert_all_one_mode(modname, datanp, startind)
            startind += datanp.shape[0]
        assert startind == self.loading_buffer.framenum * modality.freq_mult, \
            "Load simple mod {} for {} frames, which does not match {}".format(modname, startind, self.loading_buffer.framelist * modality.freq_mult)
        self.vprint('  type {} loaded: traj {} frames {}'.format(self.active_modname, len(self.loading_buffer.trajlist), startind))
        self.loading_buffer.set_full(self.active_modname)

    def set_load_mod(self, k): 
        '''
        set the k-th modality as the active modality
        return: whether it ends up with a modality that needs to be loaded by worker
        '''
        modobj = self.modalities[k]
        self.active_mod = k
        self.active_modname = self.modnames[k]

        if isinstance(modobj, SimpleModBase):
            self.load_simple_mod(self.active_modname, modobj)
            return False

        else:
            assert isinstance(modobj, FrameModBase), "Unknow modality type {}".format(self.active_modname)
            cacher_dataset = CacherDataset(modobj, self.loading_buffer.trajlist, self.loading_buffer.trajlenlist, 
                                            self.loading_buffer.framelist, datarootdir=self.data_root)
            dataloader = DataLoader(cacher_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)#, persistent_workers=True)
            self.dataiter = iter(dataloader)
            self.modind = 0
            return True

    def reset_buffer(self):
        '''
        This function allocates the shared memory
        '''
        trajlist, trajlenlist, framelist, framenum = self.splitter_func()
        self.loading_buffer.reset(framenum, trajlist, trajlenlist, framelist)

        self.active_mod = -1
        self.active_modname = ""
        self.update_mod()

        self.new_buffer_available = False
        self.starttime = time.time()

    def switch_buffer(self):
        if (self.loading_b and self.buffer_b.full):
            # start to load buffer a
            self.loading_b = False
            self.loading_a = True
            self.loading_buffer = self.buffer_a
            self.ready_buffer = self.buffer_b

        elif self.loading_a and self.buffer_a.full:
            # switch to buffer b
            self.loading_a = False
            self.loading_b = True
            self.loading_buffer = self.buffer_b
            self.ready_buffer = self.buffer_a

        else:
            return

        self.reset_buffer()

    def __getitem__(self, index):
        return self.ready_buffer[index]

    def update_mod(self,):
        '''
        control which modality currently is working on
        '''
        while True: # loop until a FrameMod is found
            if self.active_mod+1 == self.modnum: # all modalities have been loaded
                assert self.loading_buffer.is_full, "Datacacher: the buffer is not full"
                self.new_buffer_available = True
                self.vprint('Buffer loaded: traj {}, frame {}, time {}'.format(len(self.loading_buffer.trajlist),len(self.loading_buffer), time.time()-self.starttime))
                break
            else: # load the next modality
                if self.set_load_mod(self.active_mod + 1):
                    break

    def run(self):
        # check which buffer is active
        while not self.stop_flag: # this loops forever unless the stop flag is set
            if not self.new_buffer_available:
                try:
                    sample = next(self.dataiter)
                    datanp = sample.numpy()
                    self.loading_buffer.insert_frame_one_mod(self.modind, self.active_modname, datanp)
                    self.modind += datanp.shape[0]
                except StopIteration:
                    self.vprint('  type {} loaded: traj {}, frame {}, time {}'.format(self.active_modname, len(self.loading_buffer.trajlist),len(self.loading_buffer), time.time()-self.starttime))
                    self.loading_buffer.set_full(self.active_modname)
                    self.update_mod()
            else:
                time.sleep(0.1)

    def stop_cache(self):
        self.stop_flag = True
    
    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

# python -m Datacacher.DataCacher
if __name__=="__main__":
    from .modality_type.tartanair_types import rgb_lcam_front, depth_lcam_front
    from .DataSplitter import DataSplitter
    from .utils import visflow, visdepth
    from .input_parser import parse_inputfile
    import cv2
    import numpy as np

    datafile = '/home/amigo/tmp/test_root/coalmine/analyze/data_coalmine_Data_easy_P000.txt'
    trajlist, trajlenlist, framelist, totalframenum = parse_inputfile(datafile)
    dataspliter = DataSplitter(trajlist, trajlenlist, framelist, 12)
    rgbtype = rgb_lcam_front((320, 320))
    depthtype = depth_lcam_front((320, 320))
    dataroot = "/home/amigo/tmp/test_root"

    datacacher = DataCacher({'img0':rgbtype, 'depth0':depthtype}, dataspliter, dataroot, 2, batch_size=1, load_traj=False)
     
    while not datacacher.new_buffer_available:
        print('wait for data loading...')
        time.sleep(1)
    # import ipdb;ipdb.set_trace()
    datacacher.switch_buffer()

    while True:
        for k in range(12):
            sample = datacacher[k]
            img = sample["img0"]
            # flow = sample["flow6"]
            # flowvis = visflow(flow)
            # flowvis = cv2.resize(flowvis, (0,0), fx=4, fy=4)
            depth = sample["depth0"]
            depthvis = visdepth(depth)
            # fmask = sample["fmask6"]
            # fmaskvis = (fmask>0).astype(np.uint8)*255
            # fmaskvis = np.tile(fmaskvis[:,:,np.newaxis], (1, 1, 3))
            # fmaskvis = cv2.resize(fmaskvis, (640, 480))
            disp = np.concatenate((img,depthvis), axis=1) # 
            # if flow.max()==0:
            #     print(k, 'flow zeros')
            # if fmask.max()==0:
            #     print(k, 'fmask zeros')
            cv2.imshow('img',disp)
            cv2.waitKey(0)
            # print(k, img.shape, flow.shape)
        if datacacher.new_buffer_available:
            datacacher.switch_buffer()

