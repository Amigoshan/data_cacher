import torch
from torch.utils.data import DataLoader
import time
from os.path import join
import threading

from .modality_type.ModBase import FrameModBase, SimpleModBase
torch.multiprocessing.set_sharing_strategy('file_system')

from .TrajBuffer import TrajBuffer
from .CacherDataset import  SimpleDataloader #CacherDataset,

import concurrent.futures

class DataCacher(object):

    def __init__(self, modalities, modkey_list, data_splitter, data_root, num_worker, batch_size=1, load_traj=False, verbose=False):
        '''
        modalities: object of the class under modality_type folder, e.g. rgb_lcam_front
        modality_dict: [[mod_key0, mod_key1, ...], ...], wehre mod_key is the key of the sample dict, e.g. img0
        {   
            mod_class_name0: [mod_key0, mod_key1, ...], 
            mod_class_name1: [mod_key0, mod_key1, ...]
        } 
        
        data_root: the root directory of the dataset
        num_worker: the number of workers
        batch_size: the batch size, 1 is best as tested on my local machine
        The sizes defined in the modalities_sizes are are the sizes required for training, 
        if the loaded data is in different shape with what defined in modalities_sizes, the data will be resized
        '''
        self.verbose = verbose

        assert len(modalities) == len(modkey_list), "DataCacher: Modality number {} and modkey number {} mismatch!".format(\
            len(modalities), len(modkey_list))
        self.modkey_list = modkey_list #[modality_dict[kk] for kk in modality_dict]
        self.modalities = modalities # [get_modality_type(mm) for mm in mod_type_names]
        # self.modnum = len(modalities)

        self.num_worker = num_worker
        self.batch_size = batch_size
        self.data_root = data_root
        if load_traj:
            self.splitter_func = data_splitter.get_next_trajectory
        else:
            self.splitter_func = data_splitter.get_next_split

        # initialize two buffers
        self.loading_buffer = None
        self.ready_buffer = None
        self.loading_a = False
        self.loading_b = False
        self.new_buffer_available = False
        # self.active_mod = -1
        # self.active_modkeys = ""
        # self.mod_ind = 0
        # self.dataiter = None
        # This following lines won't allocate RAM memory yet
        self.buffer_a = TrajBuffer(self.modkey_list, self.modalities, verbose)
        self.buffer_b = TrajBuffer(self.modkey_list, self.modalities, verbose)
        self.filelist = None

        # initialize a dataloader
        self.stop_flag = False
        self.loading_b = False
        self.loading_a = True
        self.loading_buffer = self.buffer_a
        self.reset_buffer() # this will allocate the memory

        # run datacacher in a sperate thread
        th = threading.Thread(target=self.run)
        th.start()

    def insert_datalist(self, ind, modnamelist, datalist):
        for modname, data in zip(modnamelist, datalist):
            self.loading_buffer.insert_frame_one_mod(ind, modname, data)

    def load_simple_mod(self, modkeys, modality):
        simpleloader = SimpleDataloader(modality, self.loading_buffer.trajlist, self.loading_buffer.trajlenlist, 
                                        self.loading_buffer.framelist, datarootdir=self.data_root)
        startind = 0
        for k in range(len(self.loading_buffer.trajlenlist)):
            datanp_list = simpleloader.load_data(k)
            assert len(datanp_list) == len(modkeys), \
                'DataCacher: Data number {} and key number {} mismatch!'.format(len(datanp_list), len(modkeys))
            
            for modkey, datanp in zip(modkeys, datanp_list):
                self.loading_buffer.insert_all_one_mode(modkey, datanp, startind)

            startind += datanp.shape[0]
            
        assert startind == self.loading_buffer.framenum * modality.freq_mult, \
            "DataCacher: Load simple mod {} for {} frames, which does not match {}".format(modkeys, startind, 
                                                                                            self.loading_buffer.framenum * modality.freq_mult)
        self.vprint('  simple type {} loaded: traj {} frames {}'.format(modkeys, len(self.loading_buffer.trajlist), startind))
        self.loading_buffer.set_full(modkeys)

    def process_filelist(self, trajlist, framelist):
        '''
        return: 1. the relative dir of trajectory 
                2. the frame string 
                3. is the frame at the end of the current trajectory (for loading flow)
        '''
        image_files = []
        for trajstr, frames in zip(trajlist, framelist):
            for k,framestr in enumerate(frames):
                image_files.append( (trajstr, framestr, len(frames)-k) )
        return image_files

    def reset_buffer(self):
        '''
        This function allocates the shared memory
        1. call data_splitter to get next batch
           get new_epoch==True at the beginning of each new epoch
        2. start to load the next frame-based modality
        '''
        trajlist, trajlenlist, framelist, framenum, new_epoch = self.splitter_func()
        self.loading_buffer.reset(framenum, trajlist, trajlenlist, framelist)
        self.filelist = self.process_filelist(trajlist, framelist)

        # self.active_mod = -1
        # self.active_modkeys = ""
        # self.update_mod()

        self.new_buffer_available = False
        self.starttime = time.time()

        return new_epoch

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
            assert False, "DataCacher: Unknow buffer state loading_a {}, loading_b {}".format(self.loading_a, self.loading_b)

        new_epoch = self.reset_buffer()
        return new_epoch

    def __getitem__(self, index):
        return self.ready_buffer[index]

    # def set_load_mod(self, k): 
    #     '''
    #     set the k-th modality as the active modality
    #     return: whether it ends up with a modality that needs to be loaded by worker
    #     '''
    #     modobj = self.modalities[k]
    #     self.active_mod = k
    #     self.active_modkeys = self.modkey_list[k]

    #     if isinstance(modobj, SimpleModBase):
    #         self.load_simple_mod(self.active_modkeys, modobj)
    #         return False

    #     elif isinstance(modobj, FrameModBase):
    #         cacher_dataset = CacherDataset(modobj, self.loading_buffer.trajlist, self.loading_buffer.trajlenlist, 
    #                                         self.loading_buffer.framelist, datarootdir=self.data_root)
    #         dataloader = DataLoader(cacher_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)#, persistent_workers=True)
    #         self.dataiter = iter(dataloader)
    #         self.modind = 0
    #         return True
    #     else:
    #         assert False, "DataCacher: Unknow modality type {}".format(self.active_modkeys)


    # def update_mod(self,):
    #     '''
    #     control which modality currently is working on
    #     '''
    #     while True: # loop until a FrameMod is found
    #         if self.active_mod+1 == self.modnum: # all modalities have been loaded
    #             assert self.loading_buffer.is_full(), "Datacacher: the buffer is not full"
    #             self.new_buffer_available = True
    #             self.vprint('==> Buffer loaded: traj {}, frame {}, time {}'.format(len(self.loading_buffer.trajlist),len(self.loading_buffer), time.time()-self.starttime))
    #             break
    #         else: # load the next modality
    #             if self.set_load_mod(self.active_mod + 1):
    #                 break

    def load_new_buffer(self,):
        # import ipdb;ipdb.set_trace()
        for modobj, modkeys in zip(self.modalities, self.modkey_list): 
            if isinstance(modobj, SimpleModBase):
                self.load_simple_mod(modkeys, modobj)
            elif isinstance(modobj, FrameModBase):
                modstarttime = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_worker) as executor:
                    # Submit tasks to the executor
                    future_to_index = {executor.submit(modobj.load_data, join(self.data_root,trajdir), 
                                        framestr, remainidx): k for k, (trajdir, framestr, remainidx) in enumerate(self.filelist)}

                    # Iterate over completed tasks
                    for future in concurrent.futures.as_completed(future_to_index):
                        data_index = future_to_index[future]
                        try:
                            data_array_list = future.result()
                            assert(len(modkeys) == len(data_array_list)), \
                                'sped keys {} for {} do not match the data returned'.format(modkeys, modobj.name)
                                
                        except Exception as exc:
                            self.vprint(f"Failed to load image {data_index}: {exc}")
                        else:
                            for datanp, modkey in zip(data_array_list, modkeys):
                                self.loading_buffer.insert_frame_one_mod(data_index, modkey, datanp[np.newaxis,...])

                        if self.stop_flag:
                            return

                    if all(future.done() for future in future_to_index):
                        self.loading_buffer.set_full(modkeys)
                        self.vprint('  type {} loaded: traj {}, frame {}, time {}'.format( \
                            modkeys, len(self.loading_buffer.trajlist),len(self.loading_buffer), time.time()-modstarttime))
                    else:
                        self.vprint("Not all tasks have completed yet.")
            else:
                assert False, "DataCacher: Unknow modality type {}".format(modkeys)
        
        assert self.loading_buffer.is_full(), "Datacacher: the buffer is not full!"
        self.new_buffer_available = True
        self.vprint('==> Buffer loaded: traj {}, frame {}, time {}'.format( \
            len(self.loading_buffer.trajlist),len(self.loading_buffer), time.time()-self.starttime))

    def run(self):
        # check which buffer is active
        while not self.stop_flag: # this loops forever unless the stop flag is set
            if not self.new_buffer_available:
                # load modalities one by one
                self.load_new_buffer()

                # try:
                #     sample = next(self.dataiter)
                #     assert len(sample) == len(self.active_modkeys), \
                #         "DataCacher: Data number {} and key number {} mismatch!".format(len(sample), len(self.active_modkeys))
                    
                #     for modkey, data in zip(self.active_modkeys, sample):
                #         datanp = data.numpy()
                #         self.loading_buffer.insert_frame_one_mod(self.modind, modkey, datanp)

                #     self.modind += datanp.shape[0]
                # except StopIteration:
                #     self.vprint('  type {} loaded: traj {}, frame {}, time {}'.format(self.active_modkeys, len(self.loading_buffer.trajlist),len(self.loading_buffer), time.time()-self.starttime))
                #     self.loading_buffer.set_full(self.active_modkeys)
                #     self.update_mod()
            else:
                time.sleep(0.1)

    def stop_cache(self):
        self.stop_flag = True
    
    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

# python -m Datacacher.DataCacher
if __name__=="__main__":
    from .modality_type.tartanair_types import rgb_lcam_back, pose_lcam_bottom, flow_lcam_front
    from .DataSplitter import DataSplitter
    from .datafile_editor import read_datafile
    from .utils import visflow
    import cv2
    import numpy as np

    datafile = 'data_cacher/data/local_test.txt'
    dataroot = '/peru/tartanairv2'
    trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)
    dataspliter = DataSplitter(trajlist, trajlenlist, framelist, 100)
    rgbtype = rgb_lcam_back([(320, 320)])
    posetype = pose_lcam_bottom([7])
    flowtype = flow_lcam_front([(320,320),(320,320)])

    datacacher = DataCacher([rgbtype, flowtype, posetype], 
                            [['img0'], ['flow', 'mask'], ['pose']], 
                            dataspliter, 
                            dataroot, 8, batch_size=1, 
                            load_traj=False, verbose=True)
     
    while not datacacher.new_buffer_available:
        print('wait for data loading...')
        time.sleep(1)
    # import ipdb;ipdb.set_trace()
    datacacher.switch_buffer()

    while True:
        for k in range(100):
            sample = datacacher[k]
            img = sample["img0"].transpose(1,2,0)
            flow = sample["flow"].transpose(1,2,0)
            flowvis = visflow(flow)
            # flowvis = cv2.resize(flowvis, (0,0), fx=4, fy=4)

            fmask = sample["mask"]
            fmaskvis = (fmask>0).astype(np.uint8)*255
            fmaskvis = cv2.applyColorMap(fmaskvis, cv2.COLORMAP_JET)

            disp = np.concatenate((img,flowvis,fmaskvis), axis=1) # 
            # if flow.max()==0:
            #     print(k, 'flow zeros')
            # if fmask.max()==0:
            #     print(k, 'fmask zeros')
            cv2.imshow('img',disp)
            cv2.waitKey(10)
            # print(k,sample["pose"])
        if datacacher.new_buffer_available:
            datacacher.switch_buffer()

