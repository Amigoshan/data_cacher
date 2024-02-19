import os
from torch.utils.data import DataLoader
import numpy as np
from os.path import isfile, split
import time
from .data_roots import *
from .modality_type.ModBase import get_modality_type
from .DataSplitter import DataSplitter
from .DataCacher import DataCacher
from .RAMDataset import RAMDataset
from .ConfigParser import ConfigParser
from .datafile_editor import read_datafile
import torch 
import numbers

class MultiDatasets(object):
    '''
    This class reads the specfile and create a wrapper dataset that can combine all the data files
    '''
    def __init__(self, dataset_specfile, 
                       platform, 
                       batch, 
                       workernum, 
                       shuffle=True, 
                       verbose=False):
        '''
        dataconfigs: 'modality', 'cacher', 'transform', 'dataset'
        shuffle: if the load_traj is set to true, two things will happen, 
                 the multiple datacacher will be called in sequential order
                 in each datacacher, the data will be returned in sequential order

        '''

        configparser = ConfigParser()
        if isinstance(dataset_specfile, str):
            assert isfile(dataset_specfile), "MultiDatasetsBase: Cannot find spec file {}".format(dataset_specfile)
            dataconfigs = configparser.parse_from_fp(dataset_specfile)

        elif isinstance(dataset_specfile, dict):
            dataconfigs = configparser.parse_from_dict(dataset_specfile)

        self.datasetNum = len(dataconfigs['data'])

        self.platform = platform
        self.batch = batch
        self.workernum = workernum
        self.shuffle = shuffle
        self.verbose = verbose

        self.datafiles = []
        self.datasetnames = []
        self.datacachers = [ ] 
        self.datasets = [None, ] * self.datasetNum
        self.dataloaders = [None, ] * self.datasetNum
        self.dataiters = [None, ] * self.datasetNum
        self.datasetlens = []

        self.datalens = [] # the framenum in each sub_dataset
        self.datasetparams = [] # the parameters used to create dataset
        self.modalitylengths = [] # the modality_length used to create the dataset
        self.modalityfreqs = [] # the modality frequency used to create the dataset
        self.modalitydroplast = [] # the modality drop_last used to create the dataset
        self.modalitytypes = [] 
        self.paramparams = [] # dataset parameters such as camera intrinsics
        self.subsetrepeat = [0,] * self.datasetNum # keep a track of how many times the subset is sampled

        self.current_dataset_ind = 0 # keep track of the dataset being loaded in the case of sequential way of data loading (shuffle = False)
        self.new_epoch_loading_buffer = [False, ] * self.datasetNum
        self.epoch_count = [0, ] * self.datasetNum
        self.batch_count_in_buffer = [-1, ] * self.datasetNum
        self.batch_count_in_epoch = [-1, ] * self.datasetNum

        self.first_buffer_flag = [True, ] * self.datasetNum
        self.init_datasets(dataconfigs)

    def parse_modality_types(self, modality_param):
        '''
        modality_dict: 
        {   
            mod_class_name0: 
            {
                mod_key0: 
                {
                    cacher_size: [w, h]
                    length: k
                }
                mod_key1: 
                {
                    cacher_size: [w, h]
                    length: k
                }
            }
            mod_class_name1: 
            {
                ...
            }
        }    
        The modality types are initialized here  
        return 
            Used by data_cacher: 
                - modality_objs: [mod_obj, ...] 
                - modality_keys: [[key0, key1], ...]
            Used by the RanDataset:
                - modality_length_dict: {modkey: modlen, ...}   
                - modality_freq_mult: {modkey: freq_mult, ...}
                - modality_drop_last: {modkey: drop_last, ...}
        '''
        modality_objs, modality_keys = [], []
        modality_length_dict, modality_freq_mult, modality_drop_last = {}, {}, {}
        for modtype_name, modparam in modality_param.items():
            modtype_class = get_modality_type(modtype_name)
            mod_shapes = [modparam[kk]['cacher_size'] for kk in modparam]
            mod_obj = modtype_class(mod_shapes) # create a mod type
            modality_objs.append(mod_obj)

            modkeys = list(modparam.keys())
            modality_keys.append(modkeys)

            mod_lengths = [modparam[kk]['length'] for kk in modparam]

            for modkey, modlen in zip(modkeys, mod_lengths):
                modality_length_dict[modkey] = modlen
                modality_freq_mult[modkey] = mod_obj.freq_mult
                modality_drop_last[modkey] = mod_obj.drop_last

        return modality_objs, modality_keys, modality_length_dict, modality_freq_mult, modality_drop_last

    def update_dataloader(self, k):
        '''
        the dataset/loader/iter needs to be updated, when the datacacher buffer is switched
        '''
        dataset = RAMDataset(self.datacachers[k], \
                            self.modalitylengths[k], \
                            self.modalityfreqs[k], \
                            self.modalitydroplast[k], \
                            **self.datasetparams[k], \
                            verbose=self.verbose, \
                            )
        dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
        self.datasets[k] = dataset
        self.dataloaders[k] = dataloader
        self.dataiters[k] = iter(dataloader)

    def init_datasets(self, dataconfigs):
        # modalities = dataconfigs['modalities']
        for datafileind, params in dataconfigs['data'].items():
            datafile = params['file']
            self.datafiles.append(datafile)
            datasetname = split(datafile)[-1].split('.')[0] # use the datafile name as the dataset name
            self.datasetnames.append(datasetname)
            modality_param = params['modality']
            cacher_param = params['cacher']
            dataset_param = params['dataset']
            parameter_param = params['parameter']

            # Allow to pass a data root path directly via config.
            # TODO(yoraish): is this a good idea? I feel like overrides may be dangerous.
            if 'data_root_path_override' in cacher_param and cacher_param['data_root_path_override'] is not None:
                # Check path integrity. Check if the path is to a directory.
                assert os.path.exists(cacher_param['data_root_path_override']), "MultiDatasets: Cannot find data root path provided as override {}".format(cacher_param['data_root_path_override'])
                data_root = cacher_param['data_root_path_override']

            else:
                data_root_key = cacher_param['data_root_key']
                data_root = DataRoot[self.platform][data_root_key]

            modality_objs, modality_keys, modality_lengths, modality_freq_mult, modality_drop_last = self.parse_modality_types(modality_param)
            self.modalitylengths.append(modality_lengths)
            self.modalityfreqs.append(modality_freq_mult)
            self.modalitydroplast.append(modality_drop_last)
            self.modalitytypes.append(modality_objs)

            trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)
            self.datasetlens.append(totalframenum)
            subsetframenum = cacher_param['subset_framenum']
            self.datalens.append(subsetframenum)
            data_splitter = DataSplitter(trajlist, trajlenlist, framelist, subsetframenum, shuffle=True) 
            
            workernum = cacher_param['worker_num']
            load_traj = cacher_param['load_traj'] if 'load_traj' in cacher_param else False
            datacacher = DataCacher(modality_objs, modality_keys, data_splitter, data_root, workernum, batch_size=1, load_traj=load_traj, verbose = self.verbose) # TODO: test if batch_size here matters
            self.datacachers.append(datacacher)

            # parameters for the RAMDataset
            self.datasetparams.append(dataset_param)
            # this is the parameters returned in each sample
            self.paramparams.append(parameter_param)

        self.accDataLens = np.cumsum(self.datalens).astype(np.float64)/np.sum(self.datalens)

        # wait for all datacacher being ready
        for k, datacacher in enumerate(self.datacachers):
            while not datacacher.new_buffer_available:
                time.sleep(1)
            self.new_epoch_loading_buffer[k] = self.datacachers[k].switch_buffer()
            self.update_dataloader(k)

    def load_sample(self, fullbatch=True, notrepeat=False, maxrepeatnum=3):
        '''
        fullbatch: set to true to discard the imcomplete batch at the end of the epoch
        notrepeat: set to true if you don't want to repeat training on the ready buffer
                    the code will just wait there until the next buffer is filled
        maxrepeatnum: set the maximum repeat number to avoid overfitting on current buffer too much
        '''
        if self.shuffle:
            # Randomly pick the dataset in the list
            randnum = np.random.rand()
            datasetind = 0 
            while randnum > self.accDataLens[datasetind]: # 
                datasetind += 1
        else: # load the data in sequential order, stick to the current dataset until all the data is loaded
            datasetind = self.current_dataset_ind

        new_buffer = False 
        # load sample from the dataloader
        try:
            sample = next(self.dataiters[datasetind])
            
            if sample[list(sample.keys())[0]].shape[0] < self.batch and (fullbatch is True): # the imcomplete batch is thrown away
                sample = next(self.dataiters[datasetind])

            if self.first_buffer_flag[datasetind]: # set flag for the first batch 
                new_buffer = True
                self.first_buffer_flag[datasetind] = False

        except StopIteration:
            # The current buffer is completed, 
            # We have two options, 
            # 1) move to the next buffer 
            #    when moving to next buffer, we have two options
            #       i. in random mode, the same dataset will be used for sampling the next buffer
            #       ii. in the sequential mode, the next buffer will be sampled from the next dataset
            # 2) repeat on the current buffer 
            new_buffer = True
            self.batch_count_in_buffer[datasetind] = -1
            
            if not self.shuffle: # sequential loading, switch to the next buffer in the same dataset or the first buffer of the next dataset
                while not self.datacachers[datasetind].new_buffer_available:
                    time.sleep(1.0) # in sequential way, we always want the next buffer be loaded
                    self.vprint('  Wait for the next buffer...')
                    # import ipdb;ipdb.set_trace()
                self.new_epoch_loading_buffer[datasetind] = self.datacachers[datasetind].switch_buffer()

                if self.new_epoch_loading_buffer[datasetind]: # sequential loading, the buffer just been loaded is a new epoch
                    self.epoch_count[datasetind] += 1
                    self.batch_count_in_epoch[datasetind] = -1
                    self.current_dataset_ind = (self.current_dataset_ind + 1) % self.datasetNum # current buffer is a new epoch, go to the next dataset

                self.update_dataloader(datasetind)
                self.subsetrepeat[datasetind] = 0
                datasetind = self.current_dataset_ind

            else: # in random order, both datasets and subsets should be shuffled
                if notrepeat or self.subsetrepeat[datasetind] > maxrepeatnum: # wait for the new buffer ready, do not repeat the current buffer
                    while not self.datacachers[datasetind].new_buffer_available:
                        time.sleep(1.0)
                        self.vprint('  Wait for the next buffer...')

                if self.new_epoch_loading_buffer[datasetind]:
                    self.epoch_count[datasetind] += 1
                    self.batch_count_in_epoch[datasetind] = -1
                    
                if self.datacachers[datasetind].new_buffer_available : # switch to the next buffer
                    self.new_epoch_loading_buffer[datasetind] = self.datacachers[datasetind].switch_buffer()
                    self.update_dataloader(datasetind)
                    self.subsetrepeat[datasetind] = 0
                else: # repeat the current buffer 
                    self.dataiters[datasetind] = iter(self.dataloaders[datasetind])
                    self.subsetrepeat[datasetind] += 1

            sample = next(self.dataiters[datasetind])
            self.vprint('==> Working on {} for the {} time'.format(self.datafiles[datasetind], self.subsetrepeat[datasetind]))

        self.batch_count_in_buffer[datasetind] += 1
        self.batch_count_in_epoch[datasetind] += 1

        datainfo = {'new_buffer': new_buffer, 
                    'epoch_count': self.epoch_count[datasetind], 
                    'batch_count_in_buffer': self.batch_count_in_buffer[datasetind],
                    'batch_count_in_epoch': self.batch_count_in_epoch[datasetind], 
                    'dataset_name': self.datasetnames[datasetind], 
                    'buffer_repeat': self.subsetrepeat[datasetind]}

        sample['dataset_info'] = datainfo
        params = self.paramparams[datasetind]
        for param, value in params.items():
            if isinstance(value, numbers.Number):
                sample[param] = value
            elif isinstance(value, list):
                sample[param] = torch.Tensor(value).float()

        # print("sample time: {}".format(time.time()-cachertime))
        return sample

    def stop_cachers(self):
        for datacacher in self.datacachers:
            datacacher.stop_cache()

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

if __name__ == '__main__':

    def vis_intrinsics(intrinsics):
        dispintrinsics = intrinsics.cpu().numpy().transpose(1,2,0) 
        dispintrinsics = np.clip(dispintrinsics * 255, 0, 255).astype(np.uint8)
        dispintrinsics = np.concatenate((dispintrinsics, np.zeros((intrinsics.shape[1],intrinsics.shape[2],1),dtype=np.uint8)), axis=2)
        return dispintrinsics

    # ===== Test MultiDatasets ======
    # from .utils import visflow, tensor2img
    import time
    from .ConfigParser import ConfigParser
    from .utils import visflow, visdepth
    import cv2
    from .modality_type.tartandrive_types import get_vis_costmap, get_vis_heightmap
    # dataset_specfile = 'data_cacher/dataspec/flowvo_train_local_v1.yaml'
    # dataset_specfile = 'data_cacher/dataspec/flowvo_train_local_v2.yaml'
    # dataset_specfile = 'data_cacher/dataspec/test_yorai.yaml'
    # dataset_specfile = '/home/wenshan/workspace/pytorch/geometry_vision/specs/dataspec/flowvo_train_local_v2.yaml'
    # dataset_specfile = '/home/wenshan/workspace/pytorch/geometry_vision/specs/trajspec/flowvo_euroc.yaml'
    # dataset_specfile = '/home/wenshan/workspace/pytorch/geometry_vision/specs/trajspec/flowvo_kitti.yaml'
    dataset_specfile = 'data_cacher/dataspec/test_tartandrive.yaml'
    # configparser = ConfigParser()
    # dataconfigs = configparser.parse_from_fp(dataset_specfile)
    batch = 3
    trainDataloader = MultiDatasets(dataset_specfile, 
                       'psc', 
                       batch=batch, 
                       workernum=0, 
                       shuffle=False,
                       verbose=True)
    tic = time.time()
    num = 23201                       
    for k in range(num):
        sample = trainDataloader.load_sample(notrepeat=True, fullbatch=False)
        print(k, sample['trajdir'], sample.keys())
        print(sample['dataset_info'])
        # time.sleep(0.02)
        # import ipdb;ipdb.set_trace()
        for b in range(len(sample['trajdir'])):
            # ss=sample['img0'][b][0].numpy()
            # ss = np.repeat(ss[...,np.newaxis], 3, axis=2)
            # ss2=sample['img1'][b][0].numpy().transpose(1,2,0)
            # ss3=sample['heightmap'][b][0].numpy().transpose(1,2,0)
            # ss3 = get_vis_heightmap(ss3)
            ss4=sample['costmap'][b][0].numpy()
            ss4=get_vis_costmap(ss4)
            # disp = cv2.vconcat((ss3, ss4))
            cv2.imshow('img', ss4)
            cv2.waitKey(0)

    print((time.time()-tic))
    trainDataloader.stop_cachers()
