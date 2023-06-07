import os
from torch.utils.data import DataLoader
import numpy as np
from os.path import isfile
import time
from .data_roots import *
from .modality_type.ModBase import get_modality_type
from .DataSplitter import DataSplitter
from .DataCacher import DataCacher
from .RAMDataset import RAMDataset
from .ConfigParser import ConfigParser
from .input_parser import parse_inputfile
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
        self.datacachers = [ ] 
        self.datasets = [None, ] * self.datasetNum
        self.dataloaders = [None, ] * self.datasetNum
        self.dataiters = [None, ] * self.datasetNum

        self.datalens = [] # the framenum in each sub_dataset
        self.datasetparams = [] # the parameters used to create dataset
        self.modalitylengths = [] # the modality_length used to create the dataset
        self.modalitytypes = [] 
        self.paramparams = [] # dataset parameters such as camera intrinsics
        self.subsetrepeat = [0,] * self.datasetNum # keep a track of how many times the subset is sampled

        self.current_dataset_ind = 0 # keep track of the dataset being loaded in the case of sequential way of data loading (shuffle = False)
        self.new_epoch = [False, ] * self.datasetNum
        self.epoch_count = [0, ] * self.datasetNum

        self.init_datasets(dataconfigs)

    def parse_modality_types(self, modality_param):
        modality_types, modality_lengths = {}, {}
        for modkey, modparam in modality_param.items():
            modtype_class = get_modality_type(modparam['type'])
            modality_types[modkey] = modtype_class(modparam['cacher_size'])
            modality_lengths[modkey] = modparam['length']
        return modality_types, modality_lengths

    def update_dataloader(self, k):
        '''
        the dataset/loader/iter needs to be updated, when the datacacher buffer is switched
        '''
        dataset = RAMDataset(self.datacachers[k], \
                            self.modalitytypes[k], \
                            self.modalitylengths[k], \
                            **self.datasetparams[k], \
                            verbose=self.verbose, \
                            )
        dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
        self.datasets[k] = dataset
        self.dataloaders[k] = dataloader
        self.dataiters[k] = iter(dataloader)
        self.subsetrepeat[k] = 0

    def init_datasets(self, dataconfigs):
        # modalities = dataconfigs['modalities']
        for datafileind, params in dataconfigs['data'].items():
            datafile = params['file']
            self.datafiles.append(datafile)
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

            modality_types, modality_lengths = self.parse_modality_types(modality_param)
            self.modalitylengths.append(modality_lengths)
            self.modalitytypes.append(modality_types)

            trajlist, trajlenlist, framelist, framenum = parse_inputfile(datafile)
            subsetframenum = cacher_param['subset_framenum']
            self.datalens.append(subsetframenum)
            data_splitter = DataSplitter(trajlist, trajlenlist, framelist, subsetframenum, shuffle=True) 
            
            workernum = cacher_param['worker_num']
            load_traj = cacher_param['load_traj'] if 'load_traj' in cacher_param else False
            datacacher = DataCacher(modality_types, data_splitter, data_root, workernum, batch_size=1, load_traj=load_traj, verbose = self.verbose) # TODO: test if batch_size here matters
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
            self.new_epoch[k] = self.datacachers[k].switch_buffer()
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
        else: # load the data in sequential order
            datasetind = self.current_dataset_ind

        new_buffer = False
        # load sample from the dataloader
        try:
            sample = next(self.dataiters[datasetind])
            if sample[list(sample.keys())[0]].shape[0] < self.batch and (fullbatch is True): # the imcomplete batch is thrown away
                sample = next(self.dataiters[datasetind])
        except StopIteration:
            new_buffer = True

            if not self.shuffle: # sequential loading
                while not self.datacachers[datasetind].new_buffer_available:
                    time.sleep(1.0) # in sequential way, we always want the next buffer be loaded
                if self.new_epoch[datasetind]: # sequential loading, new epoch coming
                    self.current_dataset_ind = (self.current_dataset_ind + 1) % self.datasetNum
                self.new_epoch[datasetind] = self.datacachers[datasetind].switch_buffer()
                self.update_dataloader(datasetind)
                datasetind = self.current_dataset_ind

            else: # in random order, both datasets and subsets should be shuffled
                if notrepeat or self.subsetrepeat[datasetind] > maxrepeatnum: # wait for the new buffer ready, do not repeat the current buffer
                    while not self.datacachers[datasetind].new_buffer_available:
                        time.sleep(1.0)
                        self.vprint('  Wait for the next buffer...')

                if self.datacachers[datasetind].new_buffer_available : 
                    self.new_epoch[datasetind] = self.datacachers[datasetind].switch_buffer()
                    self.update_dataloader(datasetind)
                else:
                    self.dataiters[datasetind] = iter(self.dataloaders[datasetind])

            sample = next(self.dataiters[datasetind])
            self.subsetrepeat[datasetind] += 1
            self.vprint('==> Working on {} for the {} time'.format(self.datafiles[self.current_dataset_ind], self.subsetrepeat[self.current_dataset_ind]))

        sample['new_buffer'] = new_buffer
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
    # dataset_specfile = 'data_cacher/dataspec/flowvo_train_local_v1.yaml'
    # dataset_specfile = 'data_cacher/dataspec/flowvo_train_local_v2.yaml'
    # dataset_specfile = 'data_cacher/dataspec/test_yorai.yaml'
    # dataset_specfile = '/home/wenshan/workspace/pytorch/geometry_vision/specs/dataspec/flowvo_train_local_v2.yaml'
    # dataset_specfile = '/home/wenshan/workspace/pytorch/geometry_vision/specs/trajspec/flowvo_euroc.yaml'
    dataset_specfile = '/home/wenshan/workspace/pytorch/geometry_vision/specs/trajspec/flowvo_kitti.yaml'
    # configparser = ConfigParser()
    # dataconfigs = configparser.parse_from_fp(dataset_specfile)
    batch = 1
    trainDataloader = MultiDatasets(dataset_specfile, 
                       'local', 
                       batch=batch, 
                       workernum=0, 
                       shuffle=False,
                       verbose=True)
    tic = time.time()
    num = 23201                       
    for k in range(num):
        sample = trainDataloader.load_sample(notrepeat=True)
        print(k, sample['trajdir'])
        # time.sleep(0.02)
        # import ipdb;ipdb.set_trace()
        # for b in range(batch):
            # ss=sample['img0'][b][0].numpy().transpose(1,2,0)
            # ss2=sample['depth0'][b][0].numpy()
            # ss3=sample['flow'][b][0].numpy().transpose(1,2,0)
            # depthvis = visdepth(80./ss2)
            # flowvis = visflow(ss3)
            # disp = cv2.hconcat((ss, depthvis, flowvis))
            # cv2.imshow('img', flowvis)
            # cv2.waitKey(10)

    print((time.time()-tic))
    trainDataloader.stop_cachers()