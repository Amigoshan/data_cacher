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
        self.subsetrepeat = [0,] * self.datasetNum # keep a track of how many times the subset is sampled

        self.init_datasets(dataconfigs)

    def parse_modality_types(self, modality_param):
        modality_types, modality_lengths = {}, {}
        for modkey, modparam in modality_param.items():
            modtype_class = get_modality_type(modparam['type'])
            modality_types[modkey] = modtype_class(modparam['cacher_size'])
            modality_lengths[modkey] = modparam['length']
        return modality_types, modality_lengths

    def init_datasets(self, dataconfigs):
        # modalities = dataconfigs['modalities']
        for datafile, params in dataconfigs['data'].items():
            self.datafiles.append(datafile)
            modality_param = params['modality']
            cacher_param = params['cacher']
            dataset_param = params['dataset']

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
            dataset_param = params['dataset']
            self.datasetparams.append(dataset_param)

        self.accDataLens = np.cumsum(self.datalens).astype(np.float64)/np.sum(self.datalens)    

        # wait for all datacacher being ready
        for k, datacacher in enumerate(self.datacachers):
            while not datacacher.new_buffer_available:
                time.sleep(1)
            self.datacachers[k].switch_buffer()
            dataset = RAMDataset(self.datacachers[k], \
                                 self.modalitytypes[k], \
                                 self.modalitylengths[k], \
                                 **self.datasetparams[k],
                                 verbose=self.verbose
                                )
            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            self.datasets[k] = dataset
            self.dataloaders[k] = dataloader
            self.dataiters[k] = iter(dataloader)
            self.subsetrepeat[k] = 0

    def load_sample(self, fullbatch=True, notrepeat=False):
        # Randomly pick the dataset in the list
        randnum = np.random.rand()
        datasetInd = 0 
        new_buffer = False
        while randnum > self.accDataLens[datasetInd]: # 
            datasetInd += 1

        # load sample from the dataloader
        try:
            sample = next(self.dataiters[datasetInd])
            if sample[list(sample.keys())[0]].shape[0] < self.batch and (fullbatch is True): # the imcomplete batch is thrown away
                # self.dataiters[datasetInd] = iter(self.dataloaders[datasetInd])
                # sample = self.dataiters[datasetInd].next()
                sample = next(self.dataiters[datasetInd])
        except StopIteration:
            # import ipdb;ipdb.set_trace()
            if notrepeat: # wait for the new buffer ready, do not repeat the current buffer
                while not self.datacachers[datasetInd].new_buffer_available:
                    time.sleep(1.0)
                    self.vprint('Wait for the next buffer...')
            if self.datacachers[datasetInd].new_buffer_available : 
                self.datacachers[datasetInd].switch_buffer()
                self.datasets[datasetInd] = RAMDataset(self.datacachers[datasetInd], \
                                                       self.modalitytypes[datasetInd], \
                                                       self.modalitylengths[datasetInd], \
                                                       **self.datasetparams[datasetInd]
                                                      )
                self.dataloaders[datasetInd] = DataLoader(self.datasets[datasetInd], batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
                self.subsetrepeat[datasetInd] = -1
            self.dataiters[datasetInd] = iter(self.dataloaders[datasetInd])
            sample = next(self.dataiters[datasetInd])
            new_buffer = True
            self.subsetrepeat[datasetInd] += 1
            self.vprint('==> Working on {} for the {} time'.format(self.datafiles[datasetInd], self.subsetrepeat[datasetInd]))
        sample['new_buffer'] = new_buffer
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
    dataset_specfile = 'data_cacher/dataspec/flowvo_train_local_v2.yaml'
    # configparser = ConfigParser()
    # dataconfigs = configparser.parse_from_fp(dataset_specfile)
    batch = 3
    trainDataloader = MultiDatasets(dataset_specfile, 
                       'local', 
                       batch=batch, 
                       workernum=0, 
                       shuffle=False)
    tic = time.time()
    num = 100                       
    for k in range(num):
        sample = trainDataloader.load_sample()
        print(sample.keys())
        # time.sleep(0.02)
        # import ipdb;ipdb.set_trace()
        for b in range(batch):
            ss=sample['img0'][b][0].numpy()
            ss2=sample['depth0'][b][0].numpy()
            ss3=sample['flow'][b][0].numpy()
            depthvis = visdepth(80./ss2)
            flowvis = visflow(ss3)
            disp = cv2.hconcat((ss, depthvis, flowvis))
            cv2.imshow('img', disp)
            cv2.waitKey(100)

    print((time.time()-tic))
    trainDataloader.stop_cachers()