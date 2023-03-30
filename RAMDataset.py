import numpy as np
from torch.utils.data import Dataset
# from .utils import make_intrinsics_layer

class RAMDataset(Dataset):
    '''
    Update: 
    1. remove task-specific things from dataloader: imu_freq, intrinsics, blxfx, random_blur
    2. code clean

    datacacher: 
    modality_dict: e.g. {"img0": rgb_lcam_front, "depth0": depth_lcam_left}
    modalities_lengths: e.g. {"img0": 2, "img1": 1, "flow": 1, "imu": 10}
    the following modalities are supported: 
    img0(img0blur),img1,disp0,disp1,depth0,depth1,flow,fmask,motion,imu,trajdir

    Note if trajdir is asked for, the trajectory's path will be returned in sample['trajdir'] for debugging purpose

    imu_freq: only useful when imu modality is queried 
    intrinsics: [w, h, fx, fy, ox, oy], used for generating intrinsics layer. No intrinsics layer added if the value is None
    blxfx: used to convert between depth and disparity

    Note it is different from the CacherDataset, flow/flow2/flow4 are not differetiated here, but controled by frame_skip
    Note only one of the flow/flow2/flow4 can be queried in one Dataset
    Note now the img sequence and the flow sequence are coorelated, which means that you can not ask for a image seq with 0 skipping while querying flow2

    When a sequence of data is required, the code will automatically adjust the length of the dataset, to make sure the every modality exists. 
    The IMU has a higher frequency than the other modalities. The frequency is imu_freq x other_freq. 

    If intrinsics is not None, a intrinsics layer is added to the sample
    The intrinsics layer will be scaled wrt the intrinsics_sclae
    '''
    def __init__(self, \
        datacacher, \
        modality_dict, \
        modalities_lengths, \
        transform = None, \
        frame_skip = 0, \
        seq_stride = 1, \
        frame_dir = False, \
        verbose = False
        ):  

        super(RAMDataset, self).__init__()
        self.datacacher = datacacher
        self.modalities_lengths = modalities_lengths
    
        self.modkeylist, self.modtypelist, self.modlenlist= [], [], []
        for k, v in self.modalities_lengths.items(): 
            assert k in modality_dict, "Missing key {} in modality_dict".format(k)
            self.modkeylist.append(k) # ["img0", "img1", "depth0", ...]
            self.modlenlist.append(v) # [1,2,1,100,...]
            self.modtypelist.append(modality_dict[k]) # [rgb_lcam_front, depth_lcam_right, ...]

        self.transform = transform

        self.frame_skip = frame_skip # sample not consequtively, skip a few frames within a sequences
        self.seq_stride = seq_stride # sample less sequence, skip a few frames between two sequences 
        self.frame_dir = frame_dir # return the trajdir and framestr if set to True

        # initialize the trajectories and figure out the seqlen
        assert datacacher.ready_buffer.full, "Databuffer in RAM is not ready! "
        self.trajlist = datacacher.ready_buffer.trajlist
        self.trajlenlist = datacacher.ready_buffer.trajlenlist 
        self.framelist = datacacher.ready_buffer.framelist
        self.dataroot = datacacher.data_root

        self.seqnumlist = self.parse_seqnum()

        self.framenumFromFile = sum(self.trajlenlist)
        self.N = sum(self.seqnumlist)
        self.trajnum = len(self.trajlenlist)
        self.acc_trajlen = [0,] + np.cumsum(self.trajlenlist).tolist()
        self.acc_seqlen = [0,] + np.cumsum(self.seqnumlist).tolist() # [0, num[0], num[0]+num[1], ..]

        self.is_epoch_complete = False # flag is set to true after all the data is sampled

        self.verbose = verbose

        self.vprint('Loaded {} sequences from the RAM, which contains {} frames...'.format(self.N, self.framenumFromFile))

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def parse_seqnum(self):
        seqnumlist = []
        for trajlen in self.trajlenlist:
            minseqnum = trajlen + 1
            for modlen, modtype in zip(self.modlenlist, self.modtypelist):
                mod_droplast = modtype.drop_last
                mod_freqmult = modtype.freq_mult
                seqnum = self.sample_num_from_traj(trajlen, self.frame_skip, self.seq_stride, 
                             modlen, mod_freqmult, mod_droplast)
                if seqnum < minseqnum:
                    minseqnum = seqnum
            seqnumlist.append(minseqnum)
        return seqnumlist

    def sample_num_from_traj(self, trajlen, skip, stride, 
                             mod_sample_len, mod_freq_mul, mod_drop_last):
        # the valid data lengh of this modality
        mod_trajlen = trajlen * mod_freq_mul - mod_drop_last
        # sequence length with skip frame 
        # e.g. x..x..x (sample_length=3, skip=2, seqlen_w_skip=1+(2+1)*(3-1)=7)
        seqlen_w_skip = (skip + 1) * mod_sample_len - skip
        mod_stride = stride * mod_freq_mul
        seqnum = int((mod_trajlen - seqlen_w_skip)/ mod_stride) + 1
        if mod_trajlen<seqlen_w_skip:
            seqnum = 0      
        return seqnum  

    def idx2trajind(self, idx):
        for k in range(self.trajnum):
            if idx < self.acc_seqlen[k+1]:
                break
        # the frame is in the k-th trajectory
        remainingframes = (idx-self.acc_seqlen[k]) * self.seq_stride
        return k, remainingframes

    def idx2slice(self, mod_freq_mult, mod_sample_len, trajind, frameind):
        '''
        handle the stride and the skip
        return: a slice object for querying the RAM
        '''

        start_frameind = self.acc_trajlen[trajind] * mod_freq_mult + frameind * self.seq_stride * mod_freq_mult
        seqlen_w_skip = (self.frame_skip + 1) * mod_sample_len - self.frame_skip
        end_frameind = start_frameind + seqlen_w_skip
        assert end_frameind - self.acc_trajlen[trajind]* mod_freq_mult <= self.trajlenlist[trajind]* mod_freq_mult, \
            "End-frameind {}, trajlen {}. Sample a sequence cross two trajectories! This should never happen! ".format( \
                end_frameind - self.acc_trajlen[trajind], self.trajlenlist[trajind]* mod_freq_mult)

        return slice(start_frameind, end_frameind, self.frame_skip+1)

    def __len__(self):
        return self.N

    def epoch_complete(self):
        return self.is_epoch_complete

    def set_epoch_complete(self):
        self.is_epoch_complete = True

    def __getitem__(self, idx):
        # import ipdb;ipdb.set_trace()
        # sample = self.datacacher[ramslice]
        sample = {}
        trajind, frameind = self.idx2trajind(idx)
        for key, modlen, modtype in zip(self.modkeylist, self.modlenlist, self.modtypelist):

        # for datatype, datalen in self.modalities_lengths.items(): 
            # parse the idx to trajstr
            mod_freqmult = modtype.freq_mult
            ramslice = self.idx2slice(mod_freqmult, modlen, trajind, frameind)
            # print(key, ramslice)
            sample[key] = self.datacacher.ready_buffer.get_frame(key, ramslice)

        if self.frame_dir:
            sample['trajdir'] = self.dataroot + '/' + self.trajlist[trajind] + '/' + self.framelist[trajind][frameind]

        # Transform.
        if ( self.transform is not None):
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    from .modality_type.tartanair_types import rgb_lcam_front, depth_lcam_front, flow_lcam_front
    from .DataSplitter import DataSplitter
    from .utils import visflow, visdepth
    from .input_parser import parse_inputfile
    from .DataCacher import DataCacher
    import cv2
    import numpy as np
    from torch.utils.data import DataLoader
    import time

    datafile = '/home/amigo/tmp/test_root/coalmine/analyze/data_coalmine_Data_easy_P000.txt'
    trajlist, trajlenlist, framelist, totalframenum = parse_inputfile(datafile)
    dataspliter = DataSplitter(trajlist, trajlenlist, framelist, 12)
    rgbtype = rgb_lcam_front((320, 320))
    depthtype = depth_lcam_front((320, 320))
    flowtype = flow_lcam_front((320, 320))
    dataroot = "/home/amigo/tmp/test_root"
    skip = 1
    stride = 1
    modality_types = {'img0':rgbtype, 'depth0':depthtype, 'flow':flowtype}
    modalities_lengths = {'img0':2, 'depth0':1, 'flow':3}
    datacacher = DataCacher({'img0':rgbtype, 'depth0':depthtype, 'flow':flowtype}, dataspliter, dataroot, 2, batch_size=1, load_traj=False)
     
    while not datacacher.new_buffer_available:
        print('wait for data loading...')
        time.sleep(1)
    # import ipdb;ipdb.set_trace()
    datacacher.switch_buffer()
    dataset = RAMDataset(datacacher, \
                modality_types, \
                modalities_lengths, \
                transform = None, \
                frame_skip = skip, seq_stride = stride)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dataiter = iter(dataloader)

    subset_repeat_count = 0
    for k in range(100):
        print('---',k,'---')
        try:
            sample = dataiter.next()
        except StopIteration:
            if datacacher.new_buffer_available:
                datacacher.switch_buffer()
                dataset = RAMDataset(datacacher, \
                            modality_types, \
                            modalities_lengths, \
                            transform = None, \
                            frame_skip = skip, seq_stride = stride)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                subset_repeat_count = -1
            dataiter = iter(dataloader)
            sample = dataiter.next()
            subset_repeat_count += 1
            print("==> Work on subset for {} time".format(subset_repeat_count))
        print(sample.keys())
        # import ipdb;ipdb.set_trace()
        ss=sample['img0'][0][0].numpy()
        ss2=sample['depth0'][0][0].numpy()
        ss3=sample['flow'][0][0].numpy()
        depthvis = visdepth(80./ss2)
        flowvis = visflow(ss3)
        disp = cv2.hconcat((ss, depthvis, flowvis))
        cv2.imshow('img', disp)
        cv2.waitKey(10)

    datacacher.stop_cache()
