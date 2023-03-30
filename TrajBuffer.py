from .RAMBuffer import RAMBufferBase
from .modality_type.ModBase import get_modality_type 

class TrajBuffer(object):
    '''
    Store the multi-modal data in the form of trajectories
    keylist: a list of names that will be used as the key of returned dict
    modtypelist: a list of modality_type objects
    mod_names: list of strings
    mod_datatypes: list of data types
    mod_sizes: list of data sizes
    '''
    def __init__(self, keylist, modtypelist, verbose=False):
        assert len(keylist) == len(modtypelist), "The keylist length {} and modlist length {} don't match".format(len(keylist), len(modtypelist))
        self.mod_names = keylist
        self.modtypelist = modtypelist
        self.mod_datatypes = []
        self.mod_sizes = []

        self.buffer = {}
        self.full = {}
        for modname, modtype in zip(self.mod_names, self.modtypelist):
            self.mod_datatypes.append(modtype.data_type)
            self.mod_sizes.append(modtype.data_shape)
            self.buffer[modname] = RAMBufferBase(modtype.data_type, verbose = verbose)
            self.full[modname] = False

        self.trajlist, self.trajlenlist, self.framelist = [],[],[]
        self.framenum = 0

    def reset(self, framenum, trajlist, trajlenlist, framelist):
        self.trajlist, self.trajlenlist, self.framelist = trajlist, trajlenlist, framelist
        for mod_size, mod_name, modality in zip(self.mod_sizes, self.mod_names, self.modtypelist):
            freq_mult = modality.freq_mult
            self.buffer[mod_name].reset((framenum*freq_mult,) + mod_size)
            self.full[mod_name] = False
        self.framenum = framenum

    # deprecated
    def insert_frame(self, index, sample):
        '''
        deprecated
        sample: a dictionary
        {
            'mod0': n x h x w x c,
            'mod1': n x h x w x c, 
            ...
        }
        '''
        sample_mod = sample.keys()
        for mod in sample_mod:
            assert mod in self.mod_names, "Error: mod {} not in datatype when inserting".format(mod)
            datanp = sample[mod].numpy() # n x h x w x c
            for k in range(datanp.shape[0]):
                self.buffer[mod].insert(index+k, datanp[k])

    # deprecated
    def insert_all(self, sample):
        '''
        sample: a dictionary
        {
            'mod0': n x h x w x c,
            'mod1': n x h x w x c, 
            ...
        }
        '''
        sample_mod = sample.keys()
        for mod in sample_mod:
            assert mod in self.mod_names, "Error: mod {} not in datatype when inserting".format(mod)
            datanp = sample[mod].numpy() # n x h x w x c
            self.buffer[mod].load(datanp)

    def insert_frame_one_mod(self, index, modname, sample):
        '''
        sample: an numpy array n x h x w x c
        '''
        assert modname in self.mod_names, "Error: mod {} not in datatype when inserting".format(modname)
        for k in range(sample.shape[0]):
            self.buffer[modname].insert(index+k, sample[k])

    def insert_all_one_mode(self, modname, sample, startind):
        '''
        sample: an array n x h x w x c
        '''
        assert modname in self.mod_names, "Error: mod {} not in datatype when inserting".format(modname)
        self.buffer[modname].load(sample, startind)

    def is_full(self):
        for mod in self.mod_names:
            if self.full[mod] == False:
                return False
        return True

    def set_full(self, mod):
        assert mod in self.mod_names, "Error: mod {} not in datatype when set_full".format(mod)
        self.full[mod] = True

    def __len__(self):
        return self.framenum

    def __getitem__(self, index):
        '''
        Note this function won't copy the data
        so do not modify this data! 
        '''
        sample = {}
        for mod in self.mod_names:
            sample[mod] = self.buffer[mod][index]
        return sample

    def get_frame(self, mod, index):
        return self.buffer[mod][index]

if __name__=="__main__":
    from .modality_type.tartanair_types import rgb_lcam_front, depth_lcam_front
    from .input_parser import parse_inputfile
    from .CacherDataset import CacherDataset
    import torch

    rgbtype = rgb_lcam_front((320, 320))
    depthtype = depth_lcam_front((320, 320))
    datafile = '/home/amigo/tmp/test_root/coalmine/analyze/data_coalmine_Data_easy_P000.txt'
    trajlist, trajlenlist, framelist, totalframenum = parse_inputfile(datafile)

    buffer = TrajBuffer(['img0', 'depth0'], [rgbtype, depthtype])
    buffer.reset(50, trajlist, trajlenlist, framelist)

    dataset0 = CacherDataset(rgbtype, trajlist, trajlenlist, framelist, datarootdir="/home/amigo/tmp/test_root")
    dataset1 = CacherDataset(depthtype, trajlist, trajlenlist, framelist, datarootdir="/home/amigo/tmp/test_root")
    for k in range(50):
        ss=torch.from_numpy(dataset0[k]).unsqueeze(0)
        ss2=torch.from_numpy(dataset1[k]).unsqueeze(0)

        buffer.insert_frame_one_mod(k, 'img0', ss)
        buffer.insert_frame_one_mod(k, 'depth0', ss2)
        # depthvis = visdepth(80./ss2)
        # disp = cv2.hconcat((ss, depthvis))
        # cv2.imshow('img', disp)
        # cv2.waitKey(0)
    print(buffer.is_full())
    buffer.set_full('img0')
    buffer.set_full('depth0')
    print(buffer.is_full())