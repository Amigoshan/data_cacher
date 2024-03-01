from .RAMBuffer import RAMBufferBase

class TrajBuffer(object):
    '''
    One modality can provide multiple types of data. 
    E.g. flow modality returns flow and flow_mask two modalities that require two buffers
    Store the multi-modal data in the form of trajectories

    keylist: a list of names that will be used as the key of returned dict
    modtypelist: a list of modality_type objects, each type cooresponds to a list of keys
    
    mod_names: list of list of strings, 
    mod_datatypes: list of list of data types
    mod_sizes: list of list of data sizes
    '''
    def __init__(self, keylist, modtypelist, verbose=False):
        assert len(keylist) == len(modtypelist), "The keylist length {} and modlist length {} don't match".format(len(keylist), len(modtypelist))
        # self.mod_namelist = keylist
        # self.modtypelist = modtypelist

        self.mod_names = []
        self.mod_datatypes = []
        self.mod_sizes = []
        self.mod_freq = []

        self.buffer = {}
        self.full = {}
        for modnames, modtype in zip(keylist, modtypelist):
            datatype_list = modtype.data_types
            datashape_list = modtype.data_shapes

            assert len(datatype_list) == len(datashape_list) and len(datatype_list) == len(modnames), \
                "Data number do not match! {} data types {} data shapes, {} mod names".format( \
                    len(datatype_list), len(datashape_list), len(modnames))

            for datatype, datashape, modname in zip (datatype_list, datashape_list, modnames):
                self.mod_datatypes.append(datatype)
                self.mod_sizes.append(datashape)
                self.mod_names.append(modname)
                self.mod_freq.append(modtype.freq_mult)

                self.buffer[modname] = RAMBufferBase(datatype, verbose = verbose)
                self.full[modname] = False

        self.trajlist, self.trajlenlist, self.framelist = [],[],[]
        self.framenum = 0

    def reset(self, framenum, trajlist, trajlenlist, framelist):
        self.trajlist, self.trajlenlist, self.framelist = trajlist, trajlenlist, framelist
        for mod_size, mod_name, freq_mult in zip(self.mod_sizes, self.mod_names, self.mod_freq):
            self.buffer[mod_name].reset((framenum*freq_mult,) + tuple(mod_size))
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
        if isinstance(mod, list):
            for mm in mod:
                self.set_full(mm)
            return

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
    from .modality_type.tartandrive_types import rgb_left, costmap, get_vis_costmap
    from .datafile_editor import read_datafile
    from .CacherDataset import CacherDataset
    import numpy as np
    import cv2

    rgbtype = rgb_left([(320, 320)])
    costmaptype = costmap([(320, 320)])
    datafile = 'data_cacher/data/tartandrive.txt'
    trajlist, trajlenlist, framelist, totalframenum = read_datafile(datafile)
    dataroot = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output'

    buffer = TrajBuffer([['img0'], ['cost','vel']], [rgbtype, costmaptype])
    buffer.reset(50, trajlist, trajlenlist, framelist)

    dataset0 = CacherDataset(rgbtype, trajlist, trajlenlist, framelist, datarootdir=dataroot)
    dataset1 = CacherDataset(costmaptype, trajlist, trajlenlist, framelist, datarootdir=dataroot)
    for k in range(50):
        # import ipdb;ipdb.set_trace()
        img0=dataset0[k][0][np.newaxis,...]
        cost=dataset1[k][0][np.newaxis,...]
        vel =dataset1[k][1][np.newaxis,...]

        buffer.insert_frame_one_mod(k, 'img0', img0)
        buffer.insert_frame_one_mod(k, 'cost', cost)
        buffer.insert_frame_one_mod(k, 'vel', vel)

        disp = get_vis_costmap(cost[0])
        # depthvis = visdepth(80./ss2)
        # disp = cv2.hconcat((ss, depthvis))
        cv2.imshow('img', disp)
        cv2.waitKey(0)
    print(buffer.is_full())
    buffer.set_full('img0')
    buffer.set_full('cost')
    buffer.set_full('vel')
    print(buffer.is_full())