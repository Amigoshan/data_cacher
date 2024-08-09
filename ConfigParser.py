import yaml

from collections import OrderedDict

class ConfigParser(object):
    """
    Class that reads in the spec dataset.
    This is simply reading in the yaml file and handling the global default value
    """
    def __init__(self):
        # these params are required, assert error if not provided
        self.global_paramlist = ['task']

        self.modality_paramlist = ['cacher_size', # (h, w) the size of the modality
                                   'length', # the sequence lengh of one particular modality
                                  ]

        self.cacher_paramlist = [   'data_root_key',
                                    'data_root_path_override', # override the default data root path with this one.
                                    'subset_framenum', # frame number in cacher
                                    'worker_num', # how many works for the cacher
                                    'load_traj' # load one trajectory into the cacher at one time
                                ]

        self.dataset_paramlist = ['frame_skip',
                                  'seq_stride',
                                  'frame_dir'
                                 ]

        # This defines parameters that come with the dataset
        # such as camera intrinsics and stereo baseline     
        # These params can be used in RAMDataset, 
        # which will utilizing these values or directly return them for up-level class   
        self.parameter_paramlist = ['intrinsics', # [w, h, fx, fy, ox, oy] - w corresponds to x and h cooresponds to y
                                    'intrinsics_scale', #[scale_x, scale_y] - note that the order of x and y are is different from the way in cacher_size
                                    'fxbl', # [focal_length * baseline]
                                    'input_size', # [h, w] - allow different datasets be rcr to different input size
                                    'cam_model_for_flow', # [w, h, fx, fy, ox, oy] # replace the intrinsics parameter
                                    'cam_model_for_intrinsics_layer', # [w, h, fx, fy, ox, oy]  
                                   ]

    def parse_from_fp(self, fp):
        x = yaml.safe_load(open(fp, 'r'))
        return self.parse(x)

    def parse_from_dict(self, x):
        return self.parse(x)

    def parse_sub_global_param(self, spec, param_name, paramlist):
        '''
        the global param list should be the same with the individual param list
        this function simply reads the global params 
        the missing param will be assigned with None
        '''
        default_params = {}
        for param in paramlist:
            if not 'global' in spec or \
                spec['global'] is None or (not param_name in spec['global']) or \
                (spec['global'][param_name] is None) or (not param in spec['global'][param_name]):
                default_params[param] = None 
            else:
                default_params[param] = spec['global'][param_name][param]
        return default_params

    def parse_sub_data_param(self, subparams, paramlist, default_params):
        '''
        params_spec: the raw data from the file under one datafile
        param_name: modality/cacher/dataset
        paramlist: self.dataset_paramlist/modality_paramlist/..
        default_params: the return of the parse_sub_global_param function
        '''

        data_params = {}
        # assert param_name in params_spec, 'ConfigParser: Missing {} in the spec'.format(param_name)
        # subparams = params_spec[param_name]
        for param in paramlist:
            if subparams is not None and param in subparams: # use specific param
                data_params[param] = subparams[param]
            elif default_params[param] is not None: # use default param
                data_params[param] = default_params[param]

        return data_params

    def parse(self, spec):
        dataset_config = OrderedDict()

        # Check if the global params are available
        for param in self.global_paramlist:
            if param in spec: 
                dataset_config[param] = spec[param]
            else:
                assert False, "ConfigParser: Missing {} in the spec file".format(param)

        default_modality_params = self.parse_sub_global_param(spec, "modality", self.modality_paramlist)
        default_cacher_params = self.parse_sub_global_param(spec, "cacher", self.cacher_paramlist)
        default_dataset_params = self.parse_sub_global_param(spec, "dataset", self.dataset_paramlist)
        default_parameter_params = self.parse_sub_global_param(spec, "parameter", self.parameter_paramlist)

        data_config = {}
        for datasetind, params in spec['data'].items():
            all_params = {}
            assert 'file' in params, 'ConfigParser: Missing filename in the spec data/{}'.format(datasetind)
            datafile = params['file']
            all_params['file'] = datafile

            # import ipdb;ipdb.set_trace()
            assert 'modality' in params, 'ConfigParser: Missing modality in the data/{}'.format(datasetind)
            all_modality_params = {}
            modality_list = params['modality']
            for mod_type in modality_list: 
                modtype_params = {}
                for modkey in modality_list[mod_type]:
                    modality_params = self.parse_sub_data_param(modality_list[mod_type][modkey], self.modality_paramlist, default_modality_params)
                    modtype_params[modkey] = modality_params
                all_modality_params[mod_type] = modtype_params
            all_params['modality'] = all_modality_params

            assert 'cacher' in params, 'ConfigParser: Missing cacher in the data/{}'.format(datasetind)
            cacher_params = self.parse_sub_data_param(params["cacher"], self.cacher_paramlist, default_cacher_params)
            all_params['cacher'] = cacher_params

            assert 'dataset' in params, 'ConfigParser: Missing dataset in the data/{}'.format(datasetind)
            dataset_params = self.parse_sub_data_param(params["dataset"], self.dataset_paramlist, default_dataset_params)
            all_params['dataset'] = dataset_params

            assert 'parameter' in params, 'ConfigParser: Missing parameter in the data/{}'.format(datasetind)
            parameter_params = self.parse_sub_data_param(params["parameter"], self.parameter_paramlist, default_parameter_params)
            all_params['parameter'] = parameter_params

            data_config[datasetind] = all_params

        dataset_config['data'] = data_config
        return dataset_config


if __name__ == "__main__":
    # fp = open('dataspec/flowvo_train_local_new.yaml')
    dataset_specfile = '/home/amigo/workspace/pytorch/ss_costmap/data_cacher/dataspec/flowvo_test_local_v1.yaml'
    fp = open(dataset_specfile)
    d = yaml.safe_load(fp)
    print(d)
    print(type(d))
    parser = ConfigParser()
    x = parser.parse(d)
    print(x)
