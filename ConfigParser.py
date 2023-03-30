from email.policy import default
from cv2 import transform
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
                                   'type', # the class type under modality_type
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

    def parse_from_fp(self, fp):
        x = yaml.safe_load(open(fp, 'r'))
        return self.parse(x)

    def parse_from_dict(self, x):
        return self.parse(x)

    def parse_sub_global_param(self, spec, param_name, paramlist):
        default_params = {}
        for param in paramlist:
            key = param_name + '_'+ param
            default_params[param] = spec[key] if key in spec else None

        return default_params

    def parse_sub_data_param(self, params_spec, param_name, paramlist, default_params):
        '''
        params_spec: the raw data from the file under one datafile
        param_name: modality/cacher/dataset
        paramlist: self.dataset_paramlist/modality_paramlist/..
        default_params: the return of the parse_sub_global_param function
        '''

        data_params = {}
        assert param_name in params_spec, 'ConfigParser: Missing {} in the spec'.format(param_name)
        subparams = params_spec[param_name]
        for param in paramlist:
            if subparams is not None and param in subparams: # use specific param
                data_params[param] = subparams[param]
            elif default_params[param] is not None: # use default param
                data_params[param] = default_params[param]

        return data_params

    def parse(self, spec):
        dataset_config = OrderedDict()
        for param in self.global_paramlist:
            if param in spec: 
                dataset_config[param] = spec[param]
            else:
                assert False, "ConfigParser: Missing {} in the spec file".format(param)
        
        default_modality_params = self.parse_sub_global_param(spec, "modality", self.modality_paramlist)
        default_cacher_params = self.parse_sub_global_param(spec, "cacher", self.cacher_paramlist)
        default_dataset_params = self.parse_sub_global_param(spec, "dataset", self.dataset_paramlist)

        data_config = {}
        for datafile, params in spec['data'].items():
            all_params = {}

            assert 'modality' in params, 'ConfigParser: Missing modality in the spec {}'.format(datafile)
            all_modality_params = {}
            for modkey in params['modality']:
                modality_params = self.parse_sub_data_param(params['modality'], modkey, self.modality_paramlist, default_modality_params)
                all_modality_params[modkey] = modality_params
            all_params['modality'] = all_modality_params

            cacher_params = self.parse_sub_data_param(params, "cacher", self.cacher_paramlist, default_cacher_params)
            all_params['cacher'] = cacher_params

            dataset_params = self.parse_sub_data_param(params, "dataset", self.dataset_paramlist, default_dataset_params)
            all_params['dataset'] = dataset_params

            data_config[datafile] = all_params

        dataset_config['data'] = data_config
        return dataset_config


if __name__ == "__main__":
    fp = open('dataspec/flowvo_train_local_new.yaml')
    d = yaml.safe_load(fp)
    print(d)
    print(type(d))
    parser = ConfigParser()
    x = parser.parse(d)
    print(x)
