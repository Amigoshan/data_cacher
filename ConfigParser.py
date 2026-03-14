import yaml
from collections import OrderedDict

class ConfigParser(object):
    """
    Class that reads and validates dataset specification files.

    Supports YAML format with global defaults and per-dataset overrides.
    Validates configuration against predefined schemas for better error messages.
    """

    def __init__(self):
        # Required global params
        self.global_paramlist = ['task']

        # Modality params (per key like 'img0')
        self.modality_paramlist = {
            'cacher_size': {'type': 'list', 'minlength': 2, 'schema': {'type': 'integer'}, 'required': True},
            'length': {'type': 'integer', 'min': 1, 'required': True}
        }

        # Cacher params
        self.cacher_paramlist = {
            'data_root_key': {'type': 'string', 'required': True},
            'data_root_path_override': {'type': 'string', 'required': False},
            'subset_framenum': {'type': 'integer', 'min': 1, 'required': True},
            'worker_num': {'type': 'integer', 'min': 0, 'required': True},
            'load_traj': {'type': 'boolean', 'required': True}
        }

        # Dataset params (optional, with defaults)
        self.dataset_paramlist = {
            'frame_skip': {'type': 'integer', 'min': 0, 'required': False},
            'seq_stride': {'type': 'integer', 'min': 1, 'required': False},
            'frame_dir': {'type': 'boolean', 'required': False}
        }

        # Parameter params (optional)
        self.parameter_paramlist = {
            'intrinsics': {'type': 'list', 'minlength': 6, 'maxlength': 6, 'required': False},
            'intrinsics_scale': {'type': 'list', 'minlength': 2, 'maxlength': 2, 'required': False},
            'fxbl': {'type': 'number', 'required': False},
            'input_size': {'type': 'list', 'minlength': 2, 'maxlength': 2, 'required': False},
            'cam_model_for_flow': {'type': 'list', 'minlength': 6, 'maxlength': 6, 'required': False},
            'cam_model_for_intrinsics_layer': {'type': 'list', 'minlength': 6, 'maxlength': 6, 'required': False}
        }


    def parse_from_fp(self, fp):
        """
        Parse YAML from file path.

        Args:
            fp (str): Path to YAML file.

        Returns:
            dict: Parsed and validated config.

        Raises:
            ValueError: If validation fails.
        """
        try:
            with open(fp, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {fp}: {e}")
        return self.parse(data)

    def parse_from_dict(self, data):
        """
        Parse from dict (for programmatic configs).

        Args:
            data (dict): Config dict.

        Returns:
            dict: Parsed and validated config.
        """
        return self.parse(data)

    def parse(self, spec):
        """
        Parse and validate the config dict.

        Applies global defaults and validates structure.

        Args:
            spec (dict): Raw config dict from YAML.

        Returns:
            dict: Parsed config with defaults applied.

        Raises:
            ValueError: If required fields missing or validation fails.
        """
        if not isinstance(spec, dict):
            raise ValueError("Config must be a dict")

        dataset_config = OrderedDict()
        dataset_config['task'] = spec.get('task')
        if not dataset_config['task']:
            raise ValueError("Missing required 'task' in config")

        # Get global defaults
        default_modality_params = self.parse_sub_global_param(spec, "modality", self.modality_paramlist)
        default_cacher_params = self.parse_sub_global_param(spec, "cacher", self.cacher_paramlist)
        default_dataset_params = self.parse_sub_global_param(spec, "dataset", self.dataset_paramlist)
        default_parameter_params = self.parse_sub_global_param(spec, "parameter", self.parameter_paramlist)

        data_config = {}
        for datasetind, params in spec.get('data', {}).items():
            all_params = {}
            if 'file' not in params:
                raise ValueError(f"Missing 'file' in data/{datasetind}")
            datafile = params['file']
            all_params['file'] = datafile

            if 'modality' not in params:
                raise ValueError(f"Missing 'modality' in data/{datasetind}")
            all_modality_params = {}
            modality_list = params['modality']
            for mod_type in modality_list:
                modtype_params = {}
                for modkey in modality_list[mod_type]:
                    modality_params = self.parse_sub_data_param(modality_list[mod_type][modkey], self.modality_paramlist, default_modality_params)
                    modtype_params[modkey] = modality_params
                all_modality_params[mod_type] = modtype_params
            all_params['modality'] = all_modality_params

            if 'cacher' not in params:
                raise ValueError(f"Missing 'cacher' in data/{datasetind}")
            cacher_params = self.parse_sub_data_param(params["cacher"], self.cacher_paramlist, default_cacher_params)
            all_params['cacher'] = cacher_params

            dataset_params = self.parse_sub_data_param(params.get("dataset", {}), self.dataset_paramlist, default_dataset_params)
            all_params['dataset'] = dataset_params

            parameter_params = self.parse_sub_data_param(params.get("parameter", {}), self.parameter_paramlist, default_parameter_params)
            all_params['parameter'] = parameter_params

            data_config[datasetind] = all_params

        dataset_config['data'] = data_config

        # Validate the final config
        self._validate_final(dataset_config)

        return dataset_config

    def _validate_param(self, name, value, schema):
        """Validate a single parameter against a schema definition."""
        if value is None:
            return

        expected_type = schema.get('type')
        if expected_type:
            if expected_type == 'list':
                if not isinstance(value, list):
                    raise ValueError(f"Parameter '{name}' should be a list")
                minlength = schema.get('minlength')
                maxlength = schema.get('maxlength')
                if minlength is not None and len(value) < minlength:
                    raise ValueError(f"Parameter '{name}' must have at least {minlength} elements")
                if maxlength is not None and len(value) > maxlength:
                    raise ValueError(f"Parameter '{name}' must have at most {maxlength} elements")
                item_schema = schema.get('schema')
                if item_schema is not None:
                    for i, item in enumerate(value):
                        if item_schema.get('type') == 'integer' and not isinstance(item, int):
                            raise ValueError(f"Parameter '{name}' element {i} must be integer")
                        if item_schema.get('type') == 'number' and not isinstance(item, (int, float)):
                            raise ValueError(f"Parameter '{name}' element {i} must be number")
            elif expected_type == 'string':
                if not isinstance(value, str):
                    raise ValueError(f"Parameter '{name}' should be a string")
            elif expected_type == 'integer':
                if not isinstance(value, int):
                    raise ValueError(f"Parameter '{name}' should be an integer")
            elif expected_type == 'number':
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{name}' should be a number")
            elif expected_type == 'boolean':
                if not isinstance(value, bool):
                    raise ValueError(f"Parameter '{name}' should be a boolean")

    def parse_sub_global_param(self, spec, param_name, paramlist):
        '''
        Read global defaults for a section (modality, cacher, etc.).

        This returns a dict with all expected keys (from `paramlist`), where missing
        values are set to None.
        '''
        global_values = spec.get('global', {}).get(param_name, {}) or {}
        defaults = {}
        for param, schema in paramlist.items():
            val = global_values.get(param)
            self._validate_param(param, val, schema)
            defaults[param] = val
        return defaults

    def parse_sub_data_param(self, params, paramlist, default_params):
        '''
        Merge data-specific parameters with defaults.

        Raises:
            ValueError: if a required parameter is missing or type/shape mismatch.
        '''
        merged = dict(default_params)
        merged.update(params or {})

        # Validate parameters according to schema
        for p, schema in paramlist.items():
            val = merged.get(p)
            # Only enforce required params
            if schema.get('required', True) and val is None:
                raise ValueError(f"Missing required parameter '{p}'")
            self._validate_param(p, val, schema)

        return merged

    def _validate_final(self, config):
        """Basic validation of the final config structure."""
        if 'task' not in config:
            raise ValueError("Missing 'task'")
        if 'data' not in config or not isinstance(config['data'], dict):
            raise ValueError("'data' must be a dict")
        for key, data in config['data'].items():
            required = ['file', 'modality', 'cacher', 'dataset', 'parameter']
            for r in required:
                if r not in data:
                    raise ValueError(f"Data {key} missing '{r}'")

    def validate_config_file(self, fp):
        """
        Validate a config file without parsing (for CLI use).

        Args:
            fp (str): Path to YAML file.

        Raises:
            ValueError: If invalid.
        """
        self.parse_from_fp(fp)
        print(f"Config {fp} is valid.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fp = sys.argv[1]
        parser = ConfigParser()
        try:
            parser.validate_config_file(fp)
        except ValueError as e:
            print(f"Validation failed: {e}")
            sys.exit(1)
    else:
        print("Usage: python ConfigParser.py <config.yaml>")
