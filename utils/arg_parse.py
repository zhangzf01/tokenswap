import yaml

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def merge_args_with_yaml(args, yaml_config):
    for key, value in yaml_config.items():
        if not hasattr(args, key): 
            setattr(args, key, value)
    return args

