import yaml
import os


class ConfigDict:
    def __init__(self, dictionary, default_dict=None):
        self.default_dict = default_dict if default_dict else {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigDict(value, self.default_dict.get(key, {}))
            self.__dict__[key] = value

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            default_value = self.default_dict.get(key)
            if default_value is not None:
                if isinstance(default_value, dict):
                    return ConfigDict(default_value)
                return default_value
            raise AttributeError(f"No such attribute: {key}")

    # def __setattr__(self, key, value):
    #     if key != "default_dict" and isinstance(value, dict):
    #         value = ConfigDict(value, self.default_dict.get(key, {}))
    #     self.__dict__[key] = value

    def __repr__(self):
        return f"{self.__dict__}"


def get_config(config_file="config.yaml"):
    with open(os.path.join(os.path.dirname(__file__), "default.yaml"), "r") as f:
        defaults = yaml.safe_load(f)

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        return ConfigDict(config, defaults)
    return ConfigDict(defaults)
