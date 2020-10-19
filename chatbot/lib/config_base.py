from collections import abc


class ConfigObject:
    """Basic configuration object class that allows dynamic attribute.

    Usage:
    >>> config = {
        'param1': value1,
        'param2': value2
        }
    >>> config_object = ConfigObject(config)
    >>> value1 = config_object.param1

    Attributes:
        mapping dict: The dict to be converted into ConfigObject.
    """
    def __init__(self, mapping):
        self.__data = dict(mapping)

    def __getattr__(self, name):
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return ConfigObject._build(self.__data[name])

    def __dir__(self):
        return self.mapping.keys()

    @classmethod
    def _build(cls, obj):
        if isinstance(obj, abc.Mapping):
            return cls(obj)
        elif isinstance(obj, abc.MutableSequence):
            return [cls._build(item) for item in obj]
        else:
            return obj


class PathConfig(object):
    """Path Config Class to be instantiated.

    Attributes:
        mapping dict: The dict to be converted into PathConfig.

    """
    bucket: str
    athena_output: str
    output: str

    def __init__(self, mapping):
        self.__data = dict(mapping)

    def __getattr__(self, name):
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return PathConfig._build(self.__data[name])

    def __dir__(self):
        return self.attributes.keys()

    @classmethod
    def _build(cls, obj):
        if isinstance(obj, abc.Mapping):
            return cls(obj)
        elif isinstance(obj, abc.MutableSequence):
            return [cls._build(item) for item in obj]
        else:
            return obj
