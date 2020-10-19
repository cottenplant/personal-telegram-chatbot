from collections import abc
from pathlib import Path


class PathConfig(object):
    """Path Config Class to be instantiated.

    Attributes:
        mapping dict: The dict to be converted into PathConfig.

    """
    # user: str
    # seq_length: int
    # multiplier: int
    # num_gen_words: int
    # num_epochs: int

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

# fix type hinting and self-reference
paths_local = PathConfig(dict(
    user='samco',
    seq_length=15,
    multiplier=4,
    num_gen_words=25,
    num_epochs=200,
    chats_raw_json = Path('./data/input/result.json'),
    sequences_seed_path = Path('./models/sequences_seed_text.csv'),
    chats_processed_json = Path('./data/output/samco.json'),
    chats_processed_txt = Path('./data/output/samco.txt'),
    model_path = Path('./models/chatbot_model_200.h5'),
    tokenizer_path = Path('./models/chatbot_tokenizer_200')
))

print(paths_local)