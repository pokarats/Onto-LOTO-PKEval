from abc import ABC, abstractmethod
import pickle
import logging
import csv
import json

logger = logging.getLogger(__name__)


class BaseIter(ABC):
    filename = None

    @abstractmethod
    def __iter__(self):
        pass


class CorpusIter(BaseIter):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, encoding='utf-8', errors='ignore') as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                yield line.lower()


class ProcessedIterExtended(BaseIter):
    """
    Sentence iterator class for processing .csv file; this is the version from Multi-Res CNN
    """

    def __init__(self, filename, header=True, delimiter=",", slice_pos=None):
        """

        :param filename: path to corpus file
        :type filename: str or Path
        :param header: whether or not file contains a header row, if True skip this header row
        :type header: bool
        :param delimiter: char that each row is delimited by, e.g. default is ",", use "\t" for a .tsv file
        :type delimiter: str
        :param slice_pos: column position of the data to read e.g. for MIMIC-III texts are in 2
        :type slice_pos: int
        """
        self.filename = filename
        self.header = header
        self.delimiter = delimiter
        self.slice_pos = slice_pos

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f, delimiter=self.delimiter)
            if self.header:
                next(r)
            for row in r:
                if self.slice_pos:
                    yield row[self.slice_pos]
                else:
                    yield row

class JsonCorpusReader(BaseIter):
    def __init__(self, filename, dict_key):
        self.filename = filename
        self.dict_key = dict_key
    def __iter__(self):
        with open(self.filename, "r", encoding="utf-8") as j_file:
            data = json.load(j_file)
            for item_dict in data:
                yield item_dict.get(self.dict_key)


def pickle_obj(obj_to_pickle, pickle_filepath):
    """

    :param pickle_filepath: the desired pickle filename e.g. some/directory/pickle_file.pickle
    :param obj_to_pickle:
    :param args_cl: parse args dict
    :return: None
    """

    with open(pickle_filepath, 'wb') as handle:
        pickle.dump(obj_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(pickle_filepath, 'rb') as handle:
        pickled = pickle.load(handle)

    assert (obj_to_pickle == pickled), f"Pickled object not the same as original!!"
    logger.info(f"Object pickled saved at: {pickle_filepath}")
