from src.tools.general_tools import load_pickled_data, get_filepath, load_json_file


class Controller:
    """The general controller"""

    def __init__(self, data_version: str) -> None:
        """ Initializes a controller class.

        Args:
            data_version: The version of squad data (train or dev).
        """
        if data_version not in ["train", "dev"]:
            raise ValueError(f"You should choose either train or dev")
        self.data_version = data_version


class Singleton(type):
    """Singleton metaclass"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class RawDataLoader(Controller, metaclass=Singleton):
    """Used by controllers that use the unprocessed data."""

    def __init__(self, data_version: str):
        super().__init__(data_version)

        # Load the unprocessed data in the main memory.# Load the processed data in the main memory.
        json_path = get_filepath('data', f'{self.data_version}-v1.1.json')

        self.data = load_json_file(json_path)


class ProcessedDataLoader(Controller, metaclass=Singleton):
    """Used by controllers that use the processed data."""

    def __init__(self, data_version: str, processed_data_filename: str, nlp_filename: str = None) -> None:
        super().__init__(data_version)
        self.processed_data_filename = processed_data_filename

        # Load the processed data in the main memory.

        datasets_path = get_filepath('results/data_preprocessing', self.processed_data_filename)
        self.preprocessed_data = load_pickled_data(datasets_path)
        if nlp_filename is not None:
            nlp_path = get_filepath('results/models', nlp_filename)
            self.nlp = load_pickled_data(nlp_path)


