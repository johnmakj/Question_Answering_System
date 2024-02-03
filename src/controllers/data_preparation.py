import argparse

from src.controllers import RawDataLoader
from src.etl.squad_preprocessing import SquadPreprocessor


class DataPreparation(RawDataLoader):

    def __init__(self, data_version: str) -> None:
        """Prepare data for QA system

        Args:
            data_version (str): version of squad data (train or dev)

        """
        super().__init__(data_version)

    def squad_data(self) -> None:
        """Load data of squad and extract datasets"""
        # prepare dataset and models
        SquadPreprocessor(self.data_version).preprocess_text(self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for QA system')
    parser.add_argument('data_version',
                        type=str,
                        default='dev',
                        help='Version of squad data (train or dev)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Create DataPreparation instance with parsed arguments
    data_preparation = DataPreparation(data_version=args.data_version)

    # Call the squad_data method
    data_preparation.squad_data()
