import argparse

from src.controllers import ProcessedDataLoader
from src.machine_learning.training.bert_qa_finetuning import BertQAFinetuning
from src.machine_learning.training.document_retrieval import DocumentRetrieval


class ModelTrainer(ProcessedDataLoader):
    def __init__(self, data_version: str, processed_data_filename: str, nlp_filename: str = None) -> None:

        """Training models according to the selected method

        Args:
            data_version (str): The version of squad data (train or dev).
            processed_data_filename (str): the filename of the processed data
            nlp_filename (str): the filename of the nlp pipeline used during BM25 training
        """
        super().__init__(data_version, processed_data_filename, nlp_filename)

    def train_bm25(self) -> None:
        """Train BM25"""
        DocumentRetrieval(self.preprocessed_data, self.nlp).train_ranker()

    def fine_tune_bert(self) -> None:
        """Fine tune BERT for question answering"""
        BertQAFinetuning(self.preprocessed_data).fine_tune_bert()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training of models')
    parser.add_argument('data_version', type=str, help='Version of squad data (train or dev)')
    parser.add_argument('--processed_data_filename',
                        type=str,
                        default='dev_preprocessed_lowercase_remove_punctuation_remove_stopwords.pkl',
                        help='File name of the processed data')
    parser.add_argument('--nlp_filename',
                        type=str,
                        default='dev_nlp_pipeline_lowercase_remove_punctuation_remove_stopwords.pkl',
                        help='File name of the nlp pipeline used during training')
    parser.add_argument('--model',
                        type=str,
                        default='BM25',
                        help='Model to evaluate. Available options: BM25 or BERT')

    # Parse command-line arguments
    args = parser.parse_args()

    trainer = ModelTrainer(args.data_version,
                           processed_data_filename=args.processed_data_filename,
                           nlp_filename=args.nlp_filename)

    if args.model in ['BM25', 'bm25']:
        trainer.train_bm25()
    elif args.model in ['BERT', 'bert']:
        trainer.fine_tune_bert()
    else:
        raise ValueError('Available models: BM25 or BERT')

