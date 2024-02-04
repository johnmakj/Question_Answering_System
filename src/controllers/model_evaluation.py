import argparse
import logging

from src.controllers import ProcessedDataLoader
from src.machine_learning.performance.bert_evaluation import BertEvaluation
from src.machine_learning.performance.bm25_evaluation import BM25Evaluation

logging.getLogger().setLevel(logging.INFO)


class ModelEvaluation(ProcessedDataLoader):
    def __init__(self, data_version: str, processed_data_filename: str, nlp_filename: str = None) -> None:
        """Evaluate models according to the selected method

        Args:
            data_version (str): The version of squad data (train or dev).
            processed_data_filename (str): the filename of the processed data
            nlp_filename (str): the filename of the nlp pipeline used during BM25 training
        """
        super().__init__(data_version, processed_data_filename, nlp_filename)

    def evaluate_bm25(self, ranker_filename: str) -> None:
        """BM25 evaluation"""
        logging.info("Evaluating BM25")
        accuracy = BM25Evaluation(self.preprocessed_data, self.nlp, ranker_filename).evaluate_ranker()
        logging.info("Accuracy of BM25 is: {}".format(accuracy))

    def evaluate_bert(self) -> None:
        """BERT evaluation"""
        logging.info("Evaluating BERT")
        metrics = BertEvaluation(self.preprocessed_data).evaluate_test()
        logging.info("Metrics of BERT are: {}".format(metrics))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation of models')
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
    parser.add_argument('--ranker_filename',
                        type=str,
                        default=None,
                        help='The filename of the trained BM25')
    parser.add_argument('--bert_filename',
                        type=str,
                        default=None,
                        help='The filename of the fine-tuned BERT model')

    # Parse command-line arguments
    args = parser.parse_args()

    evaluator = ModelEvaluation(args.data_version,
                                processed_data_filename=args.processed_data_filename,
                                nlp_filename=args.nlp_filename)

    if args.model in ['BM25', 'bm25']:
        evaluator.evaluate_bm25(args.ranker_filename)
    elif args.model in ['BERT', 'bert']:
        evaluator.evaluate_bert()
    else:
        raise ValueError('Available models: BM25 or BERT')
