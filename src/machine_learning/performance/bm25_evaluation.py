import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from spacy.language import Language

from src.machine_learning.training.document_retrieval import DocumentRetrieval
from src.tools.general_tools import load_pickled_data, get_filepath


class BM25Evaluation(DocumentRetrieval):
    def __init__(self, preprocessed_data: pd.DataFrame, nlp: Language, ranker_filename: str) -> None:
        """
        The constructor of the class
        """
        super().__init__(preprocessed_data, nlp)
        self.ranker = load_pickled_data(get_filepath('results/models', ranker_filename))
        self.eval_queries_dict = dict(zip(self.preprocessed_data['question'], self.preprocessed_data["context_id"]))

    def retrieve_document(self, query: str, ranker: BM25Okapi) -> int:
        """
        A method that given a query in natural language preprocess it and tokenizes it and retrieves the most
        relevant document according to BM25 score
        Args:
            query (str): The use query
            ranker(BM25Okapi): The trained ranker for document retrieval

        Returns:
            str
        """
        tokenized_processed_query = [token.text for token in self.nlp(query)]
        scores = ranker.get_scores(tokenized_processed_query)
        top_indexes = np.argsort(scores)[::-1][:1]
        return self.preprocessed_data.loc[self.preprocessed_data.index[top_indexes], 'context_id'].values[0]

    def evaluate_ranker(self) -> float:
        """
        A method that calculates the accuracy of bm25 on the test set

        Returns:
            float
        """
        accuracy = 0

        for query, context_id in self.eval_queries_dict.items():
            retrieved_document_id = self.retrieve_document(query, self.ranker)

            if context_id == retrieved_document_id:
                accuracy += 1

        return accuracy / len(self.eval_queries_dict)
