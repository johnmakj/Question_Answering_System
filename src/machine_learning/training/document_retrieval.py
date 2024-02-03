import numpy as np
import pandas as pd

from rank_bm25 import BM25Okapi
from spacy.language import Language

from src.tools.general_tools import dump_pickled_data, get_filepath


class DocumentRetrieval:
    def __init__(self, preprocessed_data: pd.DataFrame, nlp: Language) -> None:
        self.preprocessed_data = preprocessed_data
        self.nlp = nlp
        self.pipelines = "_".join(self.nlp.pipe_names[5:])

    def train_ranker(self) -> None:
        """A method to train the ranker"""
        ranker = BM25Okapi(self.preprocessed_data["processed_context"].values.tolist())
        dump_pickled_data(get_filepath('results/models/', f'bm25_{self.pipelines}.pkl'), ranker)

    def retrieve_document(self, query: str, ranker: BM25Okapi) -> str:
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
        return self.preprocessed_data.loc[self.preprocessed_data.index[top_indexes[0]], "context"]
