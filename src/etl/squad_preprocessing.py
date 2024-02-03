from typing import Dict, Tuple, List

import pandas as pd
import spacy

from spacy.language import Language

from src.tools.general_tools import get_filepath, load_yaml_config, load_json_file, time_it, dump_pickled_data, \
    load_pickled_data
from src.tools.text_tools import lemmatize_doc, lowercase_doc, remove_punctuation_doc, remove_stopwords_doc

DEFAULT_CONFIG_PREPROCESSING_PATH = get_filepath('src/etl/config', "bm25_preprocessing.yaml")

config = load_yaml_config(DEFAULT_CONFIG_PREPROCESSING_PATH)


class SquadPreprocessor:
    def __init__(self, data_version) -> None:
        self.data_version = data_version

    @staticmethod
    def json_to_dataframe(data_json: Dict) -> pd.DataFrame:
        """
        A method that converts a json-like data (dictionary) into a pandas dataframe. It is specific for
        the SquAD dataset format
        Args:
            data_json (dict): The data after loading from the json file

        Returns:
            pd.DataFrame
        """
        articles = []
        for article in data_json["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]
                    answer_starts = qa["answers"][0]["answer_start"]
                    answer_ends = answer_starts + len(answer)
                    possible_answers = [i["text"] for i in qa["answers"]]
                    id_q = qa['id']
                    inputs = {"id_q": id_q, "context": paragraph["context"], "question": question, "answer": answer,
                              "answer_starts": answer_starts, "answer_ends": answer_ends,
                              "possible_answers": possible_answers}
                    articles.append(inputs)

        data = pd.DataFrame(articles)

        return data

    @staticmethod
    def create_nlp() -> Tuple[Language, List]:
        """
        A method that creates a nlp pipeline with a predefine preprocessing steps according to a config file
        Returns:
            (Language, list)
        """
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        pipelines = nlp.pipe_names
        # Add preprocessing steps
        if config["nlp_pipeline"]["lowercase"]:
            nlp.add_pipe('lowercase', last=True)
        if config["nlp_pipeline"]["remove_punctuation"]:
            nlp.add_pipe('remove_punctuation', last=True)
        if config["nlp_pipeline"]["stopword_removal"]:
            nlp.add_pipe('remove_stopwords', last=True)
        if config["nlp_pipeline"]["lemmatize"]:
            nlp.add_pipe('lemmatize', last=True)

        return nlp, [name for name in nlp.pipe_names if name not in pipelines]

    @time_it
    def preprocess_text(self, data_json_train: Dict) -> pd.DataFrame:
        """
        A method that preprocesses the text. It applies the nlp pipeline to the text and adds a column to the dataframe
        with the preprocessed text in order for the BM25 to be trained. It also adds an id for each context for
        evaluation purposes

        Returns:
            pd.DataFrame
        """
        squad_dataframe = SquadPreprocessor.json_to_dataframe(data_json_train)
        nlp, pipelines = SquadPreprocessor.create_nlp()
        pipelines = "_".join(pipelines)
        squad_dataframe["processed_context"] = (squad_dataframe['context'].
                                                apply(lambda x: [token.text for token in nlp(x)]))
        squad_dataframe["processed_question"] = (squad_dataframe['question'].
                                                 apply(lambda x: [token.text for token in nlp(x)]))
        # add context_id for evaluation
        context_id_dict = dict(zip(squad_dataframe["context"].unique(),
                                   range(len(squad_dataframe["context"].unique()))))

        squad_dataframe['context_id'] = squad_dataframe["context"].map(context_id_dict)
        dump_pickled_data(get_filepath(f'results/data_preprocessing/',
                                       f'{self.data_version}_preprocessed_{pipelines}.pkl'), squad_dataframe)

        dump_pickled_data(get_filepath('results/models/',
                                       f'{self.data_version}_nlp_pipeline_{pipelines}.pkl'), nlp)

        return squad_dataframe
