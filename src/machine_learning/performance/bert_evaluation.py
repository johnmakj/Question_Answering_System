from typing import Dict

import pandas as pd
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast

from src.machine_learning.training.bert_qa_finetuning import BertQAFinetuning
from src.tools.evaluation_tools import metric_max_over_ground_truths, exact_match_score, f1_score
from src.tools.general_tools import get_filepath, load_yaml_config

DEFAULT_CONFIG_BERT_PATH = get_filepath('src/machine_learning/config',
                                        "bert_config.yaml")

config = load_yaml_config(DEFAULT_CONFIG_BERT_PATH)


class BertEvaluation(BertQAFinetuning):
    def __init__(self, preprocessed_data: pd.DataFrame) -> None:
        """
        The constructor of the class
        Args:
            preprocessed_data (pd.DataFrame): A preprocessed dataframe containing context, questions, answers along
            with the start and end positions of the answer
            model_filename (str): the name of file where the pretrained model is
        """
        super().__init__(preprocessed_data)
        self.model = BertForQuestionAnswering.from_pretrained("jmakj/bert-base-finetuned-squad")
        self.question_eval = zip(preprocessed_data["question"],
                                 preprocessed_data["context"],
                                 preprocessed_data["possible_answers"])

    def evaluate_test(self) -> Dict:
        """
        A method that evaluates the fine-tuned Bert on the test set. It computed f1 score and exact match according
        to the official script
        Returns:
            dict
        """
        f1 = exact_match = 0
        device = torch.device("cpu")
        model = self.model.to(device)
        model.eval()
        tokenizer = eval(config["pretrained"]["tokenizer"])
        for question, context, ground_truth in self.question_eval:
            inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True).to(device)

            output = model(**inputs)
            start_idx = torch.argmax(output.start_logits)
            end_idx = torch.argmax(output.end_logits) + 1
            answer = tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))
            print(answer)
            exact_match += metric_max_over_ground_truths(
                exact_match_score, answer, ground_truth)
            f1 += metric_max_over_ground_truths(
                f1_score, answer, ground_truth)

        exact_match = 100.0 * exact_match / len(self.preprocessed_data)
        f1 = 100.0 * f1 / len(self.preprocessed_data)
        return {'exact_match': exact_match, 'f1': f1}

