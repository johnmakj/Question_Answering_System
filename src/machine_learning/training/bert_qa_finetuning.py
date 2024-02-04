import os

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, BertModel

from src.etl.bert_squad_tokenizer import SquadTokenizer, SquadDataset
from src.machine_learning.training.bert_qa_layer import CustomQuestionAnsweringModel
from src.tools.general_tools import _init_fn, get_filepath, load_yaml_config

DEFAULT_CONFIG_BERT_PATH = get_filepath('src/machine_learning/config',
                                        "bert_config.yaml")

config = load_yaml_config(DEFAULT_CONFIG_BERT_PATH)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

CUDA_LAUNCH_BLOCKING = "1"


class BertQAFinetuning:
    def __init__(self, preprocessed_data: pd.DataFrame) -> None:
        """
        The constructor of the class
        Args:
            preprocessed_data (pd.DataFrame): A preprocessed dataframe containing context, questions, answers along
            with the start and end positions of the answer
        """

        self.preprocessed_data = SquadDataset(preprocessed_data)
        self.n_epochs = config["bert_qa"]["n_epochs"]
        self.learning_rate = eval(config["bert_qa"]["learning_rate"])
        self.weight_decay = eval(config["bert_qa"]["weight_decay"])

        self.dataloader = DataLoader(self.preprocessed_data,
                                     batch_size=config["bert_qa"]["batch_size"],
                                     shuffle=False,
                                     num_workers=config["bert_qa"]["num_workers"],
                                     worker_init_fn=_init_fn,
                                     pin_memory=True)

    def fine_tune_bert(self) -> None:
        """
        Fine-tune a pre-trained BERT model on the SQuAD dataset.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = eval(config["pretrained"]["tokenizer"])  # BertTokenizerFast.from_pretrained("bert-base-uncased")

        # Define the configuration
        config_i = BertConfig.from_pretrained(config["pretrained"]["model"])
        model = CustomQuestionAnsweringModel(
            config_i)  # eval(config["pretrained"]["model"])  # BertForQuestionAnswering.from_pretrained("bert-base-uncased")
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, eps=self.weight_decay)
        squad_tokenizer = SquadTokenizer(tokenizer)
        model.to(device)
        model.resize_token_embeddings(len(tokenizer))
        sum_loss = 0.0

        for epoch in range(self.n_epochs):
            model.train()
            for batch_token in self.dataloader:
                optimizer.zero_grad()
                encoding, token_starts, token_ends = squad_tokenizer(batch_token)
                token_starts = torch.LongTensor(token_starts).to(device)
                token_ends = torch.LongTensor(token_ends).to(device)
                input_ids = torch.LongTensor(encoding['input_ids']).to(device)
                attention_mask = torch.LongTensor(encoding['attention_mask']).to(device)
                token_type_ids = torch.LongTensor(encoding['token_type_ids']).to(device)

                _, loss = model(input_ids, attention_mask, token_type_ids, start_positions=token_starts,
                                end_positions=token_ends)
                loss.backward()
                optimizer.step()
                # sum_loss += loss.item()

        average_loss = sum_loss / self.n_epochs
        model.push_to_hub(config["bert_qa"]["model_name"], use_auth_token=config["bert_qa"]["token"])

    @staticmethod
    def question_answering(model: CustomQuestionAnsweringModel, question: str, context: str) -> str:
        """
        A method that answers a questions according to the given context and the fine-tuned BERT model
        Args:
            model (CustomQuestionAnsweringModel): The fine-tuned model
            question (str): The question
            context (str): The context

        Returns:
            str
        """
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()
        tokenizer = eval(config["pretrained"]["tokenizer"])
        inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True).to(device)

        output = model(**inputs)

        start_logits, end_logits = output
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # Get the predicted start and end positions
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))

        return answer
