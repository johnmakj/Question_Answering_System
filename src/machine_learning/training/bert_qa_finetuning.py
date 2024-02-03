import os

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizerFast

from src.etl.bert_squad_tokenizer import SquadTokenizer, SquadDataset
from src.tools.general_tools import _init_fn, get_filepath, load_yaml_config

DEFAULT_CONFIG_BERT_PATH = get_filepath('src/machine_learning/config',
                                        "bert_config.yaml")

config = load_yaml_config(DEFAULT_CONFIG_BERT_PATH)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()


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

        args:
            model (BertForQuestionAnswering) : a pre-trained BERT model for QA tasks
            n_epochs (int) : the number of epochs to train
            dataloader (DataLoader) : a data loader that provides access to one batch of data at a time
            squad_tokenizer (SQuADTokenizer) : a tokenizer instance to be called on every batch of data from dataloader
            device (torch.device) : the device (CPU or Cuda) that the model and data should be moved to
            verbose (bool) : a flag that indicates whether debug messages should be printed out

        return:
            model (BertForQuestionAnswering) : the fine-tuned model
            avg_loss (float) : the average training loss across epochs
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = eval(config["pretrained"]["tokenizer"])  # BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = eval(config["pretrained"]["model"])  # BertForQuestionAnswering.from_pretrained("bert-base-uncased")
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, eps=self.weight_decay)
        squad_tokenizer = SquadTokenizer(tokenizer)
        model.to(device)
        model.train()
        sum_loss = 0.0

        for epoch in range(self.n_epochs):
            for batch_token in self.dataloader:
                optimizer.zero_grad()
                encoding, token_starts, token_ends = squad_tokenizer(batch_token)
                token_starts = torch.LongTensor(token_starts).to(device)
                token_ends = torch.LongTensor(token_ends).to(device)
                input_ids = torch.LongTensor(encoding['input_ids']).to(device)
                attention_mask = torch.LongTensor(encoding['attention_mask']).to(device)
                token_type_ids = torch.LongTensor(encoding['token_type_ids']).to(device)

                loss = model(input_ids, attention_mask, token_type_ids, start_positions=token_starts,
                             end_positions=token_ends).loss
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()

        average_loss = sum_loss / self.n_epochs
        #model.cpu().save_pretrained(get_filepath('results/models/', "bert_fine_tuned_squad"))
        model.push_to_hub(config["bert_qa"]["model_name"], use_auth_token=config["bert_qa"]["token"])

    @staticmethod
    def question_answering(model_file, question, context) -> str:
        """
        A method that answers a questions according to the given context and the fine-tuned BERT model
        Args:
            model_file (str): The folder were fine-tuned model is stored
            question (str): The question
            context (str): The context

        Returns:
            str
        """
        device = torch.device("cpu")
        model = BertForQuestionAnswering.from_pretrained(model_file)
        model = model.to(device)
        model.eval()
        tokenizer = eval(config["pretrained"]["tokenizer"])
        inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True).to(device)

        output = model(**inputs)
        start_idx = torch.argmax(output.start_logits)
        end_idx = torch.argmax(output.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][start_idx:end_idx]))

        return answer
