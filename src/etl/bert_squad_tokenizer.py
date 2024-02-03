from typing import Tuple, List, Any

import pandas as pd
import torch

from torch.utils.data import Dataset


class SquadDataset(Dataset):
    def __init__(self, df_squad: pd.DataFrame) -> None:
        """
        Class constructor.
        """
        self.data = df_squad
        self.data_cols = ["question", "context", "answer_starts", "answer_ends"]

    def __len__(self):
        """
        Get the dataset length.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """
        Get the question, context paragraph, answer start and answer end value
        at the row specified by the input index from the dataset.
        Args:
           index(int): The index from data to retrieve

        Returns:
            tuple
       """
        return tuple(self.data.loc[index, self.data_cols])


class SquadTokenizer:
    def __init__(self, tokenizer, max_len=512):
        """
        Store the input BertTokenizer instance and the max length
        """
        self.tokenizer = tokenizer
        self.max_length = max_len

    def __call__(self, batch: Tuple[List[str], List[str], List[int], List[int]]) -> (
            Tuple)[Any, List[str], List[int]]:
        """
        Perform tokenization on a batch of data

        Args:
            batch (Tuple[questions, contexts, answer_starts, answer_ends]):
                questions (List[str]) : a list of questions
                contexts (List[str]) : a list of context paragraphs
                answer_starts (List[int]) : a list of answer start indexes
                answer_ends (List[int]) : a list of answer end indexes

        Returns:
            Tuple
        """

        questions = list(batch[0])
        contexts = list(batch[1])
        encoding = self.tokenizer(questions, contexts, padding="longest", truncation=True, max_length=self.max_length,
                                  return_tensors='pt')

        def get_index(indexes):
            final = []
            for i in range(len(indexes)):

                if encoding.char_to_token(i, char_index=indexes[i], sequence_index=1) is not None:
                    temp = encoding.char_to_token(i, char_index=indexes[i], sequence_index=1)
                    final.append(temp)
                elif encoding.char_to_token(i, char_index=indexes[i] + 1, sequence_index=1) is not None:
                    temp = encoding.char_to_token(i, char_index=indexes[i] + 1, sequence_index=1)
                    final.append(temp)
                elif encoding.char_to_token(i, char_index=indexes[i] - 1, sequence_index=1) is not None:
                    temp = encoding.char_to_token(i, char_index=indexes[i] - 1, sequence_index=1)
                    final.append(temp)
                else:
                    final.append(self.max_length)

            return final

        answer_starts = torch.LongTensor(batch[2])
        token_starts = get_index(answer_starts)
        answer_ends = torch.LongTensor(batch[3])
        token_ends = get_index(answer_ends)

        result = (encoding, token_starts, token_ends)
        return result
