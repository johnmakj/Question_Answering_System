import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class CustomQuestionAnsweringModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomQuestionAnsweringModel, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # 2 output units for start and end positions
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            **kwargs
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        sequence_output = outputs.last_hidden_state
        logits = self.qa_outputs(sequence_output)

        # Apply softmax activation to logits
        start_probs = torch.nn.functional.softmax(logits[:, :, 0], dim=-1)
        end_probs = torch.nn.functional.softmax(logits[:, :, 1], dim=-1)

        # Calculate log-probabilities for numerical stability
        start_log_probs = torch.nn.functional.log_softmax(logits[:, :, 0], dim=-1)
        end_log_probs = torch.nn.functional.log_softmax(logits[:, :, 1], dim=-1)

        # Calculate the cross-entropy loss
        loss = None

        if start_positions is not None and end_positions is not None:
            if (start_positions >= 0).all() and (end_positions < input_ids.size(1)).all():
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_log_probs, start_positions)
                end_loss = loss_fct(end_log_probs, end_positions)
                loss = (start_loss + end_loss) / 2
            else:
                # Handle the case where start_positions or end_positions are out of range
                print("Problem")
                start_positions = torch.clamp(start_positions, 0, input_ids.size(1) - 1)
                end_positions = torch.clamp(end_positions, 0, input_ids.size(1) - 1)
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_log_probs, start_positions)
                end_loss = loss_fct(end_log_probs, end_positions)
                loss = (start_loss + end_loss) / 2

        # Return probabilities along with the log-probabilities and the loss during training
        if self.training:
            return (start_probs, end_probs, start_log_probs, end_log_probs), loss
        else:
            return start_probs, end_probs
