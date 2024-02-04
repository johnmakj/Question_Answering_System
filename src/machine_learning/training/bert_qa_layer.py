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

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        loss = None
        """
        if start_positions is not None and end_positions is not None:
            start_positions = torch.clamp(start_positions, 0, input_ids.size(1) - 1)
            end_positions = torch.clamp(end_positions, 0, input_ids.size(1) - 1)
            # If start and end positions are provided, calculate the loss
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2
        """

        if start_positions is not None and end_positions is not None:

            if (start_positions >= 0).all() and (end_positions < input_ids.size(1)).all():
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2
            else:
                # Handle the case where start_positions or end_positions are out of range
                start_positions = torch.clamp(start_positions, 0, input_ids.size(1) - 1)
                end_positions = torch.clamp(end_positions, 0, input_ids.size(1) - 1)
                loss_fct = nn.CrossEntropyLoss()
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                loss = (start_loss + end_loss) / 2

        # Return logits along with the loss during training
        if self.training:
            return (start_logits, end_logits), loss
        else:
            return start_logits, end_logits
