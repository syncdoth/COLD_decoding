from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.utils import logging
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers import GPT2ForSequenceClassification

logger = logging.get_logger(__name__)


class AttributeClassifier(GPT2ForSequenceClassification):
    """
    adapted from huggingface transformers v.4.22.1

    added option of pooling over hidden states: max, mean, last
    """

    def pool_hidden_states(self,
                           hidden_states: torch.LongTensor,
                           batch_size: int,
                           input_ids: Optional[torch.LongTensor] = None,
                           pool_method: str = 'last'):
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                real_sequence = torch.ne(input_ids, self.config.pad_token_id)
                sequence_lengths = real_sequence.sum(-1) - 1
            else:
                real_sequence = torch.ones(hidden_states.shape[:-1]).bool()
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
        if pool_method == 'last':
            pooled_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        elif pool_method in ('mean', 'max'):
            real_hidden_states = hidden_states[real_sequence]  # [-1, E]
            real_states_split = torch.split(real_hidden_states, (sequence_lengths + 1).tolist())  # B * [T, E]
            if pool_method == 'mean':
                # [B, E]
                pooled_states = torch.stack([state.mean(0) for state in real_states_split], dim=0)
            else:
                pooled_states = torch.stack([state.max(0)[0] for state in real_states_split], dim=0)
        else:
            raise NotImplementedError(f'pool method `{pool_method}` not supported.'
                                      f'please choose from [max, mean, last (default)].')

        return pooled_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pool_method: Optional[str] = 'last',
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."

        pooled_states = self.pool_hidden_states(hidden_states, batch_size, input_ids=input_ids, pool_method=pool_method)
        pooled_logits = self.score(pooled_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
