
from transformers import AutoModel
import torch
import torch.nn as nn
import config


# class XLM_RoBertamodel(nn.Module):
#     def __init__(self, modelname_or_path, config):
#         super(XLM_RoBertamodel, self).__init__()
#         self.config = config
#         self.xlm_roberta = AutoModel.from_pretrained(
#             modelname_or_path, config=config)
        
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.qa_outputs = nn.Linear(config.hidden_size, 2)
#         self._init_weights(self.qa_outputs)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(
#                 mean=0.0, std=self.config.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()

#     def forward(self,
#                 input_ids,
#                 attention_mask=None,
#                 start_position=None,
#                 end_position=None,
#                 offset_mapping=None,
                
#                 # token_type_ids=None
#                 ):

#         outputs = self.xlm_roberta(input_ids, attention_mask=attention_mask,)
#         sequence_output = outputs[0]
# #         pooled_output = outputs[1]

#         sequence_output = self.dropout(sequence_output)
#         qa_logits = self.qa_outputs(sequence_output)

#         start_logits, end_logits = qa_logits.split(1, dim=-1)
#         start_logits = start_logits.squeeze(-1).contiguous()
#         end_logits = end_logits.squeeze(-1).contiguous()

#         loss = None
#         if (start_position is not None) & (end_position is not None):
#             loss = loss_fn((start_logits, end_logits),
#                            (start_position, end_position))
#             loss = loss / config.GRADIENT_ACC_STEPS

#         return start_logits, end_logits, loss



    
    
## ADAPTED

class XLM_RoBertamodel(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(XLM_RoBertamodel, self).__init__()
        self.config = config
        config.update({
            "layer_norm_eps": 1e-7,
            "output_hidden_states": True
            }) 
        self.xlm_roberta = AutoModel.from_pretrained(
            modelname_or_path, config=config)
        
        
        
        self.high_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size * 2, 2)

        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                input_ids,
                attention_mask=None,
                start_position=None,
                end_position=None,
                offset_mapping=None,
                
                # token_type_ids=None
                ):
        
        
        out = self.xlm_roberta(input_ids,attention_mask=attention_mask,)
        LAST_HIDDEN_LAYERS = 12

        out = out.hidden_states 
        out = torch.stack(tuple(out[-i - 1] for i in range(LAST_HIDDEN_LAYERS)), dim=0)
       
        out_mean = torch.mean(out, dim=0) 
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1) 


        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([self.qa_outputs(self.high_dropout(out))for _ in range(5)], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

       
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if (start_position is not None) & (end_position is not None):
            loss = loss_fn((start_logits, end_logits),
                           (start_position, end_position))
            loss = loss / config.GRADIENT_ACC_STEPS

        return start_logits, end_logits, loss
    


    

    
def loss_fn(preds, labels):
    
    start_preds, end_preds = preds
    start_labels, end_labels = labels

    start_loss = nn.CrossEntropyLoss(
        ignore_index=-1)(start_preds, start_labels)
    end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_preds, end_labels)
    total_loss = (start_loss + end_loss)/2

    return total_loss


# def pos_weight(pred_tensor, pos_tensor, neg_weight=1, pos_weight=1):
#     # neg_weight for when pred position < target position
#     # pos_weight for when pred position > target position
#     gap = torch.argmax(pred_tensor, dim=1) - pos_tensor
#     gap = gap.type(torch.long)
#     return torch.where(gap < 0, -neg_weight * gap, pos_weight * gap)


    
# def loss_fn(preds, labels):
    
#     start_logits, end_logits = preds
#     start_position, end_position = labels
    
    
#     loss_fct = nn.CrossEntropyLoss(reduce='none')
    
#     start_loss = loss_fct(start_logits, start_position) * pos_weight(start_logits, start_position, 1, 1)
#     end_loss = loss_fct(end_logits, end_position) * pos_weight(end_logits, end_position, 1, 1)
    
#     start_loss = torch.mean(start_loss)
#     end_loss = torch.mean(end_loss)
    
#     total_loss = (start_loss + end_loss)/2
#     return total_loss
