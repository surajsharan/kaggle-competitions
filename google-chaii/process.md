## Strategies to try 
2) only train one model to see what it does and what the distribution is on the test set 
3) Train a model differently for diff lan [use muril]



# todo

1) Use a modified loss function

    def loss_fn(start_logits, end_logits,
            start_positions, end_positions):
    m = torch.nn.LogSoftmax(dim=1)
    loss_fct = torch.nn.KLDivLoss()
    start_loss = loss_fct(m(start_logits), start_positions)
    end_loss = loss_fct(m(end_logits), end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss



2) while post processing play with the best score in order to tune the answers (removing the junk characters say['.,!~'] some special characters from text)
3) read about the character level word matching 
4) level 2 model in order to fine tune the model

5) Model architechture : multiple dropout 
     [multisample dropout (wut): https://arxiv.org/abs/1905.09788]
            logits = torch.mean(
                torch.stack(
                    [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
            
    
 6) re-reanking mechanism (start value + end value) * 0.5 + predicted jaccard score
 
 7) jaccard expectation maximization [done]
 
 8) replace special characters in tokenizer
 
 9) apply this model architucture 
     Model-1
        Concat([last 2 hidden_layers from BERT]) -> Conv1D -> Linear
        End position depends on start (taken from here), which looks like,
        # x_head, x_tail are results after Conv1D
        logit_start = linear(x_start)
        logit_end = linear(torch.cat[x_start, x_end], dim=1)
        
        
## apply :
https://github.com/heartkilla/kaggle_tweet/blob/master/src/1st_level/roberta_distil/models.py