import pandas as pd
import numpy as np
from tqdm import tqdm

import config
import dataset
import preprocessing
from transformers import AutoTokenizer ,AutoConfig
from model import XLM_RoBertamodel
import engine
import gc

## two approaches for ensembling 1)take the logits from models and the take mean , median, weighted etc
## 2) store the predictions from all the models and then take the mode of them [max no of time which occurs]


def get_prediction(checkpoint_path):
    
    model_config = AutoConfig.from_pretrained(config.CONFIG_NAME)
    model = XLM_RoBertamodel(config.MODEL_PATH, config=model_config)
    model = nn.DataParallel(model)
    
    model.cuda();
    model.load_state_dict(torch.load((base_model_path + checkpoint_path) ,map_location=torch.device('cuda')));
    
    start_logits = []
    end_logits = []

    for batch in test_dataloader:
        with torch.no_grad():
            outputs_start, outputs_end,_ = model(batch['input_ids'].cuda(), batch['attention_mask'].cuda())
            start_logits.append(outputs_start.cpu().numpy().tolist())
            end_logits.append(outputs_end.cpu().numpy().tolist())
            del outputs_start, outputs_end
    del model, model_config
    gc.collect()
    return np.vstack(start_logits), np.vstack(end_logits)
            
    
    
def inference():
    
    test = pd.read_csv('input/train_folds.csv')
    base_model_path = 'output/xlm_roberta_v1/'

    
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)
    
    test_features = []
    for i, row in test.iterrows():
        test_features += preprocessing.get_features(row, tokenizer,inference=True)
        
    
    test_dataloader = dataset.Datasetloader(features=test_features, test=True,inference=True).fetch(
        batch_size=config.VALID_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=False)
    
    start_logits0, end_logits0 = get_predictions('checkpoint-fold--1/pytorch_model.bin')
    
    predictions = preprocessing.postprocess_qa_predictions(test, test_features, (start_logits0, end_logits0))

    test['PredictionString'] = test['id'].map(predictions)
    test.to_csv('export/submission.csv', index=False)


    
if __name__ == "__main__":
    
    inference()