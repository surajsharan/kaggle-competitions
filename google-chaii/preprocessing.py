import config
import collections
import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd
from string import punctuation
import itertools


def get_features(example, tokenizer, inference=False):
    example["question"] = example["question"].lstrip()
    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=config.MAX_SEQ_LEN,
        stride=config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    features = []

    if not inference:

        sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_example.pop("offset_mapping")
        
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            feature = {}
            
            feature['context'] = example['context']
            feature['answer_text']= example['answer_text']
            feature["example_id"] = example['id']
            
            feature['sequence_ids'] = [
                0 if i is None else i for i in tokenized_example.sequence_ids(i)]
            feature['question'] = example['question']
            
            

            input_ids = tokenized_example["input_ids"][i]
            attention_mask = tokenized_example["attention_mask"][i]

            feature['input_ids'] = input_ids
            feature['attention_mask'] = attention_mask
            feature['offset_mapping'] = offsets

            cls_index = input_ids.index(tokenizer.cls_token_id)
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_example.sequence_ids(i)
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = example["answers"]
            
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                feature["start_position"] = cls_index
                feature["end_position"] = cls_index
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    feature["start_position"] = cls_index
                    feature["end_position"] = cls_index
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    feature["start_position"] = token_start_index - 1
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    feature["end_position"] = token_end_index + 1
            features.append(feature)
    else:
        for i in range(len(tokenized_example["input_ids"])):
            feature = {}
            feature["example_id"] = example['id']
            feature['context'] = example['context']
            feature['question'] = example['question']
            feature['input_ids'] = tokenized_example['input_ids'][i]
            feature['attention_mask'] = tokenized_example['attention_mask'][i]
            feature['offset_mapping'] = tokenized_example['offset_mapping'][i]
            feature['sequence_ids'] = [
                0 if i is None else i for i in tokenized_example.sequence_ids(i)]
            features.append(feature)

    return features



##actual
# def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
#     all_start_logits, all_end_logits = raw_predictions
    
#     example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
#     features_per_example = collections.defaultdict(list)
#     for i, feature in enumerate(features):
#         features_per_example[example_id_to_index[feature["example_id"]]].append(i)

#     predictions = collections.OrderedDict()

#     print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

#     for example_index, example in examples.iterrows():
#         feature_indices = features_per_example[example_index]

#         min_null_score = None
#         valid_answers = []
        
#         context = example["context"]
#         for feature_index in feature_indices:
#             start_logits = all_start_logits[feature_index]
#             end_logits = all_end_logits[feature_index]

#             sequence_ids = features[feature_index]["sequence_ids"]
#             context_index = 1

#             features[feature_index]["offset_mapping"] = [
#                 (o if sequence_ids[k] == context_index else None)
#                 for k, o in enumerate(features[feature_index]["offset_mapping"])
#             ]
#             offset_mapping = features[feature_index]["offset_mapping"]
#             cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
#             feature_null_score = start_logits[cls_index] + end_logits[cls_index]
#             if min_null_score is None or min_null_score < feature_null_score:
#                 min_null_score = feature_null_score

#             start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
#             end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
#             for start_index in start_indexes:
#                 for end_index in end_indexes:
#                     if (
#                         start_index >= len(offset_mapping)
#                         or end_index >= len(offset_mapping)
#                         or offset_mapping[start_index] is None
#                         or offset_mapping[end_index] is None
#                     ):
#                         continue
#                     # Don't consider answers with a length that is either < 0 or > max_answer_length.
#                     if end_index < start_index or end_index - start_index + 1 > max_answer_length:
#                         continue

#                     start_char = offset_mapping[start_index][0]
#                     end_char = offset_mapping[end_index][1]
#                     valid_answers.append(
#                         {
#                             "score": start_logits[start_index] + end_logits[end_index],
#                             "text": context[start_char: end_char]
#                         }
#                     )
        
#         if len(valid_answers) > 0:
#             best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
#         else:
#             best_answer = {"text": "", "score": 0.0}
        
#         predictions[example["id"]] = best_answer["text"]
        
        
#     return predictions



def _get_jaccard_score(data, start_logits0,end_logits0,tokenizer,n_best_size = 20,max_answer_length = 30):

    example_id_to_index = {k: i for i, k in enumerate(list(dict.fromkeys(data['id'])))}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(data['id']):
        features_per_example[example_id_to_index[feature]].append(i)

    predictions = collections.OrderedDict()

    start_logits_stack = np.vstack(start_logits0.detach().cpu().numpy())
    end_logits_stack = np.vstack(end_logits0.detach().cpu().numpy())
    
    jaccard_batch = 0.0
    count_batch = 0
    


    for k,v in example_id_to_index.items():

        feature_indices = features_per_example[v]


        min_null_score = None

        valid_answers = []
        count_batch +=1
        
        context = data['context'][feature_indices[0]]
        true_answer = data['answer_text'][feature_indices[0]]


        for feature_index in feature_indices:
            start_logits = start_logits_stack[feature_index]
            end_logits   = end_logits_stack[feature_index]



            sequence_ids = [ int(i[feature_index].detach().cpu().numpy()) for i in  data['sequence_ids']]
            context_index = 1


            offset_mapping = [ tuple(i) for i in data['offset_mapping'][feature_index].detach().cpu().numpy()]
            offset_mapping = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(offset_mapping)]

            cls_index = data['input_ids'][feature_index].tolist().index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        jaccard_batch += jaccard(true_answer,best_answer["text"])
            
    return jaccard_batch , jaccard_batch/count_batch
        

def new_get_jaccard_score(data, outputs_start,outputs_end,tokenizer,n_best_size = 20,max_answer_length = 30):
    
    example_id_to_index = {k: i for i, k in enumerate(data["id"])}
    outputs_start=outputs_start.cpu().numpy().tolist()
    outputs_end = outputs_end.cpu().numpy().tolist()
    
    start_logits_stack = np.vstack(outputs_start)
    end_logits_stack = np.vstack(outputs_end)
    
    jaccard_batch = 0.0
    
    
    for feature_index in example_id_to_index.values():

        min_null_score = None

        valid_answers = []
        
        context = data['context'][feature_index]
        true_answer = data['answer_text'][feature_index]
        

        
            
        start_logits = start_logits_stack[feature_index]
        end_logits   = end_logits_stack[feature_index]



        sequence_ids = [ int(i[feature_index].detach().cpu().numpy()) for i in  data['sequence_ids']]
        context_index = 1


        offset_mapping = [ tuple(i) for i in data['offset_mapping'][feature_index].detach().cpu().numpy()]
        offset_mapping = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(offset_mapping)]

        cls_index = data['input_ids'][feature_index].tolist().index(tokenizer.cls_token_id)
        feature_null_score = start_logits[cls_index] + end_logits[cls_index]
        if min_null_score is None or min_null_score < feature_null_score:
            min_null_score = feature_null_score

        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                    continue

                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]
                valid_answers.append(
                    {
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char: end_char]
                    }
                )
        
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
            

        jaccard_batch += jaccard(true_answer,best_answer["text"])
            
    return jaccard_batch
        

def safe_div(x,y):
    if y == 0:
        return 1
    return x / y

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return safe_div(float(len(c)) , (len(a) + len(b) - len(c)))


# final evaluate an jem 

def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer,n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    
    valid_ans_all = []

    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in examples.iterrows():
        feature_indices = features_per_example[example_index]

        min_null_score = None
        valid_answers = []
        
        truth_ans = example["answer_text"]
        context = example["context"]
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            
            
            start_prob = F.softmax(torch.tensor(start_logits),dim=0).cpu().numpy()
            end_prob = F.softmax(torch.tensor(end_logits),dim=0).cpu().numpy()

            sequence_ids = features[feature_index]["sequence_ids"]
            context_index = 1

            features[feature_index]["offset_mapping"] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(features[feature_index]["offset_mapping"])
            ]
            offset_mapping = features[feature_index]["offset_mapping"]
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    
                    pred = context[start_char: end_char]
                    
                    pred = " ".join(pred.split())
                    pred = pred.strip(punctuation)
                    
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": pred,
#                             "text": context[start_char: end_char],
#                             "start_index" : start_index,
#                             "end_index" : end_index,
                            "prob" : 0.5* (start_prob[start_index]+ end_prob[end_index])
                        }
                    )
            valid_ans_all.append(
               {
                example["id"] : valid_answers  
               } )        
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = jaccard(best_answer["text"],truth_ans)
        
        
    return np.mean(pd.DataFrame([predictions]),axis=1)[0] , valid_ans_all



def _get_jem_score(valid_ans,n=4):
    jem_predictions= collections.OrderedDict()
    for i in range(len(valid_ans)):

        k =list(valid_ans[i].keys())[0]

        eval_ans_id_sel = sorted(valid_ans[i].get(k), key=lambda x: x["score"], reverse=True)[:4]

        eval_df = pd.DataFrame(eval_ans_id_sel)
        eval_df['jem_score'] = _get_jem(eval_df,n=n)
        eval_df.sort_values(by='jem_score',ascending=False,inplace=True,ignore_index=True)

        jem_predictions[k] =  eval_df['text'][0]
        
    return jem_predictions


def _get_jem(eval_df,n=4):

    prob_score = []
    for prob , jacc_comb in zip (eval_df['prob'].tolist() *n ,list(itertools.product(eval_df['text'].values,eval_df['text'].values)) ):
        str1 , str2 = jacc_comb
        score = jaccard(str1 , str2)
        prob_score.append(prob*score)

    return [sum(prob_score[i:i+n]) for i in range(0,len(prob_score),n)]