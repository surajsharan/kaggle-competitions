import torch


class DatasetRetriever(torch.utils.data.Dataset):
    def __init__(self, features,test=False, inference=False):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.inference = inference
        self.test = test

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.inference:
            
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                'sequence_ids': feature['sequence_ids'],
                'id': feature['example_id'],
                'context': feature['context'],
                'question': feature['question']
            } 
            
        elif self.test:
            
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
                'sequence_ids': feature['sequence_ids'],
                'id': feature['example_id'],
                'context': feature['context'],   
                'answer_text': feature['answer_text']
                
                }
        else:
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                'end_position': torch.tensor(feature['end_position'], dtype=torch.long)
            }


class Datasetloader:
    def __init__(self, features, test,inference):
        self.features = features
        self.test = test
        self.inference= inference
        self.dataset = DatasetRetriever(features=self.features,test=self.test, inference=self.inference)

    def fetch(self, batch_size, num_workers, drop_last=False, shuffle=True):
        if not self.test:
            sampler = torch.utils.data.RandomSampler(self.dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(self.dataset)

        data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        return data_loader
