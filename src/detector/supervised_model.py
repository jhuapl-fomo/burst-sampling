# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Any, Dict
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from src.detector.detector_model import DetectorModel
from src.detector.utils import metrics


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class SupervisedModel(DetectorModel):

    def __init__(self, model_name: str, finetune: bool, batch_size: int = 1, epochs: int = 3) -> None:
        super().__init__()

        self.model_name = model_name

        self.finetune = finetune

        self.epochs = epochs

        self.batch_size = batch_size
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2, ignore_mismatched_sizes=True).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def train_model(self, x_train, y_train, x_test, y_test) -> Dict[str, Any]:
        if self.finetune:
            self.fine_tune_model(x_train, y_train)


        train_preds = self.get_supervised_model_prediction(x_train)
        test_preds = self.get_supervised_model_prediction(x_test)

        y_train_pred_prob = train_preds
        y_train_pred = [round(_) for _ in y_train_pred_prob]

        y_test_pred_prob = test_preds
        y_test_pred = [round(_) for _ in y_test_pred_prob]

        acc_train, precision_train, recall_train, f1_train, auc_train = metrics(y_train, y_train_pred, y_train_pred_prob)
        acc_test, precision_test, recall_test, f1_test, auc_test = metrics(y_test, y_test_pred, y_test_pred_prob)

        del self.model
        torch.cuda.empty_cache()

        return {
            'name': self.model_name,
            'acc_train': acc_train,
            'precision_train': precision_train,
            'recall_train': recall_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'recall_test': recall_test,
            'f1_test': f1_test,
            'auc_test': auc_test
        }

    def get_supervised_model_prediction(self, data):

        self.model.eval()

        with torch.no_grad():
            # get predictions for real
            preds = []
            for start in tqdm(range(0, len(data), self.batch_size)):
                end = min(start + self.batch_size, len(data))
                batch_data = data[start:end]
                batch_data = self.tokenizer(batch_data, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
                preds.extend(self.model(**batch_data).logits.softmax(-1)
                            [:, 1].tolist())
        return preds

    def fine_tune_model(self, x_train, y_train):
        train_encodings = self.tokenizer(x_train, truncation=True, padding='max_length', return_tensors='pt')
        train_dataset = CustomDataset(train_encodings, y_train)

        self.model.train()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

        for epoch in range(self.epochs):
            for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
        
        self.model.eval()





