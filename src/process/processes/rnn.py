# Copyright 2025, The Johns Hopkins University Applied Physics Laboratory LLC
# Distributed under the terms of the MIT License.

from typing import Any, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

from src.process.unsupervised.unsupervised_process import UnsupervisedProcess


class RNNModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: str = "tanh",
        dropout: float = 0,
        bidirectional: bool = False,
        epochs: int = 40,
        lr: float = 0.01,
    ):
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.epochs = epochs
        self.lr = lr

        self.model = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            nonlinearity=nonlinearity,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, h_n = self.model(input)

        output = self.fc(output)

        return output


class RNNSeqDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class RNN(UnsupervisedProcess):
    def __init__(self, model_args: Dict[str, Any]) -> None:
        super().__init__(model_args["output_size"])

        self.model_args = model_args

        self.model = RNNModel(**self.model_args)

    def input_data(self, seq, ws):
        inputs = []
        labels = []
        L = len(seq)

        for i in range(L - ws):
            window = seq[i : i + ws]
            label = seq[i + ws : i + ws + 1]
            inputs.append(window)
            labels.append(label)

        return inputs, labels

    def train(self, output: List[List[Any]]) -> None:
        window_size = self.model_args["input_size"]

        train_inputs = []
        labels = []

        for training_example in output:
            train_input, label = self.input_data(training_example, window_size)

            train_inputs.extend(train_input)
            labels.extend(label)

        train_dataset = RNNSeqDataset(train_inputs, labels)

        train_data = DataLoader(
            train_dataset, batch_size=self.model_args["batch_size"], shuffle=True
        )

        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), self.model_args["lr"])

        for i in range(self.model_args["epoch"]):
            for batch in train_data:
                batch_seq, batch_label = batch

                optimizer.zero_grad()

                y_pred = self.model(batch_seq)
                loss = criterion(y_pred, batch_label)

                loss.backward()
                optimizer.step()

    def predict(self, input: List[Any]) -> int:
        y_pred = self.model(input)

        return torch.argmax(y_pred)
