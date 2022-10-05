from typing import Dict

import torch
from torch.nn import Embedding, LSTM, Dropout, Linear
import torch.nn.functional as F


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.embed_dim = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        # TODO: model architecture
        self.lstm = LSTM(
            input_size=embeddings.shape[1], # embedding dim
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            batch_first=True,
            bidirectional=bidirectional, 
        )
        
        # Fully connected linear layer that converts final hidden state to output 
        self.hidden2out = Linear(2*self.hidden_size, self.num_class) if self.bidirectional else Linear(self.hidden_size, self.num_class)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return 2*self.hidden_size if self.bidirectional else self.hidden_size
        raise NotImplementedError

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        batch_size = batch.size(0)

        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

        embeds = self.embed(batch.t()) #不確定要不要transpose或維度
        
        # Input: (batch_size, seq_len, input_size)
        lstm_out, (_, _) = self.lstm(embeds)
        # use the newest result h(t), which is lstm[-1], to predict
        # assert 1 == 0, lstm_out.size()
        # out = self.hidden2out(lstm_out[-1].view(batch_size, -1))
        out = self.hidden2out(lstm_out[-1])
        
        return out
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
