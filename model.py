from tokenize import group
from typing import Dict

import torch
from torch.nn import Embedding, RNN, LSTM, GRU, Dropout, Linear, Sequential, ReLU
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
            # batch_first=True,
            bidirectional=bidirectional, 
        )
        
        self.dropout = Dropout(self.dropout)
        # Fully connected linear layer that converts final hidden state to output 
        self.hidden2out = Linear(2*self.hidden_size, self.num_class) if self.bidirectional else Linear(self.hidden_size, self.num_class)

    # lstm_output : [batch_size, seq_len, hidden_size * num_directions(=2)], F matrix
    # final_state: [batch_size, hidden_size*2]
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size * 2, 1) # add 2nd dim
        # hidden: [batch_size, hidden_size*2, 1]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
         # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context # context : [batch_size, n_hidden * num_directions(=2)]


    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return 2*self.hidden_size if self.bidirectional else self.hidden_size
        raise NotImplementedError

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        batch_size = batch.size(0)

        # Initialize hidden state and cell state with zeros
        # h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()
        # c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_()

        # print("Now: {}. After transpose {}".format(batch.size(), batch.t().size()))
        # batch: tensor(batch_size, seq_len)
        # batch.t(): tensor(seq_len, batch_size,)
        embeds = self.embed(batch.t())
        # embeds: tensor(seq_len, batch_size, num_class*2 if bidirectional else num_class)
        # print("==Embedding Layer==")
        # print("Now: {}".format(embeds.size()))
        embeds = self.dropout(embeds)
        
        lstm_out, (hn, _) = self.lstm(embeds)
        # lstm_out: tensor(seq_len, batch_size, 2*hidden_size = 1024 if bidirectional else hidden_size)
        # hn: (2 if bi * num_layer, batch_size, hidden_size) = (4,128,512)

        # print(f"==LSTM==\nNow: lstm_out {lstm_out.size()}, hn {hn.size()}")

        if(self.bidirectional):
            hn = torch.cat((hn[-1],hn[-2]), axis = 1) # hn[-1], hn[-2]: top-most layer for forward/backward hidden state
        else:
            hn = hn[-1]
        # print(f"hn: {hn.size()}")
        # hn: (batch_size, hidden_size * 2) = (128, 1024)

        lstm_out = lstm_out.permute(1,0,2)
        attn_out = self.attention_net(lstm_out, hn)
        # should be (batch_size, hidden_size*2 if bi else hidden_size)

        # use the result from the last lstm layer (containing all timesteps), which is lstm[-1], to predict
        # out = self.hidden2out(lstm_out[-1].view(batch_size, -1))
        out = self.dropout(attn_out)
        out = self.hidden2out(out)

        # out = self.hidden2out(lstm_out[-1])
        # out: (batch_size, num_class)
        # print("==Fully Connected==\nNow: {}\n".format(out.size()))
        
        return out


class SeqTagger(SeqClassifier):
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
        self.num_class = num_class + 1 # class + padding

        # self.lstm = LSTM(
        # self.lstm = GRU(
        self.lstm = RNN(
            input_size=embeddings.shape[1], # embedding dim
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=dropout, 
            bidirectional=bidirectional, 
        )
        
        # Fully connected linear layer that converts final hidden state to output 
        self.hidden2out = Linear(2*self.hidden_size, self.num_class) if self.bidirectional else Linear(self.hidden_size, self.num_class)
        # self.hidden2out = Sequential(
        #     Linear(2*self.hidden_size, 200) if self.bidirectional else Linear(self.hidden_size, 200),
        #     ReLU(),
        # Linear(200, self.num_class),
        # )

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        batch_size = batch.size(0)
        # print("Now: {}. After transpose {}".format(batch.size(), batch.t().size()))
        embeds = self.embed(batch.t()) 
        # print("==Embedding Layer==")
        # print("Now: {}".format(embeds.size()))
        lsrm_out, _ = self.lstm(embeds)
        # lsrm_out: tensor(seq_len, batch_size, 2*hidden_size if bidirectional else hidden_size)
        # print("==GRU==\nNow: {}".format(lsrm_out.size()))
        out = self.hidden2out(lsrm_out.permute(1,0,2).contiguous())
        # print("==Fully Connected==\nNow: {}\n".format(out.size()))
        # out: tensor(batch_size=128, seq_len=26, class=9)

        return out
        # return F.log_softmax(out, dim=2)
