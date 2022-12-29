from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from seq2seq.constants import EMB_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

class Encoder(nn.Module):

    def __init__(self, embedding_layer, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_p=DROPOUT):
        super().__init__()
        self.embedding_layer = embedding_layer   
        self.RNN = nn.LSTM(input_size=EMB_DIM,
                           hidden_size=hidden_size, 
                           num_layers=num_layers, 
                           batch_first=True,
                           dropout=dropout_p)

    def forward(self, x):
        input_lengths = (x != self.pad_index).long().sum(dim=1)
        input_lengths = input_lengths.to("cpu")
        x_embedded = self.embedding_layer(x)
        x_packed = pack_padded_sequence(x_embedded, input_lengths, batch_first=True, enforce_sorted=False)
        x_packed, memory = self.RNN(x_packed)
        x_output, _ = pad_packed_sequence(x_packed, batch_first=True)
        return x_output, memory