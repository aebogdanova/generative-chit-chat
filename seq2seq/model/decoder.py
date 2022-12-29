import torch 
from torch import nn
from seq2seq.constants import EMB_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

class Decoder(nn.Module):

    def __init__(self, embedding_layer, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout_p=DROPOUT):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.RNN = nn.LSTM(input_size=EMB_DIM,
                           hidden_size=hidden_size, 
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout_p)

    def forward(self, x, encoder_memory, encoder_output):
        x_embedded = self.embedding_layer(x)
        decoder_outputs, memory = self.RNN(x_embedded, encoder_memory)
        # attention
        attention_scores = torch.bmm(decoder_outputs, encoder_output.transpose(1, 2))
        attention_distribution = torch.softmax(attention_scores, 2)
        attention_vectors = torch.bmm(attention_distribution, encoder_output)
        decoder_with_attention = torch.cat([decoder_outputs, attention_vectors], dim=-1)
        return decoder_with_attention, memory