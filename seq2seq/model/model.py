from torch import nn
from seq2seq.constants import EMB_DIM, VOCAB_SIZE

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, hidden_size=EMB_DIM, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lm_head = nn.Linear(hidden_size*2, vocab_size)

    def forward(self, decoder_states):
        logits = self.lm_head(decoder_states)
        return logits