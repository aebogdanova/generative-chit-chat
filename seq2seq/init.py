import torch
from torch import nn
import torch.nn.functional as F
import youtokentome as yttm
from seq2seq.model.encoder import Encoder 
from seq2seq.model.decoder import Decoder
from seq2seq.model.model import EncoderDecoder
from seq2seq.constants import EMB_DIM, VOCAB_SIZE, PAD_INDEX, BOS_INDEX, EOS_INDEX, TOKENIZER_PATH, MAX_LEN

def init_model():
    embedding_layer = nn.Embedding(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=EMB_DIM
    )
    encoder = Encoder(embedding_layer)
    decoder = Decoder(embedding_layer)
    model = EncoderDecoder(encoder, decoder)
    return model

def init_criterion(pad_index=PAD_INDEX):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    return criterion

def init_tokenizer(model_path=TOKENIZER_PATH):
    tokenizer = yttm.BPE(model=model_path)
    return tokenizer

def init_collater(tokenizer, pad_index=PAD_INDEX, max_length=MAX_LEN):
    collater = Collater(tokenizer)
    return collater

def init_searcher(model):
    searcher = GreedySearchDecoder(model)
    return searcher

class Collater:
    
    def __init__(self, tokenizer, pad_index=PAD_INDEX, max_length=MAX_LEN):
        self.tokenizer = tokenizer
        self.pad_index = pad_index
        self.max_length = max_length

    def tokenize(self, texts): 
        return self.tokenizer.encode(texts, bos=True, eos=True)

    def padding(self, texts_tokenized):
        sequences = [sequence[:self.max_length] for sequence in texts_tokenized]
        pads = [[self.pad_index] * (self.max_length - len(sequence)) for sequence in sequences]
        return [sequence + pad for sequence, pad in zip(sequences, pads)]
    
    def __call__(self, batch):
        questions, responses = list(), list()
        for question, response in batch:
            questions.append(question)
            responses.append(response)
        questions_tokenized = self.padding(self.tokenize(questions))
        responses_tokenized = self.padding(self.tokenize(responses))
        questions_tensor = torch.LongTensor(questions_tokenized)
        responses_tensor = torch.LongTensor(responses_tokenized)
        return questions_tensor, responses_tensor

class GreedySearchDecoder(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_seq, max_length):
        input_seq = input_seq
        encoder_output, encoder_memory = self.model.encoder(input_seq)
        decoder_states = encoder_memory
        decoder_input = torch.ones(1, 1, dtype=torch.long) * BOS_INDEX
        all_tokens = torch.zeros(1, 1, dtype=torch.long)
        for i in range(max_length):
            decoder_output, decoder_states = self.model.decoder(decoder_input, decoder_states, encoder_output)
            decoder_logits = self.model(decoder_output)
            decoder_output_distribution = F.softmax(decoder_logits, dim=2)
            decoder_input = decoder_output_distribution.argmax(dim=2)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=1)
            if decoder_input[0][0] == EOS_INDEX:
                break
        return all_tokens