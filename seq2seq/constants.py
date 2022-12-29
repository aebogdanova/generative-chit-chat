import torch

EMB_DIM = 512
VOCAB_SIZE = 30_000
HIDDEN_SIZE = 512
NUM_LAYERS = 3
DROPOUT = 0.25
MAX_LEN = 30
BATCH_SIZE = 128
EPOCHS = 3
TOKENIZER_PATH = "seq2seq/tokenizer/pretrained_bpe.model"
DATA_PATH = "seq2seq/data/qa_data.jsonl"

# indicies depend on tokenizer
PAD_INDEX = 0
BOS_INDEX = 2
EOS_INDEX = 3