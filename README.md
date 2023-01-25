## Generative Chit-Chat

### Data preparation
The model was trained on [Ответы Mail.ru](https://otvet.mail.ru) dataset (1000K examples) tokenized with pre-trained BPE-tokenizer `youtokentome`.

### Model
- Encoder: LSTM
- Decoder: LSTM with Attention Mechanism

To run training:
```
python train.py
```

### Evaluation 
For decoding answers greedy-search decoder is used.
To see results:
```
python test.py
```

### Examples
See examples of model answers in `test.tsv` file.