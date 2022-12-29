import tqdm
import json
import numpy as np
import torch 
import logging
from torch.utils.data import DataLoader
from seq2seq.init import init_model, init_criterion, init_tokenizer, init_collater, init_searcher
from seq2seq.preparation import prepare
from seq2seq.constants import BATCH_SIZE, EPOCHS, DATA_PATH, PAD_INDEX, BOS_INDEX, EOS_INDEX, MAX_LEN

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Experiment:

    def __init__(self, data_path=DATA_PATH, use_gpu=False):
        logger.info("Start initialization")
        self.use_gpu = use_gpu
        self.tokenizer = init_tokenizer()
        self.model = init_model()
        if use_gpu:
            self.model = self.model.cuda()
        self.criterion = init_criterion()
        self.collater = init_collater(self.tokenizer)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)
        self.searcher = init_searcher(self.model)
        self.train_set, self.test_set = prepare(data_path)
        self.train_loader = DataLoader(self.train_set[:1_000_000], batch_size=BATCH_SIZE, shuffle=False, collate_fn=self.collater)
        self.test_set = self.test_set[:200]

        logger.info("Initialization complete")

    def train(self, clip=3.0, verbose=True):

        self.model.train()
        losses = list()
        progress_bar = tqdm(total=len(self.train_loader), disable=not verbose, desc="Train")

        for x, y in self.train_loader:

            loss = 0
            n_items = 0
            encoder_sequences = x
            decoder_sequences = y[:, :-1]
            target_sequences = y[:, 1:]
            if self.use_gpu:
                encoder_sequences = encoder_sequences.cuda()
                decoder_sequences = decoder_sequences.cuda()
                target_sequences = target_sequences.to.cuda()

            self.optimizer.zero_grad()

            encoder_output, encoder_memory = self.model.encoder(encoder_sequences)
            decoder_states = encoder_memory
            decoder_input = decoder_sequences[:, :1]

            for i in range(decoder_sequences.shape[1]):
                decoder_output, decoder_states = self.model.decoder(decoder_input, decoder_states, encoder_output)
                decoder_input = target_sequences[:, i:i+1]  # teacher forcing
                decoder_logits = self.model(decoder_output)
                item_loss = self.criterion(decoder_logits.view(-1, decoder_logits.size(-1)), target_sequences[:, i:i+1].view(-1))
                if not torch.isnan(item_loss):
                    loss += item_loss
                    n_items += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            losses.append(loss.item() / n_items)

            progress_bar.set_postfix(loss=losses[-1], perplexity=np.exp(losses[-1]))
            progress_bar.update()

        progress_bar.close()

        return losses

    def trainEpochs(self, epochs=EPOCHS):

        logger.info("Start train")

        train_losses = list()
        train_perplexities = list()

        for n_epoch in range(1, epochs+1):
            
            epoch_train_losses = self.train()
            mean_train_loss = np.mean(epoch_train_losses)
            
            train_losses.append(epoch_train_losses)
            train_perplexities.append(np.exp(mean_train_loss))
            
            message = f"Epoch: {n_epoch}\n"
            message += f"Train: loss - {mean_train_loss:.4f} | perplexity - {train_perplexities[-1]:.3f}\n"
            
            print(message)

            torch.save(self.model.state_dict(), f"model_states/{n_epoch}state_dict_model.pth")
            torch.save(self.optimizer.state_dict(), f"model_states/{n_epoch}state_dict_optimizer.pth")

            with open(f'{n_epoch}_info.json', 'w') as file_object:
                info = {
                    'message': message,
                    'train_losses': train_losses,
                    'train_perplexities': train_perplexities,
                }
                file_object.write(json.dumps(info, indent=2))

        logger.info(f"Train with loss: {mean_train_loss}")


    def generate(self, max_length=MAX_LEN, pad_index=PAD_INDEX):

        self.model.load_state_dict(torch.load('seq2seq/model_states/3_state_dict_model.pth'))

        test_file = open("test.tsv", "a", encoding="utf-8")
        test_file.write("Question\tGold_Answer\tModel_Answer\n")

        for x, y in self.test_loader:

            tokenized = self.tokenizer.encode(x.lower(), bos=True, eos=True)
            padded = tokenized + [pad_index] * (max_length - len(tokenized))
            input_tensor = torch.LongTensor(padded).unsqueeze(0)
        
            tokens = self.searcher(input_tensor, max_length)
            if self.use_gpu:
                tokens = tokens.cuda()

            output_text = self.tokenizer.decode(tokens.tolist(), ignore_ids=[PAD_INDEX, BOS_INDEX, EOS_INDEX])[0]

            test_file.write(f"{x}\t{y}\t{output_text}\n")