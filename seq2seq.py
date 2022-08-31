# standard library
import math
from typing import List
import time

# third party
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import pytorch_lightning as pl
import numpy as np
from torch.nn import Transformer

# project
from utils import logger


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 3000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(1), :]
        )


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(pl.LightningModule):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        configuration,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.configuration = configuration
        self.sos = configuration.num_classes + configuration.num_nucleobase_letters
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor = None,
        tgt_padding_mask: Tensor = None,
        memory_key_padding_mask: Tensor = None,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        outs = self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )
        return self.generator(outs)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(
            0, 1
        )
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(
            torch.bool
        )
        return src_mask, tgt_mask

    def create_tgt_mask(self, tgt):
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        return tgt_mask

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        tar_input = targets[:, :-1]
        src_mask, tgt_mask = self.create_mask(samples, tar_input)
        tar_output = targets[:, 1:]
        logits = self.forward(samples, tar_input, src_mask, tgt_mask)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tar_output.reshape(-1))
        self.log("train_loss", loss)
        return loss

    def on_test_start(self):
        self.targets = torch.empty(0).to(self.device)
        self.predict_targets = torch.empty(0).to(self.device)

    def on_test_end(self):
        if self.configuration.num_sample_predictions > 0:
            with torch.random.fork_rng():
                torch.manual_seed(int(time.time() * 1000))
                permutation = torch.randperm(self.targets.shape[0])
        self.targets = self.targets[
            permutation[0 : self.configuration.num_sample_predictions], :
        ].tolist()

        self.predict_targets = self.predict_targets[
            permutation[0 : self.configuration.num_sample_predictions], :
        ].tolist()

        logger.info("\nsample assignments")
        self.configuration.category_mapper.print_label_and_emoji(logger)
        for target, predict in zip(self.targets, self.predict_targets):
            logger.info(
                "".join(list(map(lambda x: self.class_transform(int(x)), target)))
            )
            logger.info("-------------------------------------------------------")
            logger.info(
                "".join(list(map(lambda x: self.class_transform(int(x)), predict)))
            )

    def class_transform(self, label_index):
        num_nucleobase_letters = (
            self.configuration.dna_sequence_mapper.num_nucleobase_letters
        )
        if label_index < num_nucleobase_letters:
            return self.configuration.dna_sequence_mapper.label_encoding_to_nucleobase_letter(
                label_index
            )
        label_index -= num_nucleobase_letters
        return self.configuration.category_mapper.label_to_emoji(label_index)

    def test_step(self, batch, batch_idx):
        samples, targets = batch
        tar_input = targets[:, :-1]
        src_mask, tgt_mask = self.create_mask(samples, tar_input)
        tar_output = targets[:, 1:]
        logits = self.forward(samples, tar_input, src_mask, tgt_mask)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tar_output.reshape(-1))
        self.log("test_loss", loss)
        predict_class = self.predict(samples)[:, :-1]
        if self.targets.shape[0] > 100:
            self.targets = self.targets[-100:, :]
            self.predict_targets = self.predict_targets[-100:, :]
        self.targets = torch.cat((self.targets, tar_input[:, 1:]))
        self.predict_targets = torch.cat((self.predict_targets, predict_class))

        return loss

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        encoded_x = self.encode(x, None)
        output_tokens = (
            (torch.ones((x.shape[0], 2001))).type_as(x).long()
        )  # (B, max_length)
        output_tokens[:, 0] = self.sos  # Set start token
        for Sy in range(1, 2001):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(y, encoded_x, None)  # (B, Sy, C)
            output = torch.argmax(output, dim=-1)  # (B, Sy)
            output_tokens[:, Sy] = output[:, -1]  # Set the last output token
        return output_tokens

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        tar_input = targets[:, :-1]
        src_mask, tgt_mask = self.create_mask(samples, tar_input)
        tar_output = targets[:, 1:]
        logits = self.forward(samples, tar_input, src_mask, tgt_mask)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tar_output.reshape(-1))
        self.log("val_loss", loss)
        predict_class = self.predict(samples)[:, :-1]

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configuration.lr)
        return optimizer
