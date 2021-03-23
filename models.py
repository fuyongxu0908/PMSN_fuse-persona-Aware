import torch
import torch.nn as nn
import math
from model.utils import get_bow, get_rnn_encode, get_bi_rnn_encode
import re
import numpy as np


class BaseModel(object):


    def print_model_stats(tvars):
        total_parameters = 0
        for variable in tvars:
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
            total_parameters += variable_parametes
        print("Total number of trainable parameters is %d" % total_parameters)

    def print_loss(prefix, loss_names, losses, postfix):
        template = "%s "
        for name in loss_names:
            template += "%s " % name
            template += " %f "
        template += "%s"
        template = re.sub(' +', ' ', template)
        avg_losses = []
        values = [prefix]

        for loss in losses:
            values.append(np.mean(loss))
            avg_losses.append(np.mean(loss))
        values.append(postfix)

        print(template % tuple(values))
        return avg_losses

class PMSN(nn.Module):
    def __init__(self, config, corpus):
        super(PMSN, self).__init__()
        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.vocab_size = len(self.vocab)
        self.idf = corpus.index2idf
        self.gen_vocab_size = corpus.gen_vocab_size
        self.topic_vocab = corpus.topic_vocab
        self.topic_vocab_size = len(self.topic_vocab)
        self.da_vocab = corpus.dialog_act_vocab
        self.da_vocab_size = len(self.da_vocab)
        self.max_utt_len = config.max_utt_len
        self.max_per_len = config.max_per_len
        self.max_per_line = config.max_per_line
        self.max_per_words = config.max_per_words
        self.go_id = self.rev_vocab["<s>"]
        self.eos_id = self.rev_vocab["</s>"]
        self.context_cell_size = config.cxt_cell_size
        self.sent_cell_size = config.sent_cell_size
        self.memory_cell_size = config.memory_cell_size
        self.dec_cell_size = config.dec_cell_size
        self.hops = config.hops
        self.batch_size = config.batch_size
        self.test_samples = config.test_samples
        self.balance_factor = config.balance_factor
        self.sent_type = config.sent_type
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.keep_prob = config.keep_prob

        self.embedding = nn.Embedding(self.vocab_size, config.embed_size)
        self.pos_embeddings = nn.Embedding(config.n_pos_embeddings+1, config.embed_size, padding_idx=self.rev_vocab["<pad>"])


    def Persona_Memory(self, input_embedding, persona_input_embedding):
        embedding_mask = torch.tensor([0 if i == 0 else 1 for i in range(self.vocab_size)])
        A = torch.tensor([self.vocab_size, self.memory_cell_size])
        A = A * embedding_mask
        C = []
        m_0 = input_embedding
        for hopn in range(self.hops):
            pass

    def forward(self, train_feed):
        embedding_mask = torch.tensor([0 if i == 0 else 1 for i in range(self.vocab_size)])
        utts = train_feed['new_utts']
        persona = train_feed['new_persona']
        persona_words = train_feed['new_persona_word']
        con_input_vec = []
        per_input_vec = []
        for sent in utts:
            sent_vec = []
            for wd in sent:
                sent_vec.append(self.word2vec[wd])
            con_input_vec.append(sent_vec)
        for sent in persona:
            sent_vec = []
            for wd in sent:
                sent_vec.append(self.word2vec[wd])
            per_input_vec.append(sent_vec)
        context_input_embedding = torch.tensor(con_input_vec) * embedding_mask
        context_output_embedding = torch.tensor(con_input_vec) * embedding_mask
        persona_input_embedding = torch.tensor(per_input_vec) * embedding_mask

        if self.sent_type == "bow":
            input_embedding, sent_size = get_bow(context_input_embedding)
            output_embedding, _ = get_bow(context_input_embedding)
            persona_input_embedding, _ = get_bow(persona_input_embedding)

        elif self.sent_type == "rnn":
            sent_cell = nn.GRUCell(self.vocab_size, self.hidden_size)
            input_embedding, sent_size = get_rnn_encode(sent_cell, context_input_embedding, self.hidden_size)

        elif self.sent_type == "bi_rnn":
            input_embedding = get_bi_rnn_encode(context_input_embedding, self.hidden_size)
            output_embedding, _ = get_bi_rnn_encode(input_embedding, self.hidden_size)
            persona_input_embedding, _ = get_bi_rnn_encode(persona_input_embedding, self.hidden_size)
        else:
            raise ValueError("Unknown sent_type. Must be one of [bow, rnn, bi_rnn]")

        if self.keep_prob < 1.0:
            input_embedding = nn.Dropout(input_embedding, self.keep_prob)

        self.Persona_Memory(input_embedding, persona_input_embedding)





        input_embedding = torch.tensor([])
        persona_input_embedding = ''



        for hopn in range(self.hops):
            if hopn == 0:
                pass

        pass





