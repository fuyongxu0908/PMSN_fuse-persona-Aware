from collections import Counter
import numpy as np
import nltk
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAI2DialogCorpus(object):
    dialog_act_id = 0
    sentiment_id = 1
    liwc_id = 2

    def __init__(self, corpus_path, max_vocab_cnt=20000, word2vec=None, word2vec_dim=None, vocab_files=None,
                 idf_files=None):
        self._path = corpus_path
        self.word_vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None
        self.vocab = None
        self.dialog_id = 0
        self.meta_id = 1
        self.utt_id = 2
        self.persona_id = 3
        self.persona_word_id = 4
        self.sil_utt = ["<s>", "<sil>", "</s>"]
        self.persona_precursor = ["my favorite", "i love", "i like", "i am", "i'm ", "i can", "i enjoy", "i watch",
                                  "i work", "i have", "i've", "i had", "i live", "i do", "i hate", "i believe",
                                  "i have"]
        data = self.reading_convai2_corpus(self._path)
        self.train_corpus = self.process(data["train"])
        self.valid_corpus = self.process(data["valid"])
        self.test_corpus = self.process(data["test"])
        self.build_vocab(max_vocab_cnt, vocab_files, idf_files)
        self.load_word2vec()
        print("Done loading corpus")

    def reading_convai2_corpus(self, path):
        def _read_persona_and_dialogue(name, corpus):
            load_corpus = []
            utts = []
            persona = []
            persona_word = ''
            count = 0
            for l in corpus:
                count += 1
                l = l.strip()
                if l.split(' ')[0] == '1' and count > 1:
                    segments = {'utts': utts, 'persona': persona, 'persona_word': ['<s>' + persona_word + ' </s>']}
                    load_corpus.append(segments)
                    utts = []
                    per = l.split('your persona: ')[1].strip(' .').lower().strip()
                    persona = ['</s>'] + [per + ' </s>']
                    persona_word = (per + ' ')
                else:
                    if 'your persona:' in l:
                        per = l.split('your persona: ')[1].strip(' .').lower().strip()
                        persona.append('<s>' + per + ' </s>')
                        persona_word += (per + ' ')
                    else:
                        if name == 'Null':
                            utt = (' '.join(l.split(' ')[1:])).split('\t')
                            utts.append(('A', utt[0] + ' ' + persona[np.random.randint(len(persona))],
                                         ['None', [0.0, 0.0, 0.0, 0.0]]))
                            utts.append(('B', utt[1] + ' ' + persona[np.random.randint(len(persona))],
                                         ['None', [0.0, 0.0, 0.0, 0.0]]))
                        else:
                            utt = (' '.join(l.split(' ')[1:])).split('\t')
                            utts.append(('A', utt[0], ['None', [0.0, 0.0, 0.0, 0.0]]))
                            utts.append(('B', utt[1], ['None', [0.0, 0.0, 0.0, 0.0]]))
            return load_corpus

        train_corpus = _read_persona_and_dialogue('Train', open(path + 'train.txt'))
        valid_corpus = _read_persona_and_dialogue('Valid', open(path + 'valid.txt'))
        test_corpus = _read_persona_and_dialogue('Test', open(path + 'test.txt'))
        '''
        {'train': [{'utts': [
        ('A', "hi , how are you doing ? i'm getting ready to do some cheetah chasing to stay in shape .", ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('B', 'you must be very fast . hunting is one of my favorite hobbies .', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('A', 'i am ! for my hobby i like to do canning or some whittling .', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('B', 'i also remodel homes when i am not out bow hunting .', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('A', "that's neat . when i was in high school i placed 6th in 100m dash !", ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('B', "that's awesome . do you have a favorite season or time of year ?", ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('A', 'i do not . but i do have a favorite meat since that is all i eat exclusively .', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('B', 'what is your favorite meat to eat ?', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('A', 'i would have to say its prime rib . do you have any favorite foods ?', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('B', 'i like chicken or macaroni and cheese .', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('A', 'do you have anything planned for today ? i think i am going to do some canning .', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('B', 'i am going to watch football . what are you canning ?', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('A', 'i think i will can some jam . do you also play footfall for fun ?', ['None', [0.0, 0.0, 0.0, 0.0]]), 
        ('B', 'if i have time outside of hunting and remodeling homes . which is not much !', ['None', [0.0, 0.0, 0.0, 0.0]])
        ], 
        
        'persona': [
        '<s> i like to remodel homes </s>', 
        '<s> i like to go hunting </s>', 
        '<s> i like to shoot a bow </s>', 
        '<s> my favorite holiday is halloween </s>'], 
        
        'persona_word': ['<s>i like to remodel homes i like to go hunting i like to shoot a bow my favorite holiday is halloween  </s>']}]}

        '''

        return {'train': train_corpus, 'valid': valid_corpus, 'test': test_corpus}

    def process(self, data):
        new_dialog = []
        new_meta = []
        new_utts = []
        all_lens = []
        new_persona = []
        new_persona_word = []

        for l in data:
            lower_utts = [(caller, ["<s>"] + nltk.WordPunctTokenizer().tokenize(utt.lower()) + ["</s>"], feat)
                          for caller, utt, feat in l["utts"]]
            all_lens.extend([len(u) for c, u, f in lower_utts])
            vec_a_meta = [0, 0] + [0, 0]
            vec_b_meta = [0, 0] + [0, 0]
            meta = (vec_a_meta, vec_b_meta, 'NULL')
            dialog = [(utt, int(caller == "A"), feat) for caller, utt, feat in lower_utts]
            new_utts.extend([utt for caller, utt, feat in lower_utts])
            new_dialog.append(dialog)
            new_meta.append(meta)
            new_persona.append([(p.split(' ')) for p in l['persona']])
            new_persona_word.append(l['persona_word'])

        print("Max utt len %d, mean utt len %.2f" % (np.max(all_lens), float(np.mean(all_lens))))
        return new_dialog, new_meta, new_utts, new_persona, new_persona_word

    def build_vocab(self, max_vocab_cnt, vocab_files=None, idf_files=None):

        self.vocab = []
        self.rev_vocab = []
        if vocab_files is None:
            all_words = []
            persona_words = []
            for tokens in self.train_corpus[self.utt_id]:
                all_words.extend(tokens)
            for persona in self.train_corpus[self.persona_id]:
                for p in persona:
                    all_words.extend(p)
            for persona in self.train_corpus[self.persona_id]:
                for p in persona:
                    persona_words.extend(p)

            vocab_count = Counter(all_words).most_common()
            raw_vocab_size = len(vocab_count)
            vocab_count = vocab_count[0:max_vocab_cnt]

            persona_vocab_count = Counter(persona_words).most_common()

            all_vocab_set = set([t for t, cnt in vocab_count]) - set(["<s>", "</s>"])
            persona_vocab_set = set([t for t, cnt in persona_vocab_count if cnt <= -1]) - set(["<s>", "</s>"])
            normal_vocab = ["<pad>", "<unk>"] + ["<s>", "</s>"] + ["<sentinel>"] + list(
                all_vocab_set - persona_vocab_set)
            persona_vocab = list(persona_vocab_set)
            self.gen_vocab_size = len(normal_vocab)
            self.copy_vocab_size = len(persona_vocab)

            # create vocabulary list sorted by count
            print("Build corpus with raw vocab size %d, vocab size %d, gen vocab size %d, copy vocab size %d ."
                  % (raw_vocab_size, self.gen_vocab_size + self.copy_vocab_size, self.gen_vocab_size,
                     self.copy_vocab_size))

            self.vocab = normal_vocab + persona_vocab
            self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
            self.unk_id = self.rev_vocab["<unk>"]
            print("vocab unk_id is %d, pad_id is %d" % (self.unk_id, self.rev_vocab["<pad>"]))
            print("vocab <s> is %d, </s> is %d" % (self.rev_vocab["<s>"], self.rev_vocab["</s>"]))

        else:
            with open(self._path + vocab_files, 'r') as vocab_f:
                for vocab in vocab_f:
                    vocab = vocab.strip()
                    self.vocab.append(vocab)
            self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}
            self.unk_id = self.rev_vocab["<unk>"]
            self.gen_vocab_size = len(self.vocab)
            self.copy_vocab_size = 0
            print("Load corpus from %s with vocab size %d ." % (vocab_files, len(self.vocab)))
            print("vocab unk_id is %d, pad_id is %d" % (self.unk_id, self.rev_vocab["<pad>"]))
            print("vocab <s> is %d, </s> is %d" % (self.rev_vocab["<s>"], self.rev_vocab["</s>"]))

        self.idf = {}
        self.index2idf = [1.0 for _ in range(len(self.vocab))]
        if idf_files is None:
            for vocab in self.vocab:
                self.idf[vocab] = 1.0
            print("All words' IDF are set to 1.0, as idf_files == None")
        else:
            with open(self._path + idf_files, 'r') as idf_f:
                for i, line in enumerate(idf_f):
                    line = line.strip().split('\t')
                    vocab = line[0]
                    idf = float(line[1])
                    self.idf[vocab] = idf
                    self.index2idf[i] = idf
            print("Load words' IDF from %s with size %d ." % (idf_files, len(self.idf)))

        # create topic vocab
        all_topics = []
        for a, b, topic in self.train_corpus[self.meta_id]:
            if topic is not None:
                all_topics.append(topic)
        self.topic_vocab = [t for t, cnt in Counter(all_topics).most_common()]
        self.rev_topic_vocab = {t: idx for idx, t in enumerate(self.topic_vocab)}

        # get dialog act labels
        all_dialog_acts = []
        for dialog in self.train_corpus[self.dialog_id]:
            all_dialog_acts.extend([feat[self.dialog_act_id] for caller, utt, feat in dialog if feat is not None])
        self.dialog_act_vocab = [t for t, cnt in Counter(all_dialog_acts).most_common()]
        self.rev_dialog_act_vocab = {t: idx for idx, t in enumerate(self.dialog_act_vocab)}

        all_personas = []
        for persona in self.train_corpus[self.persona_id]:
            for p in persona:
                all_personas.append(' '.join(p))

        self.persona_precursor_idx = []
        for i in self.persona_precursor:
            tmp = []
            for j in i.split(' '):
                if j != '':
                    tmp.append(self.rev_vocab[j])
            self.persona_precursor_idx.append(tmp)

    def load_word2vec(self):
        if self.word_vec_path is None:
            return
        with open(self.word_vec_path, "rb") as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines:
            w, vec = l.split(" ", 1)
            raw_word2vec[w] = vec
        self.word2vec = {}
        oov_cnt = 0
        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = np.fromstring(str_vec, sep=" ")
            self.word2vec.update({v:vec})
        print("word2vec cannot cover %f vocab" % (float(oov_cnt) / len(self.vocab)))

    def get_utt_corpus(self):
        def _to_id_corpus(data):
            results = []
            for line in data:
                results.append([self.rev_vocab.get(t, self.unk_id) for t in line])
            return results
        id_train = _to_id_corpus(self.train_corpus[self.utt_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.utt_id])
        id_test = _to_id_corpus(self.test_corpus[self.utt_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_dialog_corpus(self):
        def _to_id_corpus(data):
            results = []
            for dialog in data:
                temp = []
                for utt, floor, feat in dialog:
                    if feat is not None:
                        id_feat = list(feat)
                        id_feat[self.dialog_act_id] = self.rev_dialog_act_vocab[feat[self.dialog_act_id]]
                    else:
                        id_feat = None
                    temp.append(([self.rev_vocab.get(t, self.unk_id) for t in utt], floor, id_feat))
                results.append(temp)
            return results

        id_train = _to_id_corpus(self.train_corpus[self.dialog_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.dialog_id])
        id_test = _to_id_corpus(self.test_corpus[self.dialog_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_meta_corpus(self):
        def _to_id_corpus(data):
            results = []
            for m_meta, o_meta, topic in data:
                results.append((m_meta, o_meta, self.rev_topic_vocab[topic]))
            return results

        id_train = _to_id_corpus(self.train_corpus[self.meta_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.meta_id])
        id_test = _to_id_corpus(self.test_corpus[self.meta_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_persona_corpus(self):
        def _to_id_corpus(data):
            results = []
            for i in data:
                session = []
                for j in i:
                    session.append([self.rev_vocab.get(k, self.unk_id) for k in j])
                results.append(session)
            return results

        id_train = _to_id_corpus(self.train_corpus[self.persona_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.persona_id])
        id_test = _to_id_corpus(self.test_corpus[self.persona_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}

    def get_persona_word_corpus(self):
        def _to_id_corpus(data):
            results = []
            for i in data:
                results.append([self.rev_vocab.get(k, self.unk_id) for k in i[0].strip().split(' ')])
            return results

        id_train = _to_id_corpus(self.train_corpus[self.persona_word_id])
        id_valid = _to_id_corpus(self.valid_corpus[self.persona_word_id])
        id_test = _to_id_corpus(self.test_corpus[self.persona_word_id])
        return {'train': id_train, 'valid': id_valid, 'test': id_test}


class DataLoaderBase(object):
    batch_size = 0
    backward_size = 0
    step_size = 0
    ptr = 0
    num_batch = None
    batch_indexes = None
    grid_indexes = None
    indexes = None
    data_lens = None
    data_size = None
    prev_alive_size = 0
    name = None

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def _prepare_batch(self, cur_grid, prev_grid):
        raise NotImplementedError("Have to override prepare batch")

    def epoch_init(self, batch_size, backward_size, step_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size
        self.backward_size = backward_size
        self.step_size = step_size
        self.prev_alive_size = batch_size
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size:(i + 1) * self.batch_size])

        left_over = self.data_size - temp_num_batch * batch_size

        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        # create grid indexes
        self.grid_indexes = []
        for idx, b_ids in enumerate(self.batch_indexes):
            # assume the b_ids are sorted
            all_lens = [self.data_lens[i] for i in b_ids]
            max_len = self.data_lens[b_ids[-1]]
            min_len = self.data_lens[b_ids[0]]
            assert np.max(all_lens) == max_len
            assert np.min(all_lens) == min_len
            num_seg = (max_len - self.backward_size) // self.step_size
            if num_seg > 0:
                cut_start = range(0, num_seg * self.step_size, step_size)
                cut_end = range(self.backward_size, num_seg * self.step_size + self.backward_size, step_size)
                assert cut_end[-1] < max_len
                cut_start = [0] * (self.backward_size - 2) + cut_start  # since we give up on the seq training idea
                cut_end = range(2, self.backward_size) + cut_end
            else:
                cut_start = [0] * (max_len - 2)
                cut_end = range(2, max_len)

            new_grids = [(idx, s_id, e_id) for s_id, e_id in zip(cut_start, cut_end) if s_id < min_len - 1]
            if intra_shuffle and shuffle:
                np.random.shuffle(new_grids)
            self.grid_indexes.extend(new_grids)

        self.num_batch = len(self.grid_indexes)
        print("%s begins with %d batches." % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_grid = self.grid_indexes[self.ptr]
            if self.ptr > 0:
                prev_grid = self.grid_indexes[self.ptr - 1]
            else:
                prev_grid = None
            self.ptr += 1
            return self._prepare_batch(cur_grid=current_grid, prev_grid=prev_grid)
        else:
            return None


class ConvAI2DataLoader(DataLoaderBase):
    def __init__(self, name, data, persona_data, persona_word_data, config, vocab_size, vocab_idf):

        self.name = name
        self.data = data
        self.persona_data = persona_data
        self.persona_word_data = persona_word_data
        self.vocab_idf = vocab_idf

        self.vocab_size = vocab_size
        self.data_size = len(data)
        self.data_lens = all_lens = [len(line) for line in self.data]
        all_persona_lens = []
        for i in persona_data:
            for j in i:
                all_persona_lens.append(len(j))
        self.persona_lens = all_persona_lens
        self.max_utt_size = config.max_utt_len
        self.max_per_size = config.max_per_len
        self.max_per_line = config.max_per_line
        self.max_per_words = config.max_per_words
        print("Max utterance len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                                      np.min(all_lens),
                                                                      float(np.mean(all_lens))))
        print("Max persona len %d and min len %d and avg len %f" % (np.max(all_persona_lens),
                                                                    np.min(all_persona_lens),
                                                                    float(np.mean(all_persona_lens))))
        self.indexes = list(np.argsort(all_lens))

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_size:
            return tokens[0:self.max_utt_size - 1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_size - len(tokens))
        else:
            return tokens

    def persona_pad_to(self, personas, do_pad=True):
        padded_personas = []
        per = ""
        for tokens in personas:
            if len(tokens) >= self.max_per_size:
                per = tokens[0:self.max_per_size - 1] + [tokens[-1]]
            elif do_pad:
                per = tokens + [0] * (self.max_per_size - len(tokens))
            else:
                per = tokens
            padded_personas.append(per)
        if len(padded_personas) < self.max_per_line:
            for i in range(self.max_per_line - len(padded_personas)):
                padded_personas.append([0] * self.max_per_size)
        return padded_personas

    def persona_word_pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_per_words:
            return tokens[0:self.max_per_words]
        elif do_pad:
            return tokens + [0] * (self.max_per_words - len(tokens))
        else:
            return tokens

    def _prepare_batch(self, cur_grid, prev_grid):
        b_id, s_id, e_id = cur_grid

        batch_ids = self.batch_indexes[b_id]
        rows = [self.data[idx] for idx in batch_ids]
        meta_rows = [self.meta_data[idx] for idx in batch_ids]
        persona_rows = [self.persona_data[idx] for idx in batch_ids]
        persona_word_rows = [self.persona_word_data[idx] for idx in batch_ids]
        dialog_lens = [self.data_lens[idx] for idx in batch_ids]

        topics = np.array([meta[2] for meta in meta_rows])
        cur_pos = [np.minimum(1.0, e_id / float(l)) for l in dialog_lens]

        context_lens, context_utts, floors, out_utts, out_lens, out_floors, out_das = [], [], [], [], [], [], []
        for row in rows:
            if s_id < len(row) - 1:
                cut_row = row[s_id:e_id]
                in_row = cut_row[0:-1]
                out_row = cut_row[-1]
                out_utt, out_floor, out_feat = out_row

                context_utts.append([self.pad_to(utt) for utt, floor, feat in in_row])
                floors.append([int(floor == out_floor) for utt, floor, feat in in_row])
                context_lens.append(len(cut_row) - 1)

                out_utt = self.pad_to(out_utt, do_pad=False)
                out_utts.append(out_utt)
                out_lens.append(len(out_utt))
                out_floors.append(out_floor)
                out_das.append(out_feat[0])
            else:
                print(row)
                raise ValueError("S_ID %d larger than row" % s_id)

        personas = []
        persona_words = []
        for row in persona_rows:
            personas.append(self.persona_pad_to(row))
            wrd = []
            for i in row:
                wrd.append(sorted(list(set(i + [3]))))

            persona_words.append(self.persona_pad_to(wrd))

        my_profiles = np.array([meta[out_floors[idx]] for idx, meta in enumerate(meta_rows)])
        ot_profiles = np.array([meta[1 - out_floors[idx]] for idx, meta in enumerate(meta_rows)])
        vec_context_lens = np.array(context_lens)
        vec_context = np.zeros((self.batch_size, np.max(vec_context_lens), self.max_utt_size), dtype=np.int32)
        vec_floors = np.zeros((self.batch_size, np.max(vec_context_lens)), dtype=np.int32)
        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int32)
        vec_out_lens = np.array(out_lens)
        vec_out_das = np.array(out_das)
        vec_persona = np.array(personas)
        vec_persona_words = np.array(persona_words)

        for b_id in range(self.batch_size):
            vec_outs[b_id, 0:vec_out_lens[b_id]] = out_utts[b_id]
            vec_floors[b_id, 0:vec_context_lens[b_id]] = floors[b_id]
            vec_context[b_id, 0:vec_context_lens[b_id], :] = np.array(context_utts[b_id])

        # I. no labeled data, all 0 initialized.
        vec_persona_position = np.zeros((self.batch_size, vec_outs.shape[1]), dtype=np.int32)

        # II. no labeled data, randomly initialized. The threshold used here is 0.16 with convai2_voacb_idf.txt
        vec_selected_persona = np.zeros((self.batch_size, 1), dtype=np.int32)
        for i in range(self.batch_size):
            vec_selected_persona[i] = [np.random.randint(self.max_per_line)]

        return vec_context, vec_context_lens, vec_floors, topics, my_profiles, ot_profiles, vec_outs, vec_out_lens, vec_out_das, vec_persona, vec_persona_words, vec_persona_position, vec_selected_persona


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 2)  # 2 for bidirection

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def checkpoint_sequential(functions, segments, *inputs):
    def run_function(start, end, functions):
        def forward(*inputs):
            for j in range(start, end + 1):
                inputs = functions[j](*inputs)
            return inputs
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return run_function(end + 1, len(functions) - 1, functions)(*inputs)


class MultiheadAttention(nn.Module):
    @classmethod
    def _get_future_mask(cls, size, device):
        if not hasattr(cls, '_future_mask') or cls._future_mask.device != device or cls._future_mask.shape < size:
            cls._future_mask = torch.triu(torch.ones(size[0], size[1], dtype=torch.uint8, device=device), 1)

        mask = cls._future_mask[:size[0], :size[1]]

        return mask

    def __init__(self, n_features, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        assert n_features % n_heads == 0

        self.n_features = n_features
        self.n_heads = n_heads
        self.qkv_proj = nn.Linear(n_features, 3 * n_features)
        self.out_proj = nn.Linear(n_features, n_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.qkv_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)

    def _split_heads(self, x, is_key=False):
        x = x.view(x.shape[0], x.shape[1], self.n_heads, self.n_features // self.n_heads)
        x = x.permute(0, 2, 3, 1) if is_key else x.permute(0, 2, 1, 3)

        return x

    def _attn(self, q, k, v, apply_future_mask=True, padding_mask=None):
        w = torch.matmul(q, k) / math.sqrt(self.n_features // self.n_heads)

        if apply_future_mask:
            future_mask = MultiheadAttention._get_future_mask(w.shape[-2:], w.device).unsqueeze(0).unsqueeze(0)
            w.masked_fill_(future_mask, float('-inf'))

        if padding_mask is not None:
            w.masked_fill_(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        w = F.softmax(w, dim=-1)
        w = self.dropout(w)

        if padding_mask is not None:
            w.masked_fill_(padding_mask.all(dim=-1).unsqueeze(1).unsqueeze(2).unsqueeze(3), 0)

        out = torch.matmul(w, v)

        return out

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.n_features)

        return x

    def forward(self, query, key, value, padding_mask):
        qkv_same = (query.data_ptr() == key.data_ptr() == value.data_ptr())
        kv_same = (key.data_ptr() == value.data_ptr())

        if qkv_same:
            query, key, value = self.qkv_proj(query).split(self.n_features, dim=-1)
            apply_future_mask = True  # self-attention
        elif kv_same:
            q_w, q_b = self.qkv_proj.weight[:self.n_features, :], self.qkv_proj.bias[:self.n_features]
            query = F.linear(query, q_w, q_b)
            kv_w, kv_b = self.qkv_proj.weight[self.n_features:, :], self.qkv_proj.bias[self.n_features:]
            key, value = F.linear(key, kv_w, kv_b).split(self.n_features, dim=-1)
            apply_future_mask = False
        else:
            assert False

        query = self._split_heads(query)
        key = self._split_heads(key, is_key=True)
        value = self._split_heads(value)

        x = self._attn(query, key, value, apply_future_mask, padding_mask)
        x = self._merge_heads(x)

        x = self.out_proj(x)

        return x


class FeedForward(nn.Module):
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def __init__(self, in_features, middle_features, dropout):
        super(FeedForward, self).__init__()

        self.layer_1 = nn.Linear(in_features, middle_features)
        self.layer_2 = nn.Linear(middle_features, in_features)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer_1.weight, std=0.02)
        nn.init.normal_(self.layer_2.weight, std=0.02)

    def forward(self, x):
        x = FeedForward.gelu(self.layer_1(x))
        x = self.dropout(x)
        x = self.layer_2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, dropout, attn_dropout, ff_dropout):
        super(TransformerBlock, self).__init__()

        self.attn = MultiheadAttention(n_features, n_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(n_features)
        self.ff = FeedForward(n_features, 4 * n_features, ff_dropout)
        self.ff_norm = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask, *contexts):
        '''contexts = [(context1, padding_mask1), ...]'''

        inputs = (x, padding_mask) + contexts

        full_attn = 0
        n_attn = len(inputs) // 2
        for i in range(0, len(inputs), 2):
            c, m = inputs[i], inputs[i + 1].byte()
            a = self.attn(x, c, c, m)
            full_attn += (a / n_attn)

        full_attn = self.dropout(full_attn)
        x = self.attn_norm(x + full_attn)

        f = self.ff(x)
        f = self.dropout(f)
        x = self.ff_norm(x + f)

        return (x, padding_mask) + contexts


class TransformerModule(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size,
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 n_segments=None):
        super(TransformerModule, self).__init__()

        self.embeddings = nn.Embedding(n_embeddings, embeddings_size, padding_idx=padding_idx)
        self.pos_embeddings = nn.Embedding(n_pos_embeddings + 1, embeddings_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(embeddings_size, n_heads, dropout, attn_dropout, ff_dropout) for _ in range(n_layers)])
        self.n_segments = n_segments

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, std=0.02)

    def forward(self, x, enc_contexts=[]):
        padding_mask = x.eq(self.embeddings.padding_idx)

        positions = torch.cumsum(~padding_mask, dim=-1, dtype=torch.long)
        positions.masked_fill_(padding_mask, self.pos_embeddings.padding_idx)

        x = self.embeddings(x) * math.sqrt(self.embeddings.embedding_dim) + self.pos_embeddings(positions)
        x = self.embed_dropout(x)

        enc_contexts = sum(enc_contexts, ())

        if self.n_segments is not None:
            padding_mask = padding_mask.float()  # fucking checkpoint_sequential
            padding_mask.requires_grad_()  # fucking checkpoint_sequential
            out = checkpoint_sequential(self.layers, self.n_segments, x, padding_mask, *enc_contexts)
            x = out[0]
        else:
            for layer in self.layers:
                out = layer(x, padding_mask, *enc_contexts)
                x = out[0]

        return x, padding_mask


class TransformerModel(nn.Module):
    def __init__(self, n_layers, n_embeddings, n_pos_embeddings, embeddings_size,
                 padding_idx, n_heads, dropout, embed_dropout, attn_dropout, ff_dropout,
                 bos_id, eos_id, max_seq_len=256, beam_size=5, sample=False,
                 length_penalty=0.8, annealing_topk=None, annealing=0,
                 diversity_coef=0, diversity_groups=1, n_segments=None):

        super(TransformerModel, self).__init__()

        self.padding_idx = padding_idx
        self.n_embeddings = n_embeddings
        self.n_pos_embeddings = n_pos_embeddings
        self.embeddings_size = embeddings_size

        self.bos_id = bos_id
        self.eos_id = eos_id

        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.sample = sample
        self.length_penalty_coef = length_penalty
        self.annealing = annealing
        self.annealing_topk = annealing_topk
        self.diversity_coef = diversity_coef
        self.diversity_groups = diversity_groups

        self.transformer_module = TransformerModule(n_layers, n_embeddings, n_pos_embeddings, embeddings_size,
                                                    padding_idx, n_heads, dropout, embed_dropout, attn_dropout,
                                                    ff_dropout, n_segments)
        self.pre_softmax = nn.Linear(embeddings_size, n_embeddings, bias=False)
        self.pre_softmax.weight = self.transformer_module.embeddings.weight

    def forward(self, x, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        return self.decode(x, enc_contexts)

    def encode(self, x):
        return self.transformer_module(x)

    def generate(self, enc_x):
        return self.pre_softmax(enc_x)

    def decode(self, x, enc_contexts=[]):
        x, _ = self.transformer_module(x, enc_contexts)
        return self.generate(x)

    def predict(self, contexts=[]):
        enc_contexts = [self.encode(c) for c in contexts]
        prediction = self.beam_search(enc_contexts)

        return prediction

    def _length_penalty(self, sequence_lengths):
        """https://arxiv.org/abs/1609.08144"""
        return (5 + sequence_lengths) ** self.length_penalty_coef / (5 + 1) ** self.length_penalty_coef

    def beam_search(self, enc_contexts=[], return_beams=False):
        with torch.no_grad():
            if len(enc_contexts) == 0:
                return []

            batch_size = enc_contexts[0][0].shape[0]
            device = next(self.parameters()).device

            prevs = torch.full((batch_size * self.beam_size, 1), fill_value=self.bos_id, dtype=torch.long,
                               device=device)

            beam_scores = torch.zeros(batch_size, self.beam_size, device=device)
            beam_lens = torch.ones(batch_size, self.beam_size, dtype=torch.long, device=device)
            is_end = torch.zeros(batch_size, self.beam_size, dtype=torch.uint8, device=device)

            beam_enc_contexts = []
            for c, p in enc_contexts:
                c = c.unsqueeze(1).repeat(1, self.beam_size, 1, 1)
                c = c.view(-1, c.shape[2], c.shape[3])
                p = p.unsqueeze(1).repeat(1, self.beam_size, 1)
                p = p.view(-1, p.shape[2])
                beam_enc_contexts.append((c, p))

            current_sample_prob = 1
            group_size = self.beam_size // self.diversity_groups
            diversity_penalty = torch.zeros((batch_size, self.n_embeddings), device=device)

            for i in range(self.max_seq_len):
                outputs, _ = self.transformer_module(prevs, beam_enc_contexts)

                logits = self.generate(outputs[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)
                log_probs = log_probs.view(batch_size, self.beam_size, -1)

                beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1))
                penalty = self._length_penalty(beam_lens.float() + 1 - is_end.float())
                penalty = penalty.unsqueeze(-1).repeat(1, 1, self.n_embeddings)
                beam_scores = beam_scores / penalty

                if i == 0:
                    penalty = penalty[:, 0, :]
                    beam_scores = beam_scores[:, 0, :]

                    beam_scores, idxs = beam_scores.topk(self.beam_size, dim=-1)
                    beam_idxs = torch.zeros((batch_size, self.beam_size), dtype=torch.long, device=device)
                else:
                    penalty = penalty.view(batch_size, self.diversity_groups, group_size, -1)
                    beam_scores = beam_scores.view(batch_size, self.diversity_groups, group_size, -1)

                    all_scores, all_idxs = [], []
                    for g in range(self.diversity_groups):
                        g_beam_scores = beam_scores[:, g, :, :]
                        g_penalty = penalty[:, g, :, :]
                        g_beam_scores -= self.diversity_coef * diversity_penalty.unsqueeze(1) / g_penalty
                        g_beam_scores = g_beam_scores.view(batch_size, -1)

                        if random.random() < current_sample_prob:
                            beam_probas = F.softmax(g_beam_scores, dim=-1)
                            if self.annealing_topk is not None:
                                beam_probas, sample_idxs = beam_probas.topk(self.annealing_topk, dim=-1)
                                g_idxs = torch.multinomial(beam_probas, group_size)
                                g_idxs = torch.gather(sample_idxs, 1, g_idxs)
                            else:
                                g_idxs = torch.multinomial(beam_probas, group_size)
                        else:
                            _, g_idxs = g_beam_scores.topk(group_size, dim=-1)

                        g_scores = torch.gather(beam_scores[:, g, :, :].view(batch_size, -1), 1, g_idxs)
                        g_idxs += g * group_size * self.n_embeddings

                        all_scores.append(g_scores)
                        all_idxs.append(g_idxs)

                        diversity_penalty.scatter_add_(1, torch.fmod(g_idxs, self.n_embeddings),
                                                       torch.ones((batch_size, group_size), device=device))

                    diversity_penalty.fill_(0)
                    penalty = penalty.view(batch_size, -1)
                    beam_scores = torch.cat(all_scores, dim=-1)
                    idxs = torch.cat(all_idxs, dim=-1)

                    beam_idxs = (idxs.float() / self.n_embeddings).long()

                penalty = torch.gather(penalty, 1, idxs)
                sym_idxs = torch.fmod(idxs, log_probs.shape[-1])
                is_end = torch.gather(is_end, 1, beam_idxs)
                beam_lens = torch.gather(beam_lens, 1, beam_idxs)

                sym_idxs[is_end] = self.padding_idx
                beam_lens[~is_end] += 1
                is_end[sym_idxs == self.eos_id] = 1

                sym_idxs = sym_idxs.view(batch_size * self.beam_size, 1)
                prevs = prevs.view(batch_size, self.beam_size, -1)
                prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
                prevs = prevs.view(batch_size * self.beam_size, -1)
                prevs = torch.cat([prevs, sym_idxs], dim=1)

                if all(is_end.view(-1)):
                    break

                beam_scores *= penalty
                current_sample_prob *= self.annealing

            predicts = []
            result = prevs.view(batch_size, self.beam_size, -1)

            if return_beams:
                return result, beam_lens

            if self.sample:
                probs = F.softmax(beam_scores, dim=-1)
                bests = torch.multinomial(probs, 1).view(-1)
            else:
                bests = beam_scores.argmax(dim=-1)

            for i in range(batch_size):
                best_len = beam_lens[i, bests[i]]
                best_seq = result[i, bests[i], 1:best_len - 1]
                predicts.append(best_seq.tolist())

        return predicts


def openai_transformer_config():
    class dotdict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    cfg = dotdict({'n_layers': 12, 'n_embeddings': 40477, 'n_pos_embeddings': 512,
                   'embeddings_size': 768, 'n_heads': 12, 'dropout': 0.1,
                   'embed_dropout': 0.1, 'attn_dropout': 0.1, 'ff_dropout': 0.1})

    return cfg


def load_openai_weights(model, directory, n_special_tokens=0):
    # TODO: add check of shapes

    parameters_names_path = os.path.join(directory, 'parameters_names.json')
    parameters_shapes_path = os.path.join(directory, 'parameters_shapes.json')
    parameters_weights_paths = [os.path.join(directory, 'params_{}.npy'.format(n)) for n in range(10)]

    with open(parameters_names_path, 'r') as parameters_names_file:
        parameters_names = json.load(parameters_names_file)

    with open(parameters_shapes_path, 'r') as parameters_shapes_file:
        parameters_shapes = json.load(parameters_shapes_file)

    parameters_weights = [np.load(path) for path in parameters_weights_paths]
    parameters_offsets = np.cumsum([np.prod(shape) for shape in parameters_shapes])
    parameters_weights = np.split(np.concatenate(parameters_weights, 0), parameters_offsets)[:-1]
    parameters_weights = [p.reshape(s) for p, s in zip(parameters_weights, parameters_shapes)]

    parameters_weights[1] = parameters_weights[1][1:] # skip 0 - <unk>


    if model.pos_embeddings.num_embeddings - 1 > parameters_weights[0].shape[0]:
        xx = np.linspace(0, parameters_weights[0].shape[0], model.pos_embeddings.num_embeddings - 1)
        new_kernel = RectBivariateSpline(np.arange(parameters_weights[0].shape[0]),
                                         np.arange(parameters_weights[0].shape[1]),
                                         parameters_weights[0])
        parameters_weights[0] = new_kernel(xx, np.arange(parameters_weights[0].shape[1]))

    parameters_weights[0] = parameters_weights[0][:model.pos_embeddings.num_embeddings - 1]
    parameters_weights[1] = parameters_weights[1][:model.embeddings.num_embeddings - n_special_tokens]

    model.pos_embeddings.weight.data[1:] = torch.from_numpy(parameters_weights[0])
    model.embeddings.weight.data[n_special_tokens:] = torch.from_numpy(parameters_weights[1])


    parameters_weights = parameters_weights[2:]

    for name, weights in zip(parameters_names, parameters_weights):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ':0'
        name = name[:-2]
        name = name.split('/')

        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]

            pointer = getattr(pointer, l[0])

            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]

        if len(weights.shape) == 3: # conv1d to linear
            weights = weights[0].transpose((1, 0))

        pointer.data[...] = torch.from_numpy(weights)


def pad_sequence(sequences, batch_first=False, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def get_bow(embedding, avg=False):
    embedding_size = embedding.get_shape()[2].value
    if avg:
        return torch.mean(embedding, dim=1), embedding_size
    else:
        return torch.sum(embedding, dim=1), embedding_size


def get_rnn_encode(cell, sent_embedding, batch_size, hidden_size):
    encoded_embedding = []
    sent_size = list(sent_embedding.size())[0]
    h0 = torch.randn(batch_size, hidden_size)
    for wd in sent_embedding:
        tmp = cell(wd,h0)
        encoded_embedding.append(cell(wd,h0))
        h0 = tmp
    return torch.tensor(encoded_embedding), sent_size


def get_bi_rnn_encode(embedding, hidden_size, num_layers):
    birnn = BiRNN(embedding.size(), hidden_size, num_layers)
    return birnn(embedding)
    pass


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def f1_score(predictions, targets, average=True):
    def f1_score_items(pred_items, gold_items):
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())

        if num_same == 0:
            return 0

        precision = num_same / len(pred_items)
        recall = num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    scores = [f1_score_items(p, t) for p, t in zip(predictions, targets)]

    if average:
        return sum(scores) / len(scores)

    return scores
