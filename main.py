from config import get_PMSN_config, get_Transferrer_config, get_trainer_config
from model.utils import ConvAI2DialogCorpus, ConvAI2DataLoader, set_seed, TransformerModel, load_openai_weights
from model.utils import f1_score
from models import PMSN
import time
from model.text import BPEVocab
from model.dataset import FacebookDataset
from model.trainer import *


def train_PMSN(model, config):
    PMSN_model = PMSN(config, corpus)
    r = PMSN_model(train_feed)


def train_diamodel():
    model_config = get_Transferrer_config()
    trainer_config = get_trainer_config()
    device = 'cpu'
    vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)
    set_seed(trainer_config.seed)
    transformer = TransformerModel(n_layers=model_config.n_layers,
                                   n_embeddings=len(vocab),
                                   n_pos_embeddings=model_config.n_pos_embeddings,
                                   embeddings_size=model_config.embeddings_size,
                                   padding_idx=vocab.pad_id,
                                   n_heads=model_config.n_heads,
                                   dropout=model_config.dropout,
                                   embed_dropout=model_config.embed_dropout,
                                   attn_dropout=model_config.attn_dropout,
                                   ff_dropout=model_config.ff_dropout,
                                   bos_id=vocab.bos_id,
                                   eos_id=vocab.eos_id,
                                   max_seq_len=model_config.max_seq_len,
                                   beam_size=model_config.beam_size,
                                   length_penalty=model_config.length_penalty,
                                   n_segments=model_config.n_segments,
                                   annealing_topk=model_config.annealing_topk,
                                   annealing=model_config.annealing,
                                   diversity_coef=model_config.diversity_coef,
                                   diversity_groups=model_config.diversity_groups)

    if not trainer_config.load_last:
        load_openai_weights(transformer.transformer_module,
                            trainer_config.openai_parameters_dir,
                            n_special_tokens=vocab.n_special_tokens)
        print('OpenAI weights loaded from {}'.format(trainer_config.openai_parameters_dir))

    train_dataset = FacebookDataset(trainer_config.train_datasets, vocab, transformer.n_pos_embeddings - 1)
    test_dataset = FacebookDataset(trainer_config.test_datasets, vocab, transformer.n_pos_embeddings - 1)

    model_trainer = Trainer(transformer,
                            train_dataset,
                            test_dataset,
                            batch_size=trainer_config.batch_size,
                            batch_split=trainer_config.batch_split,
                            lr=trainer_config.lr,
                            lr_warmup=trainer_config.lr_warmup,
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight,
                            n_jobs=trainer_config.n_jobs,
                            clip_grad=trainer_config.clip_grad,
                            device=device,
                            ignore_idxs=vocab.special_tokens_ids)

    if trainer_config.load_last:
        state_dict = torch.load(trainer_config.last_checkpoint_path, map_location=device)
        model_trainer.load_state_dict(state_dict)
        print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path))

    def save_func(epoch):
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona_info, dialog]
                        if len(c) > 0]
            prediction = model_trainer.model.predict(contexts)[0]

            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            print('\n')
            print('Persona info:\n\t{}'.format(persona_info_str))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def test_func(epoch):
        if (epoch + 1) % trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs)

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        return [1 - s for s in scores]

        # helpers -----------------------------------------------------

    try:
        model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, sample_text_func, test_func],
                            risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e





    pass


if __name__ == '__main__':
    # config the PMSN
    PMSN_config = get_PMSN_config()
    corpus = ConvAI2DialogCorpus(PMSN_config.data, max_vocab_cnt = PMSN_config.vocab_size, word2vec=config.w2v_path,
                                 w2v_dim=PMSN_config.embed_size, vocab_files=PMSN_config.vocab_file, idf_files=config.idf_file)

    dia_corpus = corpus.get_dialog_corpus()
    persona_corpus = corpus.get_persona_corpus()
    persona_word_corpus = corpus.get_persona_word_corpus()
    vocab_size = corpus.gen_vocab_size
    vocab_idf = corpus.index2idf

    train_dial, valid_dial, test_dial = dia_corpus.get("train"), dia_corpus.get("valid"), dia_corpus.get("test")
    train_persona, valid_persona, test_persona = persona_corpus.get("train"), persona_corpus.get("valid"), persona_corpus.get("test")
    train_persona_word, valid_persona_word, test_persona_word = persona_word_corpus.get("train"), persona_word_corpus.get("valid"), persona_word_corpus.get("test")


    train_feed = ConvAI2DataLoader("Train", train_dial, train_persona, train_persona_word, PMSN_config,
                                   vocab_size, vocab_idf)
    valid_feed = ConvAI2DataLoader("Valid", valid_dial, valid_persona, valid_persona_word, PMSN_config,
                                   vocab_size, vocab_idf)
    test_feed = ConvAI2DataLoader("Test", test_dial, test_persona, test_persona_word, PMSN_config, vocab_size,
                                  vocab_idf)
