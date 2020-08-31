# Setting
import os
import sys
sys.path.append(os.getcwd())

from train_config import *


def main():
    # 준비 작업
    args, log = setup()
    log.info('[Program starts. Loading data...]')

    # 데이터셋 불러오기
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    
    """
    opt: {'log_per_updates': 3, 'data_file': 'SQuAD/data.msgpack',
    'model_dir': 'C:\\Users\\YY\\Documents\\Pytorch-DrQA\\models', 'save_last_only': False,
    'save_dawn_logs': False, 'seed': 1013, 'cuda': True, 'epochs': 40, 'batch_size': 32, 'resume': 'best_model.pt',
    'resume_options': False, 'reduce_lr': 0.0, 'optimizer': 'adamax', 'grad_clipping': 10, 'weight_decay': 0,
    'learning_rate': 0.1, 'momentum': 0, 'tune_partial': 1000, 'fix_embeddings': False, 'rnn_padding': False,
    'question_merge': 'self_attn', 'doc_layers': 3, 'question_layers': 3, 'hidden_size': 128, 'num_features': 4,
    'pos': True, 'ner': True, 'use_qemb': True, 'concat_rnn_layers': True, 'dropout_emb': 0.4, 'dropout_rnn': 0.4,
    'dropout_rnn_output': True, 'max_len': 15, 'rnn_type': 'lstm', 'pretrained_words': True, 'vocab_size': 91187,
    'embedding_dim': 300, 'pos_size': 11, 'ner_size': 1}
    
    opt type:  <class 'dict'>
    """
    
    # 이 아래부터는 점수 매기고 학습시키는 코드임
    log.info(opt)
    log.info('[Data loaded.]')
    if args.save_dawn_logs:
        dawn_start = datetime.now()
        log.info('dawn_entry: epoch\tf1Score\thours')

    if args.resume:
        log.info('[loading previous model...]')
        if torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(args.model_dir, args.resume))
        else:
            checkpoint = torch.load(os.path.join(args.model_dir, args.resume), map_location='cpu')
        if args.resume_options:
            opt = checkpoint['config']
        state_dict = checkpoint['state_dict']
        model = DocReaderModel(opt, embedding, state_dict)
        epoch_0 = checkpoint['epoch'] + 1

        # synchronize random seed
        random.setstate(checkpoint['random_state'])

        torch.random.set_rng_state(checkpoint['torch_state'])
        if args.cuda:
            torch.cuda.set_rng_state(checkpoint['torch_cuda_state'])
        if args.reduce_lr:
            lr_decay(model.optimizer, lr_decay=args.reduce_lr)
            log.info('[learning rate reduced by {}]'.format(args.reduce_lr))

        batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)

        #-----
        predictions = []
        for i, batch in enumerate(batches):
            predictions.extend(model.predict(batch))
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))
            
        # 여기서 score 함수 -> _exact_match 함수 -> _normalize_answer 함수 적용함
        # predictions = 모델을 통과한 예측 값
        # dev_y = 실제 답
        # score 함수로 내려가보자
        # em += _exact_match(p, t) 여기서 p, t는 pred, truth의 일부
        # em은 exact match score를 계산한다.
        em, f1 = score(predictions, dev_y)
        log.info("[dev EM: {} F1: {}]".format(em, f1))
        if math.fabs(em - checkpoint['em']) > 1e-3 or math.fabs(f1 - checkpoint['f1']) > 1e-3:
            log.info('Inconsistent: recorded EM: {} F1: {}'.format(checkpoint['em'], checkpoint['f1']))
            log.error('Error loading model: current code is inconsistent with code used to train the previous model.')
            exit(1)
        best_val_score = checkpoint['best_eval']
    else:
        model = DocReaderModel(opt, embedding)
        epoch_0 = 1
        best_val_score = 0.0

    for epoch in range(epoch_0, epoch_0 + args.epochs):
        log.warning('Epoch {}'.format(epoch))
        # train
        batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
        start = datetime.now()
        for i, batch in enumerate(batches):
            model.update(batch)
            if i % args.log_per_updates == 0:
                log.info('> epoch [{0:2}] updates[{1:6}] train loss[{2:.5f}] remaining[{3}]'.format(
                    epoch, model.updates, model.train_loss.value,
                    str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))
        log.debug('\n')
        # eval
        batches = BatchGen(dev, batch_size=args.batch_size, evaluation=True, gpu=args.cuda)

        # batches.batch_size, eval, data[0], gpu, ner_size, pos_size

        predictions = []

        # predictions[?] 예시:
        # ['trans-Atlantic wireless telecommunications facility',
        # 'trans-Atlantic wireless telecommunications', ... ]

        for i, batch in enumerate(batches):
            predictions.extend(model.predict(batch))
            log.debug('> evaluating [{}/{}]'.format(i, len(batches)))

        em, f1 = score(predictions, dev_y)
        log.warning("dev EM: {} F1: {}".format(em, f1))

        if args.save_dawn_logs:
            time_diff = datetime.now() - dawn_start
            log.warning("dawn_entry: {}\t{}\t{}".format(epoch, f1/100.0, float(time_diff.total_seconds() / 3600.0)))
        # save
        if not args.save_last_only or epoch == epoch_0 + args.epochs - 1:
            model_file = os.path.join(args.model_dir, 'checkpoint_epoch_{}.pt'.format(epoch))
            model.save(model_file, epoch, [em, f1, best_val_score])
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(
                    model_file,
                    os.path.join(args.model_dir, 'best_model.pt'))
                log.info('[new best model saved.]')


def setup():
    parser = argparse.ArgumentParser(description='Train a Document Reader model.')

    # system
    parser.add_argument('--log_per_updates', type=int, default=3,
                        help='log model loss per x updates (mini-batches).')
    parser.add_argument('--data_file', default='SQuAD/data.msgpack',
                        help='path to preprocessed data file.')
    parser.add_argument('--model_dir', default='models',
                        help='path to store saved models.')
    parser.add_argument('--save_last_only', action='store_true',
                        help='only save the final models.')
    parser.add_argument('--save_dawn_logs', action='store_true',
                        help='append dawnbench log entries prefixed with dawn_entry:')
    parser.add_argument('--seed', type=int, default=1013,
                        help='random seed for data shuffling, dropout, etc.')
    parser.add_argument("--cuda", type=str2bool, nargs='?',
                        const=True, default=torch.cuda.is_available(),
                        help='whether to use GPU acceleration.')
    # training
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-rs', '--resume', default='best_model.pt',
                        help='previous model file name (in `model_dir`). '
                             'e.g. "checkpoint_epoch_11.pt"')
    parser.add_argument('-ro', '--resume_options', action='store_true',
                        help='use previous model options, ignore the cli and defaults.')
    parser.add_argument('-rlr', '--reduce_lr', type=float, default=0.,
                        help='reduce initial (resumed) learning rate by this factor.')
    parser.add_argument('-op', '--optimizer', default='adamax',
                        help='supported optimizer: adamax, sgd')
    parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1,
                        help='only applied to SGD.')
    parser.add_argument('-mm', '--momentum', type=float, default=0,
                        help='only applied to SGD.')
    parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
                        help='finetune top-x embeddings.')
    parser.add_argument('--fix_embeddings', action='store_true',
                        help='if true, `tune_partial` will be ignored.')
    parser.add_argument('--rnn_padding', action='store_true',
                        help='perform rnn padding (much slower but more accurate).')
    # model
    parser.add_argument('--question_merge', default='self_attn')
    parser.add_argument('--doc_layers', type=int, default=3)
    parser.add_argument('--question_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_features', type=int, default=4)
    parser.add_argument('--pos', type=str2bool, nargs='?', const=True, default=True,
                        help='use pos tags as a feature.')
    parser.add_argument('--ner', type=str2bool, nargs='?', const=True, default=False,
                        help='use named entity tags as a feature.')
    parser.add_argument('--use_qemb', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--concat_rnn_layers', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--dropout_emb', type=float, default=0.4)
    parser.add_argument('--dropout_rnn', type=float, default=0.4)
    parser.add_argument('--dropout_rnn_output', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--rnn_type', default='lstm',
                        help='supported types: rnn, gru, lstm')

    args = parser.parse_args()

    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    if args.resume == 'best_model.pt' and not os.path.exists(os.path.join(args.model_dir, args.resume)):
        # means we're starting fresh
        args.resume = ''

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # 이 아래는 별로 신경쓸 거 없다.
    # setup logger
    class ProgressHandler(logging.Handler):
        def __init__(self, level=logging.NOTSET):
            super().__init__(level)

        def emit(self, record):
            log_entry = self.format(record)
            if record.message.startswith('> '):
                sys.stdout.write('{}\r'.format(log_entry.rstrip()))
                sys.stdout.flush()
            else:
                sys.stdout.write('{}\n'.format(log_entry))

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
    fh.setLevel(logging.INFO)
    ch = ProgressHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    log.addHandler(fh)
    log.addHandler(ch)

    return args, log


def lr_decay(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


def load_data(opt):
    # 학습에 필요한 데이터를 가져온다.
    # msgpack: 메시지팩
    # 데이터 통신을 원활하게 할 수 있게, 데이터 포맷을 투명하게 직렬화한다.
    # 직렬화: 개별 플랫폼에서 사용하는 데이터를 바이트 배열로 변경하는 작업
    # 역직렬화: 직렬화된 바이트 배열을 객체로 변경하는 작업
    # 전처리(preprocessing) 작업을 통해 나온 msgpack 파일을 읽어온다.
    # meta: 딕셔너리
    # meta.keys() -- ['vocab', 'vocab_tag', 'vocab_ent', 'embedding', 'wv_cased']
    # meta['vocab'] = ['<PAD>', '<UNK>', 'what', ... ] -- 길이는 91187
    # meta['vocab_tag'] = ['', 'RB', 'MD', 'PRP', 'SP', ':', 'CC', 'VBD', 'VBP', 'VB', 'VBZ']
    # meta['vocab_ent'] = [''] -- 지금 여기는 spacy 'en'이 잘 안되어서 그러함
    # meta['embedding'] = 길이는 91187
    # meta['wv_cased'] = True
    with open('SQuAD/meta.msgpack', 'rb') as f:
        # meta = msgpack.load(f, encoding='utf8')
        meta = msgpack.load(f)


    # 텐서로 만들기 shape = (91187, 300)
    embedding = torch.Tensor(meta['embedding'])

    # opt = vars(args) -- 딕셔너리
    # 딕셔너리의 key에 맞는 value를 지정한다.
    opt['pretrained_words'] = True
    opt['vocab_size'] = embedding.size(0)        # 91187
    opt['embedding_dim'] = embedding.size(1)     # 300
    opt['pos_size'] = len(meta['vocab_tag'])     # 11
    opt['ner_size'] = len(meta['vocab_ent'])     # 1
    
    # 아래 BatchGen 클래스의 속성을 정의한다.
    BatchGen.pos_size = opt['pos_size']          # 11
    BatchGen.ner_size = opt['ner_size']          # 1

    # opt['data_file'] = 'SQuAD/data.msgpack'
    # msgpack을 읽어 온다.
    with open(opt['data_file'], 'rb') as f:
        # data = msgpack.load(f, encoding='utf8')
        data = msgpack.load(f)


    # data = 앞서 읽어온 msgpack, type = 딕셔너리
    # len(data['train'] = 86536 (영어) / 85195 (한글)
    # len(data['dev'] = 10570 (영어)   / 10065 (한글)
    # data['dev']
    
    """
    data['dev'][10569][0:8]
    '5737aafd1c456719005744ff'
    [36, 11533, 25, 414, 47, 12, 6177, 8669, 18, 452, 508, 46, 97, 3, 27037, 1662, 3, 7541, 25, 414, 920, 52693, 825, 920, 486, 1, ... ]
    [[False, True, True, 0.06369426751592357], [False, False, True, 0.006369426751592357], [False, False, True, 0.025477707006369428] ...
    [0, 0, 0, ...]
    [0, 0, 0, ...]
    [4, 9, 3, 10987, 46, 414, 1363, 2048, 7, 58, 10065, 41578, 2]
    'The pound-force has a metric counterpart, less commonly used than the newton: the kilogram-force (kgf) (sometimes kilopond) ... '
    [[0, 3], [4, 9], [9, 10], [10, 15], [16, 19], [20, 21], [22, 28], [29, 40], [40, 41], [42, 46], [47, 55], [56, 60], [61, 65], [66, 69], ...
    ['sthène', 'sthène', 'sthène', 'sthène', 'sthène']
    """
    
    train = data['train']
    data['dev'].sort(key=lambda x: len(x[1]))
    
    # 위에서 나머지
    dev = [x[:-1] for x in data['dev']]
    
    # dev_y 예시: ['sthène', 'sthène', 'sthène', 'sthène', 'sthène']
    dev_y = [x[-1] for x in data['dev']]
    return train, dev, dev_y, embedding, opt


class BatchGen:
    pos_size = None
    ner_size = None

    def __init__(self, data, batch_size, gpu, evaluation=False):
        """
        input:
            data - list of lists
            batch_size - int
        """
        self.batch_size = batch_size
        self.eval = evaluation
        self.gpu = gpu

        # sort by len
        data = sorted(data, key=lambda x: len(x[1]))
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        # shuffle
        if not evaluation:
            random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 8
            else:
                assert len(batch) == 10

            context_len = max(len(x) for x in batch[1])
            context_id = torch.LongTensor(batch_size, context_len).fill_(0)
            for i, doc in enumerate(batch[1]):
                context_id[i, :len(doc)] = torch.LongTensor(doc)

            feature_len = len(batch[2][0][0])

            context_feature = torch.Tensor(batch_size, context_len, feature_len).fill_(0)
            for i, doc in enumerate(batch[2]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = torch.Tensor(feature)

            context_tag = torch.Tensor(batch_size, context_len, self.pos_size).fill_(0)
            for i, doc in enumerate(batch[3]):
                for j, tag in enumerate(doc):
                    context_tag[i, j, tag] = 1

            context_ent = torch.Tensor(batch_size, context_len, self.ner_size).fill_(0)
            for i, doc in enumerate(batch[4]):
                for j, ent in enumerate(doc):
                    context_ent[i, j, ent] = 1

            question_len = max(len(x) for x in batch[5])
            question_id = torch.LongTensor(batch_size, question_len).fill_(0)
            for i, doc in enumerate(batch[5]):
                question_id[i, :len(doc)] = torch.LongTensor(doc)

            context_mask = torch.eq(context_id, 0)
            question_mask = torch.eq(question_id, 0)
            text = list(batch[6])
            span = list(batch[7])
            if not self.eval:
                y_s = torch.LongTensor(batch[8])
                y_e = torch.LongTensor(batch[9])
            if self.gpu:
                context_id = context_id.pin_memory()
                context_feature = context_feature.pin_memory()
                context_tag = context_tag.pin_memory()
                context_ent = context_ent.pin_memory()
                context_mask = context_mask.pin_memory()
                question_id = question_id.pin_memory()
                question_mask = question_mask.pin_memory()
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,
                       question_id, question_mask, y_s, y_e, text, span)


def _normalize_answer(s):
    """
    predictions로 튀어나온 답안을 깔끔하게 바꿔준다.
    :param s: predictions
    :return: 깔끔해진 predictions
    
    한글용
    1) 관사 빼기
    2) 공백 빼기
    3) 문장부호 빼기
    
    """
    def remove_articles(text):
        return re.sub(r'\b(이|그|저)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s)))


def _exact_match(pred, answers):
    # score 함수에서 적용될 때 pred는 p, answers는 t이다.
    # 사실상 아래 두 줄은 의미가 없다.
    # pred가 전혀 나오지 않거나 answers가 없다는 것은 무언가 잘못되었다는 뜻이니까
    if pred is None or answers is None:
        return False
    
    # pred(p)는 좀 더 깔끔하게 바꿔준다.
    pred = _normalize_answer(pred)
    
    # 정말로 단어가 똑같은지 확인해 본다.
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    # _exact_match 함수에서와 마찬가지로 아래 두 줄은 큰 의미는 없다.
    if pred is None or answers is None:
        return 0
    
    # pred를 깔끔하게 바꾼 뒤 split해준다.
    g_tokens = _normalize_answer(pred).split()

    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    # 영어의 경우 len(pred) = 10570 (dev set 길이)

    f1 = em = total = 0
    # p, t의 예시
    # p: treatment unnecessary
    # t: ['the treatment', 'treatment', 'treatment']
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1


if __name__ == '__main__':
    main()