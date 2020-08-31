# Setting
import os
import sys
sys.path.append(os.getcwd())
import time
from train_config import *
# from NER_interact import *
twitter = Okt()


def main():
    args, log = setup()
    
    train = flatten_json(args.trn_file, 'train')
 #   train_contexts = [context[1] for context in train]###추가

    dev = flatten_json(args.dev_file, 'dev')
#    dev_contexts = [context[1] for context in dev]###추가
    log.info('csv data flattened.')

    # tokenize & annotate
    with Pool(args.threads) as p:
        annotate_ = partial(annotate, wv_cased=args.wv_cased, mode='train')
        train = list(tqdm(p.imap(annotate_, train, chunksize=args.batch_size), total=len(train), desc='train'))
        train = list(filter(lambda x: x is not None, train))
        
        annotate_ = partial(annotate, wv_cased=args.wv_cased, mode='dev')
        dev = list(tqdm(p.imap(annotate_, dev, chunksize=args.batch_size), total=len(dev), desc='dev  '))
        dev = list(filter(lambda x: x is not None, dev))
    # with open('train_entity.txt', 'rb') as f:
    #     train_entity = pickle.load(f)
    # with open('dev_entity.txt', 'rb') as f:
    #     dev_entity = pickle.load(f)
    for i in range(len(train)):
        train[i] = list(train[i])
        # train[i][4] = train_entity[train[i][8]]
        train[i] = train[i][:8] + train[i][9:]
        train[i] = tuple(train[i])
    for i in range(len(dev)):
        dev[i] = list(dev[i])
        # dev[i][4] = dev_entity[dev[i][8]]
        dev[i] = dev[i][:8] + dev[i][9:]
        dev[i] = tuple(dev[i])
    print(dev[:1])
    print(train[-1])
    print("Start_time : ", time.strftime('%H%M%S'))
    
    # contexts = (row for row in train_contexts)
    # from NER_interact import entity
    # for i in tqdm(range(len(train_contexts))):
    #     train[i] = list(train[i])
    #     train[i][4] = entity(next(contexts))
    #     train[i] = tuple(train[i])
    #     sys.stdout.flush()
    
    # contexts = (row for row in dev_contexts)
    # for i in tqdm(range(len(dev_contexts))):
    #     dev[i] = list(dev[i])
    #     dev[i][4] = entity(next(contexts))
    #     dev[i] = tuple(dev[i])
    #     sys.stdout.flush()
    # print("끝남", time.strftime('%H%M%S'))
    train = list(map(index_answer, train))
    initial_len = len(train)
    train = list(filter(lambda x: x[-1] is not None, train))
    log.info('drop {} inconsistent samples.'.format(initial_len - len(train)))
    log.info('tokens generated')

    # load vocabulary from word vector files
    wv_vocab = set()
    with open(args.wv_file, encoding='UTF-8') as f:
        for line in f:
            token = normalize_text(line.rstrip().split(' ')[0])
            wv_vocab.add(token)
    log.info('glove vocab loaded.')
        
    # build vocabulary
    full = train + dev
    vocab, counter = build_vocab([row[5] for row in full], [row[1] for row in full], wv_vocab, args.sort_all)
    total = sum(counter.values())
    matched = sum(counter[t] for t in vocab)
    log.info('vocab coverage {1}/{0} | OOV occurrence {2}/{3} ({4:.4f}%)'.format(
        len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
    counter_tag = collections.Counter(w for row in full for w in row[3])
    vocab_tag = sorted(counter_tag, key=counter_tag.get, reverse=True)
    counter_ent = collections.Counter(w for row in full for w in row[4])
    vocab_ent = sorted(counter_ent, key=counter_ent.get, reverse=True)
    w2id = {w: i for i, w in enumerate(vocab)}
    tag2id = {w: i for i, w in enumerate(vocab_tag)}
    ent2id = {w: i for i, w in enumerate(vocab_ent)}
    log.info('Vocabulary size: {}'.format(len(vocab)))
    log.info('Found {} POS tags.'.format(len(vocab_tag)))
    log.info('Found {} entity tags: {}'.format(len(vocab_ent), vocab_ent))

    to_id_ = partial(to_id, w2id=w2id, tag2id=tag2id, ent2id=ent2id)
    train = list(map(to_id_, train))
    dev = list(map(to_id_, dev))
    log.info('converted to ids.')

    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, args.wv_dim))
    embed_counts = np.zeros(vocab_size)
    embed_counts[:2] = 1  # PADDING & UNK
    with open(args.wv_file, encoding='UTF-8') as f:
        for line in f:
            elems = line.rstrip().split(' ')
            token = normalize_text(elems[0])
            
            if token in w2id:
                word_id = w2id[token]
                embed_counts[word_id] += 1
                try:
                    embeddings[word_id] += [float(v) for v in elems[1:]]
                except:
                    pass
    log.info('got embedding matrix.')

    meta = {
        'vocab': vocab,
        'vocab_tag': vocab_tag,
        'vocab_ent': vocab_ent,
        'embedding': embeddings.tolist(),
        'wv_cased': args.wv_cased,
    }
    with open('SQuAD/meta.msgpack', 'wb') as f:
        msgpack.dump(meta, f)



    result = {
        'train': train,
        'dev': dev
    }
    # train: id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer_start, answer_end
    # dev:   id, context_id, context_features, tag_id, ent_id,
    #        question_id, context, context_token_span, answer
    with open('SQuAD/data.msgpack', 'wb') as f:
        msgpack.dump(result, f)



    if args.sample_size:
        sample = {
            'train': train[:args.sample_size],
            'dev': dev[:args.sample_size]
        }
        with open('SQuAD/sample.msgpack', 'wb') as f:
            msgpack.dump(sample, f)
    log.info('saved to disk.')










def setup():
    parser = argparse.ArgumentParser(
        description='Preprocessing data files, about 10 minitues to run.'
    )
    parser.add_argument('--trn_file', default='SQuAD/train-v1.1.json',
                        help='path to train file.')
    parser.add_argument('--dev_file', default='SQuAD/dev-v1.1.json',
                        help='path to dev file.')
    parser.add_argument('--wv_file', default='pretrained/w2v_model_skip_twitter_vector_textfile_clean_300_dim_naver.txt',
                        help='path to word vector file.')
    parser.add_argument('--wv_dim', type=int, default=300,
                        help='word vector dimension.')
    parser.add_argument('--wv_cased', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='treat the words as cased or not.')
    parser.add_argument('--sort_all', action='store_true',
                        help='sort the vocabulary by frequencies of all words. '
                             'Otherwise consider question words first.')
    parser.add_argument('--sample_size', type=int, default=2000,
                        help='size of sample data (for debugging).')
    parser.add_argument('--threads', type=int, default=min(multiprocessing.cpu_count(), 6),
                        help='number of threads for preprocessing.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size for multiprocess tokenizing and tagging.')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                        datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)
    log.info(vars(args))
    log.info('start data preparing...')

    return args, log

# def flatten_csv(data_file,mode):
#     data = pd.read_csv(data_file, encoding="UTF-8")
#
#     rows = []
#     for i in range(len(data)):
#         id_ = data.iloc[i,3]
#         context = data.iloc[i,4]
#         question = data.iloc[i,0]
#         answer = data.iloc[i,1]
#         if mode == 'train':
#             answer_start = data.iloc[i,2]
#             answer_end = answer_start + len(answer)
#             rows.append((id_, context, question, answer, answer_start, answer_end))
#         else:
#             rows.append((id_, context, question, answer))
#     return rows

def flatten_json(data_file, mode):
    """Flatten each article in training data."""
    with open(data_file, encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for article in data:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                id_, question, answers = qa['id'], qa['question'], qa['answers']
                if mode == 'train':
                    answer = answers[0]['text']  # in training data there's only one answer
                    answer_start = answers[0]['answer_start']
                    answer_end = answer_start + len(answer)
                    rows.append((id_, context, question, answer, answer_start, answer_end))
                else:  # mode == 'dev'
                    answers = answers[0]['text']
                    rows.append((id_, context, question, answers))
    return rows
    
def clean_spaces(text):
    """normalize spaces in a string."""
    text = re.sub(r'\s', ' ', text)
    return text


def normalize_text(text):
    return unicodedata.normalize('NFD', text)

def after_annotate(tuples):
   from NER_interact import entity
   #from NER.data_utils import Vocabulary
   for i, t in tqdm(enumerate(tuples)):
       tuples[i] = list(t)
       tuples[i][4] = entity(tuples[i][8])
       tuples[i] = tuple(tuples[i][:8]) + tuple(tuples[i][9:])
   return tuples
# def annotate(row, wv_cased, mode):
#     id_, context, question = row[:3]
#     context = re.sub('[^가-힣.0-9A-Za-z~ ]','',context)     # 한글, 마침표, 숫자, 영어, 물결 제외 제거
#     question = re.sub('[^가-힣.0-9A-Za-z~ ]','',question)
#     from NER_interact import entity
#     q_doc = twitter.pos(question)      # 토큰화
#     c_doc = entity(context)

#     q_token, q_pos = zip(*q_doc)     # 토큰, 형태소 분리
#     c_token, c_pos, c_ent = zip(*c_doc)

#     q_doc = " ".join(list(q_token))   # 토큰화 된 것을 띄어쓰기 구분으로 다시 합치기
#     c_doc = " ".join(list(c_token))

#     question_tokens = [normalize_text(w) for w in q_token]   # 코덱변경
#     context_tokens = [normalize_text(w) for w in c_token]

#     question_tokens_lower = [w.lower() for w in question_tokens]  # 소문자
#     context_tokens_lower = [w.lower() for w in context_tokens]

#     context_token_span = []    # 토큰화된 context 기준 각 토큰 start, end 위치
#     for idx, w in enumerate(c_token):
#         if idx == 0:
#             s = 0
#             context_token_span += [(s,len(w))]
#         else:
#             e = s + len(w)
#             context_token_span += [(s, e)]
#         s += len(w) + 1

#     context_tags = list(c_pos)    # pos tagging
#     context_ents = list(c_ent)     # 개체명 tagging

#     question_lemma = {twitter.pos(w, norm=True, stem=True)[0][0] for w in q_token}   # 표준화
#     question_tokens_set = set(question_tokens)      # 토큰 셋
#     question_tokens_lower_set = set(question_tokens_lower)   # 토큰 소문자 셋

#     match_origin = [w in question_tokens_set for w in context_tokens]      # context 토큰 안에 question 토큰 있으면 T, 아니면 F
#     match_lower = [w in question_tokens_lower_set for w in context_tokens_lower] # context 토큰 안에 question 토큰 있으면 T, 아니면 F
#     match_lemma = [(w.lower()) in question_lemma for w in c_doc]  # context 토큰에서 표제어와 같으면 T, 아니면 F
            
#     counter_ = collections.Counter(context_tokens_lower)     # 각 context 토큰별 빈도
#     total = len(context_tokens_lower)      # 토큰 빈도 합
#     context_tf = [counter_[w] / total for w in context_tokens_lower]    # 각 토큰의 빈도율
#     context_features = list(zip(match_origin, match_lower, match_lemma, context_tf)) 
#     answer = row[3]          # answer 전처리 시작
#     answer = re.sub('[^가-힣.0-9A-Za-z~ ]','',answer)    
#     a_doc = twitter.pos(answer)
#     a_token, a_pos = zip(*a_doc)
#     answer_token = " ".join(list(a_token))
#     answer_token_s = c_doc.find(answer_token)

#     if mode == 'train':   # train은 정답토큰 위치까지
#         answer = (answer_token, answer_token_s,answer_token_s + len(answer_token))
#     else:
#         answer = tuple([[answer_token]])

#     if not wv_cased:
#         context_tokens = context_tokens_lower
#         question_tokens = question_tokens_lower
#     return (id_, context_tokens, context_features, context_tags, context_ents,
#             question_tokens, c_doc, context_token_span) + answer

def tuples(self, A):
    try:
        return tuple(self.tuples(a) for a in A)
    except TypeError:
        return A


def annotate(row, wv_cased, mode):
    try:
        id_, context, question = row[:3]
        context = re.sub('[^가-힣.0-9A-Za-z~ ]','',context)     # 한글, 마침표, 숫자, 영어, 물결 제외 제거
        question = re.sub('[^가-힣.0-9A-Za-z~ ]','',question)

        q_doc = twitter.pos(question)      # 토큰화
        c_doc = twitter.pos(context)

        q_token, q_pos = zip(*q_doc)     # 토큰, 형태소 분리
        c_token, c_pos = zip(*c_doc)

        q_doc = " ".join(list(q_token))   # 토큰화 된 것을 띄어쓰기 구분으로 다시 합치기
        c_doc = " ".join(list(c_token))

        question_tokens = [normalize_text(w) for w in q_token]   # 코덱변경
        context_tokens = [normalize_text(w) for w in c_token]

        question_tokens_lower = [w.lower() for w in question_tokens]  # 소문자
        context_tokens_lower = [w.lower() for w in context_tokens]

        context_token_span = []    # 토큰화된 context 기준 각 토큰 start, end 위치
        for idx, w in enumerate(c_token):
            if idx == 0:
                s = 0
                context_token_span += [(s,len(w))]
            else:
                e = s + len(w)
                context_token_span += [(s, e)]
            s += len(w) + 1

        context_tags = list(c_pos)    # pos tagging
        context_ents = ['' for t in c_token]     # 개체명 tagging

        question_lemma = {twitter.pos(w, norm=True, stem=True)[0][0] for w in q_token}   # 표준화
        question_tokens_set = set(question_tokens)      # 토큰 셋
        question_tokens_lower_set = set(question_tokens_lower)   # 토큰 소문자 셋

        match_origin = [w in question_tokens_set for w in context_tokens]      # context 토큰 안에 question 토큰 있으면 T, 아니면 F
        match_lower = [w in question_tokens_lower_set for w in context_tokens_lower] # context 토큰 안에 question 토큰 있으면 T, 아니면 F
        match_lemma = [(w.lower()) in question_lemma for w in c_doc]  # context 토큰에서 표제어와 같으면 T, 아니면 F
                
        counter_ = collections.Counter(context_tokens_lower)     # 각 context 토큰별 빈도
        total = len(context_tokens_lower)      # 토큰 빈도 합
        context_tf = [counter_[w] / total for w in context_tokens_lower]    # 각 토큰의 빈도율
        context_features = list(zip(match_origin, match_lower, match_lemma, context_tf)) 
        answer = row[3]          # answer 전처리 시작
        answer = re.sub('[^가-힣.0-9A-Za-z~ ]','',answer)    
        a_doc = twitter.pos(answer)
        a_token, a_pos = zip(*a_doc)
        answer_token = " ".join(list(a_token))
        answer_token_s = c_doc.find(answer_token)

        if mode == 'train':   # train은 정답토큰 위치까지
            answer = (answer_token, answer_token_s,answer_token_s + len(answer_token))
        else:
            answer = tuple([[answer_token]])

        if not wv_cased:
            context_tokens = context_tokens_lower
            question_tokens = question_tokens_lower
        return (id_, context_tokens, context_features, context_tags, context_ents,
                question_tokens, c_doc, context_token_span, context) + answer
    except:
        pass

def annotate_interact(row, wv_cased):
    # from NER.data_utils import Vocabulary
    # from NER_interact import entity
    
    id_, context, question = row[:3]
    context = re.sub('[^가-힣.0-9A-Za-z~ ]','',context)
    question = re.sub('[^가-힣.0-9A-Za-z~ ]','',question)

    q_doc = twitter.pos(question)
    c_doc = twitter.pos(context)

    q_token, q_pos = zip(*q_doc)
    c_token, c_pos = zip(*c_doc)

    q_doc = " ".join(list(q_token))
    c_doc = " ".join(list(c_token))

    question_tokens = [normalize_text(w) for w in q_token]   # 코덱변경
    context_tokens = [normalize_text(w) for w in c_token]

    question_tokens_lower = [w.lower() for w in question_tokens]  # 소문자
    context_tokens_lower = [w.lower() for w in context_tokens]

    context_token_span = []
    for idx, w in enumerate(c_token):
        if idx == 0:
            s = 0
            context_token_span += [(s,len(w))]
        else:
            e = s + len(w)
            context_token_span += [(s, e)]
        s += len(w) + 1

    context_tags = list(c_pos)    # pos tagging
    context_ents = ['' for t in c_token]     # 개체명 tagging

    question_lemma = {twitter.pos(w, norm=True, stem=True)[0][0] for w in q_token}   # 표준화
    question_tokens_set = set(question_tokens)      # 토큰 셋
    question_tokens_lower_set = set(question_tokens_lower)   # 토큰 소문자 셋

    match_origin = [w in question_tokens_set for w in context_tokens]
    match_lower = [w in question_tokens_lower_set for w in context_tokens_lower]
    match_lemma = [(w.lower()) in question_lemma for w in c_doc] 

    counter_ = collections.Counter(context_tokens_lower)
    total = len(context_tokens_lower)
    context_tf = [counter_[w] / total for w in context_tokens_lower]
    context_features = list(zip(match_origin, match_lower, match_lemma, context_tf))

    return (id_, context_tokens, context_features, context_tags, context_ents,
            question_tokens, c_doc, context_token_span) + row[3:]


def index_answer(row):
    token_span = row[-4]
    starts, ends = zip(*token_span)
    answer_start = row[-2]
    answer_end = row[-1]
    try:
        return row[:-3] + (starts.index(answer_start), ends.index(answer_end))
    except ValueError:
        return row[:-3] + (None, None)


def build_vocab(questions, contexts, wv_vocab, sort_all=False):
    """
    Build vocabulary sorted by global word frequency, or consider frequencies in questions first,
    which is controlled by `args.sort_all`.
    """
    if sort_all:
        counter = collections.Counter(w for doc in questions + contexts for w in doc)
        vocab = sorted([t for t in counter if t in wv_vocab], key=counter.get, reverse=True)
    else:
        counter_q = collections.Counter(w for doc in questions for w in doc)
        counter_c = collections.Counter(w for doc in contexts for w in doc)
        counter = counter_c + counter_q
        vocab = sorted([t for t in counter_q if t in wv_vocab], key=counter_q.get, reverse=True)
        vocab += sorted([t for t in counter_c.keys() - counter_q.keys() if t in wv_vocab],
                        key=counter.get, reverse=True)
    vocab.insert(0, "<PAD>")
    vocab.insert(1, "<UNK>")
    return vocab, counter


def to_id(row, w2id, tag2id, ent2id, unk_id=1):
    context_tokens = row[1]
    context_features = row[2]
    context_tags = row[3]
    context_ents = row[4]
    question_tokens = row[5]
    question_ids = [w2id[w] if w in w2id else unk_id for w in question_tokens]
    context_ids = [w2id[w] if w in w2id else unk_id for w in context_tokens]
    tag_ids = [tag2id[w] for w in context_tags]
    ent_ids = [ent2id[w] for w in context_ents]
    return (row[0], context_ids, context_features, tag_ids, ent_ids, question_ids) + row[6:]





if __name__ == '__main__':
    # from NER.data_utils import Vocabulary
    main()
