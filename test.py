# Setting
import os
import sys
sys.path.append(os.getcwd())

from test_config import *


# from NER.data_utils import Vocabulary
# from NER_interact import entity
"""
This script serves as a template to be modified to suit all possible testing environments, including and not limited
to files (json, xml, csv, ...), web service, databases and so on.
To change this script to batch model, simply modify line 70 from "BatchGen([model_in], batch_size=1, ...)" to
"BatchGen([model_in_1, model_in_2, ...], batch_size=batch_size, ...)".
"""

parser = argparse.ArgumentParser(
    description='Interact with document reader model.'
)
parser.add_argument('--model-file', default='models/best_model.pt',
                    help='path to model file')
parser.add_argument("--cuda", type=str2bool, nargs='?',
                    const=True, default=torch.cuda.is_available(),
                    help='whether to use GPU acceleration.')
args = parser.parse_args()



if args.cuda:
    checkpoint = torch.load(args.model_file)
else:
    checkpoint = torch.load(args.model_file, map_location=lambda storage, loc: storage)

state_dict = checkpoint['state_dict']
opt = checkpoint['config']
with open('SQuAD/meta.msgpack', 'rb') as f:
    meta = msgpack.load(f, encoding='utf8')
embedding = torch.Tensor(meta['embedding'])
opt['pretrained_words'] = True
opt['vocab_size'] = embedding.size(0)
opt['embedding_dim'] = embedding.size(1)
opt['pos_size'] = len(meta['vocab_tag'])
opt['ner_size'] = len(meta['vocab_ent'])
opt['cuda'] = args.cuda
BatchGen.pos_size = opt['pos_size']
BatchGen.ner_size = opt['ner_size']
model = DocReaderModel(opt, embedding, state_dict)
w2id = {w: i for i, w in enumerate(meta['vocab'])}
tag2id = {w: i for i, w in enumerate(meta['vocab_tag'])}
ent2id = {w: i for i, w in enumerate(meta['vocab_ent'])}


with open('paragraph', 'rb') as fp:
    paragraph = pickle.load(fp)
with open('tfidf_p', 'rb') as fp:
    d = pickle.load(fp)
tfidf_p = d['vectorizer']
tdm_p = d['tdm']


















import pickle
def splt(x):
    return(x.split('      '))
#고유명사를 토크닝 해논걸 리스트로 만들어 놓은 변수
with open('feature_names_gpu.txt','rb') as f:
    feature_names_gpu = pickle.load(f)

#명사 tfidf 
with open('idf_maker.txt','rb') as f:
    idf_maker = pickle.load(f)
with open('tf_idf.txt','rb') as f:
    tf_idf = pickle.load(f)
#고유 명사 tfidf
with open('idf_maker_gou.txt','rb') as f:
    idf_maker_gou = pickle.load(f)
with open('tf_idf_gou.txt','rb') as f:
    tf_idf_gou= pickle.load(f)
#데이터
with open('data.txt','rb') as f:
    data = pickle.load(f)

with open('gou.txt','rb') as f:
    gou = pickle.load(f)


from itertools import chain
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from konlpy.tag import Twitter


idf_maker_gou = TfidfVectorizer(tokenizer=splt)

tf_idf_idf_maker_gou = idf_maker_gou.fit_transform(gou)
#질문의 고유명사를 토크닝하는 함수
def gou2(x):
    fff=[]
    for i in range(1,len(feature_names_gpu)):
        if x.find(feature_names_gpu[i])!=-1:
            fff.append(feature_names_gpu[i])              
    return('      '.join(fff))

#질문의 명사들을 twitter로 토크닝하는 함수
def word_separating2(movies) :


    words = twitter.nouns(movies)

    one_result_a = " ".join(words)


    return(list([one_result_a]))








while True:
    id_ = 0
    try:
        while True:
            #질문 input
            question = input("Question :"  )
            #명사들 tfidf 업뎃
            qa_tfidf = idf_maker.transform(word_separating2(question))
            #고유명사 tfidf 업뎃
            qa_tfidf3 = idf_maker_gou.transform([gou2(question).strip()])
            #고유명사 없으면 명사들로만 유사도 sorting
            if gou2(question)=='':
                c = list(chain.from_iterable(cosine_similarity(tf_idf[:],qa_tfidf[0])))
                sim_all = sorted(((v,i) for i, v in enumerate(c)),reverse=True)
            else:
                #고유명사 유사도 sorting
                qa_tfidf3 = idf_maker_gou.transform([gou2(question).strip()])
                c = list(chain.from_iterable(cosine_similarity(tf_idf[:],qa_tfidf[0])))
                e = list(chain.from_iterable(cosine_similarity(tf_idf_gou[:],qa_tfidf3[0])))
                
                #명사 유사도 + 고유명사 유사도 x 2
                sim_all = sorted(((v,i) for i, v in enumerate(list(np.array(np.array(c)+np.array(e)+np.array(e))))),reverse=True)

            #top5 인덱싱 list    
            max_sim_5 =[sim_all[0][1],sim_all[1][1],sim_all[2][1],sim_all[3][1],sim_all[4][1]]

            #top5 결과
            evidence =  " ".join(list(data['summary'][max_sim_5]))
            # evidence =  list(data['summary'][max_sim_5])[0]
            # print(evidence)
            if evidence.strip():
                break
    except EOFError:
        print()
        break
    id_ += 1
    start_time = time.time()
    annotated = annotate_interact(('interact-{}'.format(id_), evidence, question), meta['wv_cased'])
    model_in = to_id(annotated, w2id, tag2id, ent2id)
    # print(model_in)
    model_in = next(iter(BatchGen([model_in], batch_size=1, gpu=args.cuda, evaluation=True)))
    prediction = model.predict(model_in)[0]
    end_time = time.time()
    print('Answer : {}'.format(prediction))
    print('Time: {:.4f}s'.format(end_time - start_time))














