# Setting
import time
import argparse
import torch
import msgpack
from drqa.model import DocReaderModel
from drqa.utils import str2bool
from prepro import annotate, to_id, annotate_interact
from train import BatchGen
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import re

twitter = Okt()

def get_nouns(x):
    x = re.sub('[^가-힣.0-9A-Za-z~ ]', '', x)
    x = twitter.nouns(x)
    x = [word for word in x if len(word) > 1]
    return(x)