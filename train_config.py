# System Setting
import os
import sys
import json
import unicodedata
import logging
import argparse
import pandas as pd

from tqdm import tqdm
from functools import partial
import pickle
import time

# Math Setting
import math
import random
import string
import numpy as np
import collections

# warnings remove
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim

# Language Setting
import re
import spacy
from gensim.models import Word2Vec
from konlpy.tag import *

from shutil import copyfile
from datetime import datetime
from collections import Counter

# Learning
import torch
import msgpack
from drqa.model import DocReaderModel
from drqa.utils import str2bool
import multiprocessing
from multiprocessing import Pool


