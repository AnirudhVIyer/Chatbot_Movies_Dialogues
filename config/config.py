# config.py
from pathlib import Path
import os
from config import config

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR  = Path(BASE_DIR,'data')

DATA_DIR.mkdir(parents=True, exist_ok=True)



# Data Zip
zip_location = '/Users/anirudhiyer/Downloads/nlp/Chatbot/movie-corpus.zip'
corpus_name = "movie-corpus"
utterance_location = os.path.join(DATA_DIR, corpus_name,'utterances.jsonl')
movie_lines_location = os.path.join(DATA_DIR, 'formatted_movie_lines.txt')
corpus = Path(DATA_DIR, corpus_name)
SAVE_DIR = Path(DATA_DIR,'save')

# Data Constants
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 10  # Maximum sentence length to consider


# paste from model_version according to the desired model
#

#BATCH SIZE
    #BATCH SIZE
batch_size = 64
MIN_COUNT = 3


# model config
# Configure models
model_name = 'cb_model'
#attn_model = 'dot'
#attn_model = 'general'
attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
loadFileName = None
teacher_forcing_ratio = 1.0



clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500