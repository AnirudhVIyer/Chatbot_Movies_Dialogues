from pathlib import Path
import warnings


from config import config as con
from chat import utils
from chat import data
from chat import train
from chat import evaluate
from chat import model
# extract zip contents

def extract_zip():

    import zipfile
    with zipfile.ZipFile(con.zip_location,"r") as zip_ref:
        zip_ref.extractall(con.DATA_DIR)


def load_data():
    print(con.utterance_location)

    utils.extraction_wrapper(con.utterance_location,con.movie_lines_location)

    return 


def pre_process_data():
        # Load/Assemble voc and pairs
    
    voc, pairs = data.loadPrepareData(corpus=con.corpus, corpus_name=con.corpus_name, datafile=con.movie_lines_location, save_dir=con.SAVE_DIR)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    print('done')
    # Trim voc and pairs
    pairs = data.trimWords(voc, pairs)

    # print('creating tensors')
    # input_variable, lengths, target_variable, mask, max_target_len = data.tensorise_data(voc,pair)
    return voc, pairs
    

def training():
    #initialize the models and train on the dataset

    #1. pre_process_data
    voc, pairs = pre_process_data()

    train.train_model(voc,pairs,con.loadFileName)


def predict():

    voc, _ = pre_process_data()
    #intialize model
    encoder, decoder = evaluate.load_model(voc=voc)
    
    print('enter q to exit')
    # Initialize search module
    searcher = utils.GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    evaluate.evaluateInput(encoder, decoder, searcher, voc)


