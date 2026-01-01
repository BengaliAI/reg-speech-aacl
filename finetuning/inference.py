# In[1]:


import os
import csv
import time
import glob
import jiwer  # you may need to install this library
import csv
import pandas as pd
import glob
import shutil
import librosa
import argparse
import warnings
from pathlib import Path
import transformers
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from tqdm.auto import tqdm
import torch
import time

    
import warnings

warnings.filterwarnings("ignore")

################# CONSTANTS ###################

dir_root = '/home/phison/LargeFiles/'
DATASET_PATH = '/home/phison/LargeFiles/iut-comp-dataset/16_kHz_test_audio'
MODEL = '/home/phison/LargeFiles/regional-asr/whisper-base-bn'
path_gt = dir_root+'/iut-comp-dataset/test.csv'

domain_weights = {
        'Barishal': 0.125,
        'Chittagong': 0.083,
        'Habiganj': 0.125,
        'Kishoreganj': 0.083,
        'Narail': 0.083, 
        'Narsingdi': 0.083,
        'Rangpur': 0.083,
        'Sylhet': 0.125,
        'Sandwip': 0.125,
        'Tangail': 0.083,
    }
domain_weights = { key.lower(): value for key, value in domain_weights.items()}
# unseen: Habiganj, Barishal, Sylhet, Sandwip

CHUNK_LENGTH_S = 20.1
ENABLE_BEAM = True

BATCH_SIZE = 32*8

######################## UTILITY FUNCTIONS ############################
def fix_repetition(text, max_count):
    uniq_word_counter = {}
    words = text.split()
    for word in text.split():
        if word not in uniq_word_counter:
            uniq_word_counter[word] = 1
        else:
            uniq_word_counter[word] += 1

    for word, count in uniq_word_counter.items():
        if count > max_count:
            words = [w for w in words if w != word]
    text = " ".join(words)
    return text
def batchify(inputs, batch_size):
    for i in range(0, len(inputs), batch_size):
        yield inputs[i:i + batch_size]


def mean_wer(solution, submission):
    joined = solution.merge(submission.rename(columns={'sentence': 'predicted'}))
#     print(joined)
    domain_scores = joined.groupby('domain').apply(
        # note that jiwer.wer computes a weighted average wer by default when given lists of strings
        lambda df: jiwer.wer(df['sentence'].to_list(), df['predicted'].to_list()),
    )
    # print(domain_scores)
    domain_scores_unweighted = domain_scores.copy()

    for key, value in domain_weights.items():
        domain_scores.loc[key] = domain_scores.loc[key].item()*value
    # print(domain_scores)
    return domain_scores_unweighted, domain_scores.sum()

def mean_cer(solution, submission):
    joined = solution.merge(submission.rename(columns={'sentence': 'predicted'}))
#     print(joined)
    domain_scores = joined.groupby('domain').apply(
        # note that jiwer.wer computes a weighted average wer by default when given lists of strings
        lambda df: jiwer.cer(df['sentence'].to_list(), df['predicted'].to_list()),
    )
    # print(domain_scores)
    domain_scores_unweighted = domain_scores.copy()

    for key, value in domain_weights.items():
        domain_scores.loc[key] = domain_scores.loc[key].item()*value
    # print(domain_scores)
    return domain_scores_unweighted, domain_scores.sum()



#################### PREPARE DATA AND MODEL, DO INFERENCE, SAVE RESULT #######################

solution = pd.read_csv(path_gt)
print(solution.head())
solution['paths'] = solution['file_name'].apply(lambda x: os.path.join(DATASET_PATH, x))
files = solution['paths'].to_list()

print('files', len(files))


pipe = pipeline(task="automatic-speech-recognition",
                model=MODEL,
                tokenizer=MODEL,
                chunk_length_s=CHUNK_LENGTH_S, device=0, 
#                 batch_size=BATCH_SIZE
               )
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="bn", task="transcribe")

print("model loaded!")

generate_kwargs = {"max_length": 260, "num_beams": 4} if ENABLE_BEAM else None
start = time.time()
texts = []
for batch in batchify(files, BATCH_SIZE):
    texts+=pipe(batch, generate_kwargs = generate_kwargs)
    elapsed = time.time()-start
    print('completed: {}, avg. sec/sample: {:.2f}'.format(len(texts), elapsed/len(texts)))
print('total time: {:.2f}'.format(time.time()-start))

predictions = []
with open(MODEL+'/'+"submission.csv", 'wt', encoding="utf8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_name', 'sentence'])
    for f, text in zip(files, texts):
        file_id = Path(f).stem
        pred = text['text'].strip()
        pred = fix_repetition(pred, max_count=8)
        if len(pred) == 0:
            print('empty prediction on', f)
            pred = ' '
        prediction = [file_id, pred]
        writer.writerow(prediction)
        predictions.append(prediction)
print("inference finished!")
'''
submission = pd.read_csv(MODEL+'/'+"submission.csv")
print(submission)

solution = solution.rename(columns = {'transcripts': 'sentence', 'district': 'domain'}).drop(columns = ['paths'])
solution['file_name'] = solution['file_name'].apply(lambda x: x.replace('.wav', ''))
print(solution)

domain_scores_unweighted, weighted_wer = mean_wer(solution, submission)
print(domain_scores_unweighted)
print(weighted_wer)
domain_scores_unweighted.to_csv(MODEL+'/'+'domain_scores_unweighted.csv', index = False)
with open (MODEL+'/'+'weighted_wer.txt','w') as f:
    f.write(str(weighted_wer))
    
'''
