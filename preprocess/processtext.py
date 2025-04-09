import json
from datetime import datetime
import os
import time
import pickle
import numpy as np
import pandas as pd
import random
import time
import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

datasets_path = 'Clothing_Shoes_and_Jewelry/'
# Open text
datasets_name = 'Clothes'
id2text_path = 'dict/' + datasets_path + 'id2text.npy'
id2text_dict = np.load(id2text_path,allow_pickle=True).item()
asin_list_path = 'dict/' + datasets_path + 'asinlist.npy'
asin_list = np.load(asin_list_path,allow_pickle=True)
id_list = []
text_list = []

batch_size = 100

for key in tqdm(asin_list):
    value = id2text_dict[key]
    text_temp = " ".join(value)
    text_list.append(text_temp)
    # if key == 'B0000CFMU3':
    #     break

text = text_list
local_path = "/root/WWW/MMSBR/preprocess/bert-base-uncased/"
tokenizer = BertTokenizer.from_pretrained(local_path)
bert = BertModel.from_pretrained(local_path)

text_embeddings = []

num_text = len(text)
epoch = num_text//batch_size
for i in tqdm(range(epoch)):
    temp_text = text[i*batch_size:(i+1)*batch_size]
    inputs = tokenizer(temp_text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=200)
    outputs = bert(inputs['input_ids'])
    last_hidden_state = outputs.last_hidden_state  # [batch_size, sequence_length, embedding_size]
    cls = outputs.pooler_output  # [batch_size, embedding_size]
    text_embeddings += cls.tolist()
if num_text%batch_size != 0:
    temp_text = text[epoch*batch_size:]
    inputs = tokenizer(temp_text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=200)
    outputs = bert(inputs['input_ids'])
    last_hidden_state = outputs.last_hidden_state  # [batch_size, sequence_length, embedding_size]
    cls = outputs.pooler_output  # [batch_size, embedding_size]
    text_embeddings += cls.tolist()

output_df = pd.DataFrame(text_embeddings, index=text_list)
output_df.to_csv(datasets_name + '/txt_features.csv')

print(f"Text embeddings saved to 'txt_embeddings.csv'")
print('total text: ', str(len(text_embeddings)))
print('done')
