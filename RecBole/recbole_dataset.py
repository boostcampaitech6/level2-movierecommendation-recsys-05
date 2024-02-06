import os
import re
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datetime import datetime
from tqdm import tqdm

DATA_PATH = '/data/ephemeral/data/'

train = pd.read_csv(DATA_PATH + 'train_ratings.csv')
train['timestamp'] = train['time'].apply(lambda x : datetime.fromtimestamp(x))

userid, itemid = train['user'].unique(), train['item'].unique()

print(f'Number of unique train users : {len(userid)}')
print(f'Number of unique train items : {len(itemid)}')
print(f'Data sparsity ratio : {(1 - len(train) / (len(userid) * len(itemid))) * 100 :.1f}%')

directors = pd.read_csv(DATA_PATH + 'directors.tsv', sep = '\t')
titles = pd.read_csv(DATA_PATH + 'titles.tsv', sep = '\t')
writers = pd.read_csv(DATA_PATH + 'writers.tsv', sep = '\t')
genres = pd.read_csv(DATA_PATH + 'genres.tsv', sep = '\t')
years = pd.read_csv(DATA_PATH + 'years.tsv', sep = '\t')

DATA_DIR = '/data/ephemeral/RecBole/dataset/movierec/'
os.makedirs(DATA_DIR, exist_ok = True)

interfile = DATA_DIR + 'movierec.inter'
userfile  = DATA_DIR + 'movierec.user'
itemfile  = DATA_DIR + 'movierec.item'

# .inter 파일 생성
inter = pd.DataFrame()
inter[['user_id:token', 'item_id:token']] = train[['user', 'item']]
inter['rating:float'] = 1 # rating : (유저가 접한 아이템 1 / 아니면 0)
inter['timestamp:float'] = train['time'].values # timestamp
inter.to_csv(interfile, sep = '\t', index = False)
        
# .user 파일 생성
user = pd.DataFrame(sorted(userid), columns = ['user_id:token'])
user.to_csv(userfile, sep = '\t', index = False)
        

### item side-information 추가
train_df = pd.merge(train, years, on = 'item', how = 'left') # year
train_df = pd.merge(train_df, titles, on = 'item', how = 'left') # title

# title에서 연도 추출해서 year 결측치 대체
train_df['year_t'] = train_df['title'].apply(lambda x : x[-5:-1]).replace('007-', '2007').astype('float64')
train_df.loc[(train_df['year'] != train_df['year_t']) & train_df['year'].isnull(), 'year'] = train_df['year_t']
train_df['year'] = train_df['year'].astype('int')

# title에서 연도 제거 후 text preprocessing
def preprocess_title(title) :
    num_open_parentheses = title.count('(')
    if num_open_parentheses >= 2 :
        last_open_parenthesis_index = title.rfind('(')
        result = title[:last_open_parenthesis_index].strip()
    else :
        result = title.split('(')[0].strip()
    return result

train_df['title'] = train_df['title'].apply(preprocess_title)
train_df['title'].apply(lambda x : re.sub(r'[^0-9a-zA-Z\s]', '', x))

# item별 genre
item_genre = genres.groupby('item').agg(lambda x : ' '.join(x))['genre'].to_dict()
train_df['genre'] = train_df['item'].map(item_genre)


# .item 파일 생성
item = pd.DataFrame()
for i in tqdm(sorted(itemid)) :
    item_info = train_df.loc[(train_df['item'] == i), ['item', 'title', 'year', 'genre']].drop_duplicates(keep = 'first')
    item = pd.concat([item, item_info], axis = 0)

item.columns = ['item_id:token', 'movie_title:token', 'release_year:token', 'class:token_seq']
item.to_csv(itemfile, sep = '\t', index = False)