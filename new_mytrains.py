#! -*- coding:utf-8 -*-

import json

import pandas as pd
from tqdm import tqdm
import codecs
from sklearn.model_selection import train_test_split

df=pd.read_excel('三元组实例.xlsx')
print(df)
traindf,devdf=train_test_split(df,test_size=0.2)




all_50_schemas = set()

for l in tqdm(df['predicate'].to_list()):
    all_50_schemas.add(l)

id2predicate = {i+1:j for i,j in enumerate(all_50_schemas)} # 0表示终止类别
predicate2id = {j:i for i,j in id2predicate.items()}

with codecs.open('all_50_schemas_me.json', 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)


chars = {}
min_count = 2


train_data = []


for l in tqdm(set(traindf['text'].to_list())):
    a=traindf[traindf['text']==l]
    a=[a.iloc[x,:] for x in range(len(a))]

    train_data.append(
        {
            'text': l,
            'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a]
        }
    )
    for c in l:
        chars[c] = chars.get(c, 0) + 1

with codecs.open('train_data_me.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


dev_data = []


for l in tqdm(set(devdf['text'].to_list())):
    a=devdf[devdf['text']==l]
    a=[a.iloc[x,:] for x in range(len(a))]
    dev_data.append(
        {
            'text': l,
            'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a]
        }
    )
    for c in l:
        chars[c] = chars.get(c, 0) + 1

with codecs.open('dev_data_me.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)

with codecs.open('all_chars_me.json', 'w', encoding='utf-8') as f:
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # padding: 0, unk: 1
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)