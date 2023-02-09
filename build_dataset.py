import torch
import hanlp
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
from copy import deepcopy
from tqdm import tqdm

dep = hanlp.load(hanlp.pretrained.dep.PMT1_DEP_ELECTRA_SMALL)
tok = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
tok.config.output_spans = True

tokenizer = AutoTokenizer.from_pretrained("/home/gayler/.transformers/ernie-3.0-base-zh")

tokenizer.add_special_tokens({"additional_special_tokens": ['[PEOS]', '[BOS]', '[EOS]']})

asp_dict = {'外观': 0, '舒适性': 1, '操控': 2, '动力': 3, '内饰': 4, '空间': 5}


def get_next(t):
    i, j = 0, -1
    next = [-1] * len(t)
    while i < len(t) - 1:
        if j == -1 or t[i] == t[j]:
            i, j = i + 1, j + 1
            next[i] = j
        else:
            j = next[j]
    return next


def kmp(s, t):
    next = get_next(t)
    i, j = -1, -1
    while j != len(t) and i < len(s):
        if s[i] == t[j] or j == -1:
            i, j = i + 1, j + 1
        else:
            j = next[j]
    return i - j if j == len(t) else None


def make_snippet(s, t):
    start_idx = None
    left = 0
    while start_idx is None and left < len(t):
        start_idx = kmp(s, t[left:])
        left = left + 1
    if start_idx is None:
        print(s)
        print(t)
        raise ValueError('start idx not found')
    return start_idx


def make_graph(src, src_tokenized):
    src_tokenized_len = len(src_tokenized)
    hanlp_tokenized = [phrase[0] for phrase in tok(src)]
    dependency_tree = dep(hanlp_tokenized)
    hanlp_tokenized_len = len(hanlp_tokenized)
    graph = np.zeros(shape=(src_tokenized_len, src_tokenized_len)).astype('uint8')
    mapping: List = [None] * hanlp_tokenized_len
    
    i = 0
    j = 0
    prefix = ""
    prefix_idx = []
    prev_flag = False
    while i < src_tokenized_len and j < hanlp_tokenized_len:
        ernie_: str = src_tokenized[i].replace("##", "")
        hanlp_: str = hanlp_tokenized[j]
        flag = (ernie_ == "[UNK]")
        prev_flag = True if flag else prev_flag
        # find next non-UNK
        while ernie_ == "[UNK]" and i < src_tokenized_len - 1:
            i += 1
            ernie_ = src_tokenized[i].replace("##", "")
        # step hanlp
        if flag and j < hanlp_tokenized_len - 1:
            j += 1
            hanlp_ = hanlp_tokenized[j]
            prefix = ""
            prefix_idx = []
        if flag:
            while not hanlp_.startswith(ernie_) and j < hanlp_tokenized_len - 1:
                j += 1
                hanlp_ = hanlp_tokenized[j]
        if not flag or (i < src_tokenized_len and j < hanlp_tokenized_len):
            prefix += ernie_
            prefix_idx.append(i)
            if prefix == hanlp_:
                mapping[j] = deepcopy(prefix_idx)
                prefix = ""
                prefix_idx = []
                j += 1
            i += 1

    assert i == src_tokenized_len and j == hanlp_tokenized_len

    dependency_pair = [[item['id'] - 1, item['head'] - 1] for item in dependency_tree]
    for idx, head in dependency_pair:
        if head >= 0 and mapping[idx] is not None:
            for i in mapping[idx]:
                for j in mapping[head]:
                    graph[i, j] = graph[j, i] = 1
    # entire word inner link
    for entire in mapping:
        if entire is not None:
            for i in entire:
                for j in entire[i:]:
                    graph[i, j] = graph[j, i] = 2
    graph = np.pad(graph, ((1, 1), (1, 1)), 'constant')
    graph[0, 0] = graph[-1, -1] = 2
    return graph


def make_sample(sample):
    src = sample['content'].strip().lower()
    src = src.replace('～', '~')
    target = sample['target']
    opinion = sample['opinion']
    keyphrase = target + '[PEOS]' + opinion
    snippet = sample['snippet']
    sentiment = sample['sentiment']
    asp = asp_dict[sample['asp']]
    
    src_tokenized = tokenizer.tokenize(src)
    src_ids = tokenizer.encode(src)
    
    trg_ids = tokenizer.encode(keyphrase)
    
    snippet_tokenized = tokenizer.tokenize(snippet)
    snippet_ids = tokenizer.convert_tokens_to_ids(snippet_tokenized)
    
    start_idx = make_snippet(src_ids, snippet_ids)
    
    snippet_mask = [1 if start_idx <= idx < start_idx + len(snippet_ids) else 0 for idx in range(len(src_ids))]
    
    graph = make_graph(src, src_tokenized)
    
    return {'src_ids': src_ids, 'trg_ids': trg_ids, 'snippet': snippet_mask, 'graph': graph, 'sentiment': sentiment, 'asp': asp}


def build_dataset(data_path, dest_path):
    dataset = []
    with open(data_path, 'r') as f:
        for line in tqdm(f):
            sample = json.loads(line)
            dataset.append(make_sample(sample))
    torch.save(dataset, open(dest_path, 'wb'))
    

if __name__ == '__main__':
    valid_data_path = './data/valid.json'
    valid_dest_path = './data/valid_dataset_ernie.pt'
    build_dataset(valid_data_path, valid_dest_path)