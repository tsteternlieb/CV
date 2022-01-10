from numpy import mod
from tokenizers import BertWordPieceTokenizer
from torch._C import ModuleDict
from transformers import BertTokenizer
from pathlib import Path
import os, random
from typing import Union
import datasets
import torch

from torch.utils.data import Dataset, DataLoader



def generateDataset(dataset_name: str):
    """writes dataset to local files for later consumption

    Args:
        dataset_name (str): name of dataset
    """
    dataset = datasets.load_dataset(dataset_name)
    dataset = dataset['train']
    text_data = []
    file_count = 0
    if not os.path.isdir(f'./{dataset_name}'):
        os.mkdir(f'./{dataset_name}')
        
    for sample in (dataset):
        sample = sample['text'].replace('/n', '/s')
        text_data.append(sample)
        if len(text_data) == 5000:
            with open(f'./{dataset_name}/text_{file_count}.txt', 'w', encoding="utf-8") as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
    
    with open(f'./{dataset_name}/text_{file_count}.txt', 'w', encoding="utf-8") as fp:
        fp.write('\n'.join(text_data))
    


    
def trainTokenizer(dataset_name: str):
    """train the BERT tokenizer

    Args:
        dataset_name (str): name of dataset
    """
    if not os.path.isdir(f'./{dataset_name}'):
        print('generating dataset...')
        generateDataset(dataset_name)
    paths = [str(x) for x in Path(f'./{dataset_name}').glob('**/*.txt')]
    
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False
        )   
    tokenizer.train(files = paths, vocab_size = 30000, min_frequency = 2,
                    limit_alphabet = 1000, wordpieces_prefix = '##',
                    special_tokens = ['[PAD', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    
    if not os.path.isdir(f'./{dataset_name}_bert'):
        os.mkdir(f'./{dataset_name}_bert')
        tokenizer.save_model(f'./{dataset_name}_bert', 'bert')
        
    

def encodeString(string: Union(str,tuple(str,str)), tokenizer: BertWordPieceTokenizer):
    """Function to encode string or strings for BERT model training
    first tokenize the entire string/strings, then go through and mutate
    each non special token with chance 15%
    
    mutation is either a mask: 80%
                       a random token: 10%
                       no change: 10%
    
    We save which indices mutations are made for training.  
    

    Args:
        string (Union(str,tuple(str,str))): a string or tuple of strings(next sentence prediction)
        tokenizer (BertWordPieceTokenizer): tokenizer to user

    Returns:
        (dict,list): dic corresponds to {input_keys : list(int), token_type_ids : list(int),
                                         attention_mask : list(int), modified_tokens : list(int)}
                     list corresponds to the indices of mutated tokens for use as targets
    """
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]") #also final special token so anything above is fair game for replacement/mask
    vocab_length = 30000
    d = tokenizer(string)
    modified_tokens = [ 0 for i in range (len(d['input_ids']))] 
    for i,token in enumerate(d['input_ids']):
        if token > mask_id:
            if random.randint(1,15) != 1: #wrong
                continue
            modified_tokens[i] = 1
            choice = random.rand(1,5)
            if choice < 4:
                d['input_ids'] = mask_id
            elif choice == 4:
                d['input_ids'] = random.randint(mask_id+1, vocab_length)
                
    d['modified_tokens'] = modified_tokens
    return d
        

    
class _fileData(Dataset):
    """Class for a single files data"""
    def __init__(self, dataset_path, tokenizer_path):
        self.dataset_path = dataset_path
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
    def __getitem__(self,idx: int):
        """gets a training example which roughly corresponds to the given idx

        Args:
            idx (int): idx to look at

        Returns:
            dict: a dict containing relavent training info
        """
        with open(self.path, 'r', encoding="utf-8") as f:
            line = list(f)[idx].split()
            length = len(line)
            l1, l2 = line[:length//2], line[length//2:]
            
            fake = random.randint(0,1)
            if fake:
                fake_l = list(f)[idx].split()
                l2 = fake_l[:len(fake_l)//2]
                
            
            s1 = ''.join(l1)
            s2 = ''.join(l2)
            
            dic = encodeString((s1,s2), self.tokenizer)
            dic['fake'] = bool(fake)
            
            
            return dic 
                
            
            
            
            
            
            
            
# with open('./civil_comments/text_0.txt', 'r', encoding="utf-8") as file:
#     tokenizer = BertTokenizer.from_pretrained('./civil_comments_bert/bert-vocab.txt')
#     i = 0
#     line = list(file)[4]
#     # print(tokenizer.)
#     tokens = tokenizer(['hey nice to meet you', 'i want to have dinner'])['input_ids']
#     tokenizer
#     print(tokens)
#     print(tokenizer.batch_decode(tokens))
 