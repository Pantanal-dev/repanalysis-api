import json
import torch
import re

from torch.utils.data import Dataset
from transformers import BertTokenizer

with open('config.json') as jsonFile:
    config = json.load(jsonFile)

class TweetDataset(Dataset):
    def __init__(self, tweets):
        self.tweets = tweets
        self.tokenizer = BertTokenizer.from_pretrained(config['PRE_TRAINED_MODEL_NAME'])
        self.max_len = config['INPUT_LENGTH']

    def clean_tweet(self, tweet):
        # Converte tudo para minúsculo
        tweet = tweet.lower()

        # Remove menções
        tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)

        # Remove #
        tweet = re.sub(r'#', '', tweet)

        # Remove hyperlinks
        tweet = re.sub(r'https?:\/\/\S+', '', tweet)

        # Remove pontuação
        tweet = re.sub(r'[^\w\s]', '', tweet)

        # Remove \n
        tweet = re.sub(r'\n', '', tweet)

        return tweet

    def __getitem__(self, index):
        tweet = self.tweets[index]

        token = self.tokenizer(
            self.clean_tweet(tweet),
            max_length = self.max_len,
            add_special_tokens = True,
            return_token_type_ids = False,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        )

        return {
            'tweet_text' : tweet,
            'input_ids': token['input_ids'].flatten(),
            'attention_mask': token['attention_mask'].flatten(),
        }
    
    def __len__(self):
        return len(self.tweets)