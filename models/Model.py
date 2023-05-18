import json
import torch

import torch.nn.functional as F

from transformers import BertTokenizer
from .TweetClassifier import TweetClassifier

with open('config.json') as jsonFile:
    config = json.load(jsonFile)

class Model:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = BertTokenizer.from_pretrained(config['PRE_TRAINED_MODEL_NAME'])

        model = TweetClassifier()
        model.load_state_dict(
            torch.load(
                config['TRAINED_MODEL'],
                map_location = self.device
            )
        )

        model = model.eval()
        self.model = model.to(self.device)

    def predict(self, tweet):
        encoded_tweet = self.tokenizer(
            tweet,
            max_length = config['INPUT_LENGTH'],
            add_special_tokens = True,
            return_token_type_ids = False,
            padding = 'max_length',
            return_attention_mask = True,
            return_tensors = 'pt',
        )

        input_ids = encoded_tweet['input_ids'].to(self.device)
        attention_mask = encoded_tweet['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            probabilities = F.softmax(output, dim = 1)

        confidence, predicted_class = torch.max(probabilities, dim = 1)
        predicted_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        return (
            config["CLASSES"][predicted_class],
            confidence,
            dict(
                zip(
                    config["CLASSES"], probabilities
                )
            )
        )