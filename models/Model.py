import json
import torch
import time
import re

import torch.nn.functional as F
import numpy as np

from transformers import BertTokenizer
from .TweetClassifier import TweetClassifier
from .TweetDataset import TweetDataset
from torch.utils.data import Dataset, DataLoader

with open('config.json') as jsonFile:
    config = json.load(jsonFile)

class Model:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        dataset = TweetDataset(
            np.array(tweet)
        )

        dataloader = DataLoader(
            dataset,
            batch_size = 100
        )

        data = next(iter(dataloader))

        input_ids = data['input_ids'].to(self.device)
        attention_mask = data['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            probabilities = F.softmax(output, dim = 1)
            _, preds = torch.max(output, dim = 1)
        
        confidence, predicted_class = torch.max(probabilities, dim = 1)
        confidence = confidence.numpy().tolist()
        predicted_class = predicted_class.numpy().tolist()
        probabilities = probabilities.cpu().numpy().tolist()

        count_positive = predicted_class.count(2)
        count_neutral = predicted_class.count(1)
        count_negative = predicted_class.count(0)

        res = []

        for i in range(len(tweet)):
            item = {
                "tweet_text": tweet[i],
                "tweet_sentiment": config["CLASSES"][predicted_class[i]],
                "tweet_confidence": confidence[i],
                "tweet_scores": dict(
                    zip(
                        config["CLASSES"], probabilities[i]
                    )
                )
            }

            res.append(item)

        res.sort(key = lambda t: t['tweet_scores']['Positivo'], reverse = True)
        positive_tweets = res[0:4]

        res.sort(key = lambda t: t['tweet_scores']['Negativo'], reverse = True)
        negative_tweets = res[0:4]

        # confidence, predicted_class = torch.max(probabilities, dim = 1)
        # predicted_class = predicted_class.cpu().item()
        # probabilities = probabilities.flatten().cpu().numpy().tolist()

        # return (
        #     config["CLASSES"][predicted_class],
        #     confidence,
        #     dict(
        #         zip(
        #             config["CLASSES"], probabilities
        #         )
        #     )
        # )

        return {
            "timestamp": int(time.time() * 1000),
            "total": len(tweet),
            "positive": count_positive,
            "neutral": count_neutral,
            "negative": count_negative,
            "positive_tweets": positive_tweets,
            "negative_tweets": negative_tweets
        }

model = Model()
# print(model.predict(["dia", "A Shein atrasou a entrega dos meus produtos, que decepção!"]))
def getModel():
    return model