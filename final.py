import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import nltk
import string
import numpy as np

# Make sure you download NLTK punkt tokenizer
nltk.download('punkt_tab')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# ----------------- Vocabulary -----------------
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def tokenizer(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return nltk.tokenize.word_tokenize(text)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)
            for token in tokens:
                if frequencies[token] == self.freq_threshold:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized]

# ----------------- Dataset -----------------
class FlickrDataset(Dataset):
    def __init__(self, dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = dir
        self.transform = transform
        with open(caption_file, 'r') as f:
            lines = f.readlines()

        self.imgs = []
        self.captions = []
        for line in lines:
            indx = line.find(',')
            img, caption = line[:indx], line[indx+1:-2]
            self.imgs.append(img)
            self.captions.append(caption.strip())
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        numericalized = [self.vocab.stoi["<START>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized.append(self.vocab.stoi["<END>"])
        return image, torch.tensor(numericalized)

# ----------------- Collate Function -----------------
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        images = torch.stack(images)
        captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions

def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = FlickrDataset(root_dir="/Users/ricky/Desktop/CS221/Dataset/Images", caption_file="/Users/ricky/Desktop/CS221/Dataset/captions.txt", transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=Collate(pad_idx))

if __name__ == "__main__":
    train()
