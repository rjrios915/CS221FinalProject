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

# import nltk.tokenize.punkt
# print("HERE", nltk.tokenize.punkt.__file__)
# print("HERE", find(f"tokenizers/punkt_tab/{lang}/"))

# Make sure you download NLTK punkt tokenizer
nltk.download('punkt')
# Hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_LEN = 20

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
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        with open(caption_file, 'r') as f:
            lines = f.readlines()

        self.imgs = []
        self.captions = []
        for line in lines:
            indx = line.find(',')
            img, caption = line[:indx], line[indx+1:]
            img = img[:-2]
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
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        images = torch.stack(images)
        captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions

# ----------------- Encoder -----------------
class CNN_Encoder(nn.Module):
    def __init__(self, embed_size):
        super(CNN_Encoder, self).__init__()
        cnn = models.resnet18(weights=None)
        modules = list(cnn.children())[:-1]
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Linear(cnn.fc.in_features, embed_size)

    def forward(self, images):
        features = self.cnn(images).squeeze()
        return self.fc(features)

# ----------------- Decoder -----------------
class RNN_Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(RNN_Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

# ----------------- Training -----------------
def train():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = FlickrDataset(root_dir="/Users/ricky/Desktop/CS221/Dataset/Images", caption_file="/Users/ricky/Desktop/CS221/Dataset/captions.txt", transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=MyCollate(pad_idx))

    encoder = CNN_Encoder(EMBED_SIZE)
    decoder = RNN_Decoder(EMBED_SIZE, HIDDEN_SIZE, len(dataset.vocab.stoi), NUM_LAYERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    decoder.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    params = list(decoder.parameters()) + list(encoder.fc.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        for idx, (imgs, captions) in enumerate(loader):
            imgs, captions = imgs.to(device), captions.to(device)
            features = encoder(imgs)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{idx}/{len(loader)}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
