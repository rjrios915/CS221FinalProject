import os
import nltk
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import re
import pickle

# Make sure you download NLTK punkt tokenizer
nltk.download('punkt_tab')

DEBUG = True
MODE ="trin"

DATA_DIR = "/Users/ricky/Desktop/CS221/Dataset/captions.txt"
IMG_DIR = "/Users/ricky/Desktop/CS221/Dataset/Images"
WORKING_DIR = "/Users/ricky/Desktop/CS221/CS221FinalProject"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
UNITS = 256
LEARNING_RATE = 1e-3
START = '<START> '
END = ' <END>'

def preprocess(caption):
    caption = caption.lower()
    caption = re.sub(r'[^a-z]', ' ', caption)
    caption = re.sub(r'\s+', ' ', caption)
    caption =  START + " ".join([word for word in caption.split() if len(word) > 1]) + END
    return caption

def extract_features(model, mapping, features):
    batch_images = []
    batch_names = []
    for img_name in tqdm(mapping.keys()):
        img_path = IMG_DIR + '/' + img_name
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        batch_images.append(image)
        batch_names.append(img_name)

        if len(batch_images) == BATCH_SIZE:
            batch_images_np = np.array(batch_images)
            features_batch = model.predict(batch_images_np, verbose=0)
            for name, feat in zip(batch_names, features_batch):
                features[name] = feat
            batch_images = []
            batch_names = []

    # Process the remainder
    if batch_images:
        batch_images_np = np.array(batch_images)
        features_batch = model.predict(batch_images_np, verbose=0)
        for name, feat in zip(batch_names, features_batch):
            features[name] = feat
    
    with open("features.pkl", "wb") as f:
        pickle.dump(features, f)

def data_gen(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = [] , [] ,[] 
    n = 0
    while True:
        for key in data_keys:
            captions = mapping[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                if len(seq) < 2:  # Skip sequences that are too short to create (input, output) pairs
                    continue
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[key])
                    X2.append(in_seq)
                    y.append(out_seq)
            n += 1
            if n == batch_size:
                yield {"image": np.array(X1), "text": np.array(X2)}, np.array(y)
                X1.clear() 
                X2.clear()  
                y.clear()   
                n = 0

def convert_to_word(number, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == number:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = "<START>"
    image = np.expand_dims(image, axis=0)
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        y_pred = model.predict([image, sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        word = convert_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "<END>":
            break
    return in_text

def main():
    if MODE == "train":

        # Feature extractor
        model=EfficientNetB0()
        model=Model(inputs=model.inputs, outputs=model.layers[-2].output)

        features={}
        with open(DATA_DIR) as File:
            next(File)
            captions_file=File.read()

        mapping = {}
        all_captions = []
        for line in tqdm(captions_file.split('\n')):
            inx = line.find(",") 
            img_name, caption = line[:inx], line[inx + 1:]
            if len(img_name) < 2 or len(caption) < 2:
                continue
            caption = preprocess(caption)
            if img_name not in mapping:
                mapping[img_name] = []
            mapping[img_name].append(caption)
            all_captions.append(caption)
        with open("mapping.pkl", "wb") as f:
            pickle.dump(mapping, f)

        features={}
        if DEBUG: print("DEBUG: Beginning feature extraction")
        if "features.pkl" in os.listdir("/Users/ricky/Desktop/CS221/CS221FinalProject"):
            with open("features.pkl", "rb") as f:
                features = pickle.load(f)
        else:
            extract_features(model, mapping, features)
        if DEBUG: print("DEBUG: Completed feature extraction")
        
        if DEBUG: print("DEBUG: Beginning tokenizing")
        tokenizer = Tokenizer(
            num_words=10000,
            oov_token="<UNK>",
            lower=False,
            filters=''
        )   
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1
        max_length = max(len(caption.split()) for caption in all_captions)
        with open("tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        if DEBUG: print("DEBUG: Completed tokenization")

        train, test = train_test_split(list(mapping.keys()), test_size=0.2, random_state=42)

        inputs1 = Input(shape=(1280,), name='image')
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        inputs2 = Input(shape=(max_length,), name='text')
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256, return_sequences=False)(se2)

        decoder1 = add([fe2, se3]) 
        decoder2 = Dense(256, activation='relu')(decoder1) 
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())


        earlystop = EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True)
        checkpoint = ModelCheckpoint(WORKING_DIR + '/best_model.keras',
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        callbacks = [earlystop, checkpoint]

        decoder_lstm = LSTM(UNITS, return_sequences=True, return_state=True)
        steps = len(train) // BATCH_SIZE

        for i in range(EPOCHS):
            print(f"Epoch {i+1}/{EPOCHS}")
            generator = data_gen(train, mapping, features, tokenizer, max_length, vocab_size, BATCH_SIZE)
            model.fit(generator,
                    epochs=1,
                    steps_per_epoch=steps,
                    verbose=1,
                    callbacks=callbacks)

        model.save("caption_model.keras")
    else:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open("features.pkl", "rb") as f:
            features = pickle.load(f)
        with open("mapping.pkl", "rb") as f:
            mapping = pickle.load(f)
        img_name = "truck.jpg"
        # img_name = "69189650_6687da7280.jpg"
        img_path = IMG_DIR + "/" + img_name
        if img_name not in features:
            temp_model= EfficientNetB0()
            temp_model = Model(inputs=temp_model.inputs, outputs=temp_model.layers[-2].output)
        
            image = load_img(img_path,target_size=(224,224))
            image = img_to_array(image)
            image = preprocess_input(image)
            feat = temp_model.predict(np.array([image]),verbose=0)
            features[img_name] = feat[0]

        max_length = 35
        model = load_model("caption_model.keras")

        # predict the caption
        y_pred = predict_caption(model, features[img_name], tokenizer, max_length)
        print(y_pred)

        image = Image.open(img_path)
        plt.imshow(image)
        plt.show()

        actual, predicted = [] , []

    train, test = train_test_split(list(mapping.keys()), test_size=0.2, random_state=42)

    for key in tqdm(train):

        captions = mapping[key] 
        # predict the caption for image
        y_pred = predict_caption(model, features[key], tokenizer, max_length)
        # split into words
        actual_captions = [caption.split() for caption in captions]
        y_pred = y_pred.split()
        # append to the list
        actual.append(actual_captions)
        predicted.append(y_pred)

        # calcuate BLEU score
    #Unigram
    print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    #Bigram
    print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

if __name__ == "__main__":
    main()
