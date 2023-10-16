import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertModel
import torch

import nltk
nltk.download('punkt')
nltk.download('stopwords')
def preprocess(text):
    """Preprocesses the given text by lowercasing, removing punctuation, and removing stopwords."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    return ' '.join(tokens)

def get_embedding(title, tokenizer, model):
    """Generates BERT embeddings for the given title."""
    inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][0].mean(0).numpy()

def main():
    df = pd.read_csv('photo_metadata.csv')
    sampled_df = df.sample(n=100000, random_state=42)
    print("Sampling complete.")
    sampled_df['title'] = sampled_df['title'].fillna('')
    sampled_df['title'] = sampled_df['title'].apply(preprocess)
    sampled_df = sampled_df[sampled_df['title'] != '']
    print("Preprocessing complete.")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    sampled_df['embeddings'] = sampled_df['title'].apply(lambda title: get_embedding(title, tokenizer, model))

    sampled_df[['id', 'embeddings']].to_csv('embeddings.csv', index=False)
    print("Embeddings generated and saved.")
if __name__ == "__main__":
    main()