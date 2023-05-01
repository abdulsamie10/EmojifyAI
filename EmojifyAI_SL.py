import re # Regular expression library
import os
import numpy as np
import pandas as pd
import torch # PyTorch library
from transformers import AutoTokenizer, AutoModel # Hugging Face Transformers library
from sklearn.metrics.pairwise import cosine_similarity # Cosine similarity function
from nltk.corpus import stopwords # Stopwords from the NLTK library
from nltk.tokenize import word_tokenize # Tokenization function from the NLTK library
import streamlit as st # Streamlit library for web applications
import nltk

nltk.download('stopwords') # Download the stopwords dataset
nltk.download('punkt') # Download the tokenizer dataset

class EmojifyAI:
    def __init__(self):
        self.obtainModel() # Initialize the model
    
    def obtainModel(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')  # Load the tokenizer
        self.model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')  # Load the model

    def computeMeanTokensForSentence(self, text):
        text = text.lower()  # Convert text to lowercase
        text = re.sub('[^a-z]+', ' ', text)  # Remove non-alphabetic characters
        stop_words = set(stopwords.words('english'))  # Set of English stopwords
        word_tokens = word_tokenize(text)  # Tokenize the text
        text = [w for w in word_tokens if not w.lower() in stop_words]  # Remove stopwords
        text = ' '.join(text)  # Join the tokens back into a single string
        return self.computeMeanTokens([text])  # Compute the mean tokens

    def computeMeanTokens(self, text):
        self.fetchTokens(text)  # Fetch the tokens
        self.obtainEmbedding()  # Obtain the embeddings
        return self.calculateMeanValue()  # Calculate the mean value

    def fetchTokens(self, sentences):
        self.tokens = {'input_ids': [], 'attention_mask': []}  # Initialize token dictionary

        for text in sentences:
            new_tokens = self.tokenizer.encode_plus(text, max_length=128,
                                            truncation=True, padding='max_length',
                                            return_tensors='pt')  # Tokenize the text
            self.tokens['input_ids'].append(new_tokens['input_ids'][0])  # Append input IDs
            self.tokens['attention_mask'].append(new_tokens['attention_mask'][0])  # Append attention masks

        self.tokens['input_ids'] = torch.stack(self.tokens['input_ids'])  # Stack the input IDs
        self.tokens['attention_mask'] = torch.stack(self.tokens['attention_mask'])  # Stack the attention masks

    def obtainEmbedding(self):
        outputs = self.model(**self.tokens)  # Get model outputs
        self.embeddings = outputs.last_hidden_state  # Obtain embeddings

    def calculateMeanValue(self):
        attention_mask = self.tokens['attention_mask']  # Get attention masks
        mask = attention_mask.unsqueeze(-1).expand(self.embeddings.size()).float()  # Expand the mask
        masked_embeddings = self.embeddings * mask  # Apply the mask to the embeddings
        summed = torch.sum(masked_embeddings, 1)  # Sum the masked embeddings
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)  # Clamp the summed mask
        self.mean_pooled = summed / summed_mask  # Compute mean-pooled embeddings
        self.mean_pooled = self.mean_pooled.detach().numpy() # Detach the mean-pooled embeddings and convert to NumPy array
        return self.mean_pooled # Return the mean-pooled embeddings
    
    def computeSimilarity(self, sentence_tokens, mean_tokens):
        similarity = cosine_similarity([sentence_tokens], mean_tokens)  # Calculate cosine similarity
        return similarity

    def recommendEmoji(self, sentence):
        all_tokens = torch.load('checkpoint/token-all.pt')  # Load precomputed tokens
        sentence_token = self.computeMeanTokensForSentence(sentence)  # Compute tokens for input sentence
        similarity = self.computeSimilarity(sentence_token[0], all_tokens)  # Compute similarity scores
        indices = (-similarity[0]).argsort()[:5]  # Get indices of top 5 similar tokens
        emoji_df = pd.read_csv("data/emoji-data.csv")  # Load emoji data
        emoji_list = { 'emoji':[], 'description':[] }  # Initialize emoji list dictionary
        for j in indices:
            emoji_list['emoji'].append(emoji_df['emoji'][j])  # Append emoji to the list
            emoji_list['description'].append(emoji_df['description'][j])  # Append emoji description to the list
        return emoji_list  # Return the emoji list


obj = EmojifyAI() # Instantiate the EmojifyAI class

st.title("EmojifyAI v1.0") # Set the title of the web app
st.subheader("Write a text") # Set a subheader
text = st.text_input('Enter text') # Create a text input field
st.subheader("Predicted Emojis") # Set a subheader

if len(text) > 0: # If there is text entered in the input field
    emoji_list = obj.recommendEmoji(text) # Get recommended emojis
    output = ' ' # Initialize the output string

    for i in range(len(emoji_list['emoji'])):  # Iterate through the recommended emojis
        output += emoji_list['emoji'][i]  # Append each emoji to the output string
    st.write(output)  # Display the output string

    st.subheader("Developer mode:")  # Set a subheader for developer mode
    st.write("Predicted emoticon list")  # Display a description
    st.write(emoji_list)  # Display the emoji list