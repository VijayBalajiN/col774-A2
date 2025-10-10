from naive_bayes import NaiveBayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix


#tokenizer functions starting with simple_tokenizer
def simple_tokenizer(text):
    return text.split()

#tokenizer with stopword removal and stemming
def stopword_stem_tokenizer(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = text.split()
    final_words = [ps.stem(word) for word in words if word.lower() not in stop_words]
    return final_words

def tokenize_bigrams(text):
    preprocessed_text = stopword_stem_tokenizer(text)
    bigrams = [f"{preprocessed_text[i]}_{preprocessed_text[i+1]}" for i in range(len(preprocessed_text)-1)]
    return bigrams

def combine_unigrams_bigrams(text):
    unigrams = stopword_stem_tokenizer(text)
    bigrams = tokenize_bigrams(text)
    return unigrams + bigrams


def train_and_evaluate(train_df, test_df, input_columns, tokenizer, smoothening=1, show_error_analysis=False):
    model = NaiveBayes() 
    print("Training the model...")
    train_df["Combined text"] = train_df[input_columns].astype(str).agg(' '.join, axis=1)
    test_df["Combined text"] = test_df[input_columns].astype(str).agg(' '.join, axis=1)
    train_df["Tokenized Description"] = train_df["Combined text"].apply(tokenizer)
    test_df["Tokenized Description"] = test_df["Combined text"].apply(tokenizer)
    model.fit(train_df, smoothening, class_col="label", text_col="Tokenized Description")
    print("Evaluating the model...")
    train_df = model.predict(train_df)
    test_df = model.predict(test_df)
    print("Evaluation complete.")
    train_accuracy = (train_df["label"] == train_df['Predicted']).mean()
    test_accuracy = (test_df["label"] == test_df['Predicted']).mean()
    
    if not show_error_analysis:
        return train_accuracy, test_accuracy
    # Error analysis
    #analyse using accuracy, precision, recall, f1-score
    print(f"Error Analysis for {tokenizer.__name__}, input columns: {input_columns}")
    print("Train Classification Report:")
    print(classification_report(train_df["label"], train_df['Predicted']))
    print("Test Classification Report:")
    print(classification_report(test_df["label"], test_df['Predicted']))
    return train_accuracy, test_accuracy

def part1a(train_df, test_df):  #for now just load the data and call train_and_evaluate
    # train_df = pd.read_csv("data/train.csv")
    # test_df = pd.read_csv("data/test.csv")
    smoothening = 1
    train_accuracy, test_accuracy = train_and_evaluate(train_df, test_df,["content"], simple_tokenizer, smoothening)
    print(f"Train Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    
def part1b(train_df, test_df):
    # train_df = pd.read_csv("data/train.csv")
    # test_df = pd.read_csv("data/test.csv")
    smoothening = 1
    #construct a word cloud of top words in each class
    train_df["Combined text"] = train_df[["content"]].astype(str).agg(' '.join, axis=1)
    train_df["Tokenized Description"] = train_df["Combined text"].apply(simple_tokenizer)
    model = NaiveBayes()
    model.fit(train_df, smoothening, class_col="label", text_col="Tokenized Description")
    
    for c in range(model.num_classes):
        word_freq = {word: np.exp(log_prob) for word, log_prob in model.phi_j_y[c].items() if word != '<UNK>'}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Class {c}")
        # plt.show()
        #create the directory output if it does not exist
        if not os.path.exists("output"):
            os.makedirs("output")
        if not os.path.exists("output/part1b"):
            os.makedirs("output/part1b")
        plt.savefig(f"output/part1b/wordcloud_class_{c}.png")
        plt.close()
        
def part2a_2c(train_df, test_df):
    # to perform stemming and stopword removal and validate
    
    # train_df["content"].apply(stopword_stem_tokenizer)
    # test_df["content"].apply(stopword_stem_tokenizer)
    # print("hello1")
    return train_and_evaluate(train_df, test_df, ["content"], stopword_stem_tokenizer, smoothening=1)

def part2b(train_df, test_df):
    #create the word clouds again after removal of stopwords and stemming
    train_df["Combined text"] = train_df[["content"]].astype(str).agg(' '.join, axis=1)
    train_df["Tokenized Description"] = train_df["Combined text"].apply(stopword_stem_tokenizer)
    model = NaiveBayes()
    model.fit(train_df, smoothening=1, class_col="label", text_col="Tokenized Description")
    if not os.path.exists("output"):
            os.makedirs("output")
    if not os.path.exists("output/part2b"):
            os.makedirs("output/part2b")
    for c in range(model.num_classes):
        word_freq = {word: np.exp(log_prob) for word, log_prob in model.phi_j_y[c].items() if word != '<UNK>'}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Word Cloud for Class {c} after Stopword Removal and Stemming")
        plt.savefig(f"output/part2b/wordcloud_class_{c}_stopword_stem.png")
        plt.close()
        
def part3(train_df, test_df):
    #analysis using bigrams and unigrams together
    # train_df["content"].apply(combine_unigrams_bigrams)
    # test_df["content"].apply(combine_unigrams_bigrams)
    return train_and_evaluate(train_df, test_df, ["content"], combine_unigrams_bigrams, smoothening=1)

def part4(train_df, test_df, input_columns=["content"]):
    #compare all the three methods with accuracies and error analysis using accuracy, precision, recall, f1-score
    #model 1: simple_tokenizer
    #model 2: stopword_stem_tokenizer
    #model 3: combine_unigrams_bigrams
    error_analysis = []
    error_analysis.append(train_and_evaluate(train_df, test_df, input_columns, simple_tokenizer, smoothening=1, show_error_analysis=True))
    error_analysis.append(train_and_evaluate(train_df, test_df, input_columns, stopword_stem_tokenizer, smoothening=1, show_error_analysis=True))
    error_analysis.append(train_and_evaluate(train_df, test_df, input_columns, combine_unigrams_bigrams, smoothening=1, show_error_analysis=True))
    return error_analysis
    pass

def part5(train_df, test_df):
    #do the same for the title column
    #model 1: simple_tokenizer
    #model 2: stopword_stem_tokenizer
    #model 3: combine_unigrams_bigrams
    error_analysis = part4(train_df, test_df, input_columns=["title"])
    return error_analysis
    pass

def part6a(train_df, test_df):
    #use both title and content columns
    error_analysis = train_and_evaluate(train_df, test_df, tokenizer=combine_unigrams_bigrams, input_columns=["title", "content"])
    return error_analysis

def part6b(train_df, test_df):
    #use both title and content but with different theta for title and content
    
    pass
        
    
if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    # part1a(train_df, test_df)
    # part1b(train_df, test_df)
    # train_accuracy, test_accuracy = part2a_2c(train_df, test_df)
    # print(f"After stopword removal and stemming, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    # part2b(train_df, test_df)
    # train_accuracy, test_accuracy = part3(train_df, test_df)
    # print(f"Using unigrams and bigrams together, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")
    # part4(train_df, test_df)
    # error_analysis = part5(train_df, test_df)
    # print(error_analysis)
    error_analysis = part6a(train_df, test_df)
    print(error_analysis)
