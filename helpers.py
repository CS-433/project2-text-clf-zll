#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import pickle
import csv
from nltk.tokenize import word_tokenize


# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    randomlist = random.sample(range(1,len(data)),n)
    return data[randomlist], labels[randomlist]

def text_to_index_array(p_new_dic, tweets_list): 
    '''
    Mapping text data to index matrix
    '''
    new_tweets = []
    for tweet in tweets_list:
        new_tweet = []
        temp = tweet.replace("<user>", "").replace("\n", "").replace("<url>", "").split()
        for word in temp:
            try:
                new_tweet.append(p_new_dic[word]) 
            except:
                new_tweet.append(0)  # Set to 0 if not present in the vocabulary
        new_tweets.append(new_tweet)
    return np.array(new_tweets,dtype=object)   

def text_cut_to_same_long(tweets_list,maxlen):
    '''
    Cut the data to the same specified length  
    '''
    data_num = len(tweets_list)
    new_ = np.zeros((data_num,maxlen)) 
    se = []
    for i in range(len(tweets_list)):
        new_[i,:] = tweets_list[i,:maxlen]        
    new_ = np.array(new_, dtype=object)
    return new_
    
def creat_wordvec_tensor(embedding_weights,X_T,vocab_dim,maxlen):
    '''
    Map the index matrix into a word vector matrix
    '''
    X_tt = np.zeros((len(X_T),maxlen,vocab_dim))
    num1 = 0
    num2 = 0
    for j in X_T:
        for i in j:
            X_tt[num1,num2,:] = embedding_weights[int(i),:]
            num2 = num2+1
        num1 = num1+1
        num2 = 0
    return X_tt

def creat_wordvec_mean_tensor(embedding_weights,X_T):
    '''
    Map the index matrix into a mean word vector matrix
    '''
    X_tt = np.zeros((len(X_T),vocab_dim))
    num1 = 0
    num2 = 0
    for j in X_T:
        temp = np.zeros((vocab_dim,))
        for i in j:
            temp += embedding_weights[int(i),:]
            num2 = num2+1
        if num2 == 0:
            X_tt[num1,:] = temp
        else:
            X_tt[num1,:] = temp/num2
        num1 = num1+1
        num2 = 0
    return X_tt
        
def clean_words(content,doc_bool):
    '''
    Drop stop words form tweets when model is not doc2vec
    Case-folding
    Remove len < 3 if alpha word
    :return: a clean word list
    '''
    if doc_bool == True:
        clean = [word.lower() for word in content if not (word.isalpha() and len(word) <= 2)]
        return ' '.join(clean)
    else:
        stops = set(['the', '<url>', '<user>'])
        clean = [word.lower() for word in content if not (word.isalpha() and len(word) <= 2) and not word in stops]
        return ' '.join(clean)


def cut_words(content):
    '''
    Tokenize the tweet to words
    :return: a cut word list
    '''
    if content != '' and content is not None:
        seg_list = word_tokenize(content)
        each_split = ' '.join(seg_list).split(' ')
    return each_split

def get_data_df(file_list):
    '''
    Saving all data into one dataframe
    '''
    data = []
    with open(file_list[0]) as f:
        for line in f:
            doc = line.replace("\t", "").replace("<user>", "").replace("\n", "").replace("<url>", "")
            data.append([doc,1])
    with open(file_list[1]) as f:
        for line in f:
            doc = line.replace("\t", "").replace("<user>", "").replace("\n", "").replace("<url>", "")
            data.append([doc,0])

    df = pd.DataFrame (data, columns = ['tweet', 'label'])
    
    return df

def create_csv_submission(pre, name):
    '''
    Creates an output file in .csv format for submission to AIcrowd
    
    PARAMETERS:
    pre: predicted class labels)
    name: string name of .csv output file to be created)

    '''
    pre_list = pre.numpy().tolist ()
    idx = [i for i in range(1,len(pre)+1)]
    dict_ = {
        "Id" : idx,
        "Prediction" : pre_list
    }
    pred_df = pd.DataFrame(dict_)
    pred_df['Prediction'][pred_df['Prediction'] == 0] = -1
    pred_df.to_csv("pred.csv",index = False)


# FOR TensorBoard helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        





