###importing files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import nltk
import string
import sklearn as sk
import sqlite3

###connecting sqlite data
file = sqlite3.connect('')

#removing data with score 3(to simplify the prediction)
file = pd.read_sql_query("""select * from Reviews where Score != 3""", file)

##converting score to polarity preferences
def conv(x):
  if x<3:
    return 'negative'
  else:
    return 'positive'

score = file['Score']
solution = score.map(conv)
file['Score'] = solution


##removing reviews containing total reviews less then positive reviews
file = file[file.HelpfulnessNumerator<=file.HelpfulnessDenominator]

##dropping duplicates w.r.t productid and timestamp
file = file.drop_duplicates(subset = {'ProductId', 'TimeStamp'}, keep = 'first', inplace = False)


##sorting according to product id
file = file.sort_values('Product_Id', axis=0, ascending= True)

from sklearn.feature_extraction.text import CountVectorizer

# Assuming 'file' is a DataFrame with a 'Text' column containing the text documents

# Initialize the CountVectorizer
countv = CountVectorizer()

# Fit and transform the documents
file_bow = countv.fit_transform(file['Text'].values)

# 'file_bow' now contains the Bag of Words representation of the documents
# It's a sparse matrix where each row corresponds to a document and each column to a word in the vocabulary


# Assuming tf_idf_vect is a TfidfVectorizer object and file is a list of sentences/documents
#tfidf
# Get the feature names from the TF-IDF vectorizer
get_weight = tf_idf_vect.get_feature_names()

#avg tf-idf
# Initialize an empty list to store sentence vectors
sentences = []
row = 0

# Iterate through each sentence in the 'file'
for sentence in file:
    # Initialize a default vector of 50 dimensions
    sent_vec = np.zeros(50)
    # Initialize a variable to store the sum of TF-IDF weights
    weight_sum = 0
    
    # Iterate through each word in the sentence
    for word in sentence:
        try:
            # Get Word2Vec vector for the current word
            vec = w2v_model.wv[word]
            # Get TF-IDF weight for the current word in the current document (sentence)
            tfidf = file[row, get_weight.index(word)]
            # Update the sentence vector by adding the product of Word2Vec vector and TF-IDF weight
            sent_vec += vec * tfidf
            # Update the sum of TF-IDF weights
            weight_sum += tfidf
        except Exception as e:
            # Handle exceptions like word not found in Word2Vec model or TF-IDF matrix
            pass
    
    # Normalize the sentence vector by dividing by the sum of TF-IDF weights
    if weight_sum != 0:
        sent_vec /= weight_sum
    
    # Append the normalized sentence vector to the list of sentences
    sentences.append(sent_vec)
    
# Move to the next row (sentence/document)
row += 1

#avg w2v
import numpy as np

Sent = []  # List to store sentence embeddings

for sentence in file:
    sent_v = np.zeros(50)  # Initialize a default vector of 50 dimensions
    count_of_words = 0
    
    for word in sentence:
        try:
            vec = w2v_model.wv[word]
            sent_v += vec
            count_of_words += 1
        except:
            # Handle the case where the word is not in the vocabulary of the Word2Vec model
            continue
    
    if count_of_words > 0:
        sent_v /= count_of_words  # Calculate the average Word2Vec vector
    
    Sent.append(sent_v)  # Append the sentence vector to the list

# Now 'Sent' contains the average Word2Vec vectors for each sentence in 'file'

#text preprocess
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

stop = set(stopwords.words('english'))  # Set of English stopwords
sno = SnowballStemmer('english')  # Snowball stemmer for English

all_positive_words = []  # List to store all positive words
all_negative_words = []  # List to store all negative words
final_string = []  # List to store final cleaned strings

for sent in final['Text'].values:
    filtered_sentence = []
    
    # Clean HTML tags (assuming you have defined cleanhtml function)
    sent = cleanhtml(sent)
    
    # Tokenize and process each word in the sentence
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            # Check if the word is alphabetic and its length is greater than 2
            if cleaned_words.isalpha() and len(cleaned_words) > 2:
                # Check if the word is not in stopwords
                if cleaned_words.lower() not in stop:
                    # Stem the word
                    s = sno.stem(cleaned_words.lower()).encode('utf8')
                    filtered_sentence.append(s)
                    
                    # Classify words based on sentiment
                    if final['Score'].values[i] == 'positive':
                        all_positive_words.append(s)
                    elif final['Score'].values[i] == 'negative':
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    
    # Join the filtered words into a final string
    str1 = b" ".join(filtered_sentence)
    final_string.append(str1)
    
    

#To apply n-grams in TF-IDF and w2v
from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize TfidfVectorizer with n-grams (e.g., ngram_range=(1, 2) for unigrams and bigrams)
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

#word2vec
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import nltk

# Create phrases (4-grams) from sentences
phrases = Phrases(sentences, min_count=1,size=4, threshold=1)

# Apply n-grams to sentences
sentences_with_phrases = list(bigram[sentences])

# Train Word2Vec model
model = Word2Vec(sentences_with_phrases, vector_size=100, window=5, min_count=1, sg=0)


    
    
    


