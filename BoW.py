import re
import numpy as np
import pyprind
import TextProc
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords


stop = stopwords.words('english') # Initialise stopword removal

## Clean text data
def cleaner(text):
    text = re.sub('<[^>]*>', '', text) # Remove HTML elements
    emoji = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) # Store Emojis
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoji).replace('-', '')) # Remove non-words, make lowercase and readd emojitoken  = [w for w in text.split () if w not in stop]
    token  = [w for w in text.split () if w not in stop] # Split into list without stopwords
    return token

## Read in each review and return tuple with string and class label
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # Skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


## Feature vectors for 50000 imput fairly heavy, opt to apply Mini-Batch instead
def get_minibatch(doc_stream, size): # Pass in tokens and batch size
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


## Create psuedo-vectors via HasingVector
## Unable to use CountVectoriser/TfidfVectoriser due to using Minibatch
def tokenizer(text):
    return text.split()


vect = HashingVectorizer(decode_error='ignore', n_features=2**21,
                         preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path = 'movie_data.csv')


## Model Training
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes) # Model training with MBL
    pbar.update() # Update Progress

## Test Accuracy
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))