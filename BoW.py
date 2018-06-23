import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

def cleaner(text):
    text = re.sub('<[^>]*>', '', text) # Remove HTML elements
    emoji = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)'), text) # Store Emojis
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emoji).replace('-', '') # Remove non-words, make lowercase and readd emoji
    return text

df['review'] = df['review'].apply(cleaner)