import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

## Load in via Pandas
df = pd.read_csv('movie_data.csv', encoding='utf-8')

## CV using SWR, Unigram and 5000 as max features
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)

## Apply LDA Clustering Technique with 15 topics
lda = LatentDirichletAllocation(n_topics=15, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)

## Investigate results by printing 5 top words from each topic
n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))