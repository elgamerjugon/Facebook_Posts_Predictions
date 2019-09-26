#!/usr/bin/env python
# coding: utf-8

# In[275]:


import pandas as pd
import numpy as np
from translate import translate_message
import re
import emoji
import seaborn as sns
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression


# In[81]:


# Reading the dataset where annual posts from 127 newspapers are stored

annual_df = pd.read_csv("annual_posts.csv")

# Replacing all blank spaces in the columns names for unserscores and changing
# them to lower case

cols = [col.lower().replace(" ", "_") for col in annual_df.columns]
annual_df.columns = cols

# Dropping columns that are not useful for the model

annual_df.drop(columns = ["video_share_status", "video_length", "final_link",
                          "sponsor_id", "sponsor_name", "link_text", "likes",
                          "comments", "shares", "love", "wow", "haha", "sad",
                          "angry", "thankful", "page_id"], inplace = True)

# Cleaning the message column to perform translations and
# sentiment analysis score

annual_df.message = annual_df.message.replace(np.nan, "")
annual_df["message"] = annual_df["message"].apply(lambda x: re.sub(r'http.*', '', x))
annual_df["message"] = annual_df["message"].apply(lambda x: re.sub(r"\"", "", x))
annual_df["message"] = annual_df["message"].apply(lambda x: re.sub(r"\'", "", x))
annual_df["message"] = annual_df["message"].apply(lambda x: re.sub(r"[-<>\@\\]", "", x))
annual_df["message"] = annual_df["message"].apply(lambda x: re.sub(r":.*:", "", emoji.demojize(x)))
annual_df = annual_df[annual_df.message.apply(lambda x: len(x)) > 1]
annual_df.reset_index(inplace = True)
annual_df.drop(columns = ["index"], inplace = True)

# Translating the message column from the dataset

# translate_message(annual_df.message)

# The google translation API has a request limit so I just read a dataset where I have already translated the messages

translated_messages = pd.read_csv("messages_translations.csv")

annual_df["translations"] = translated_messages


# In[82]:


analyzer = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence)
    return score


# In[83]:


annual_df = annual_df.iloc[0:2529]


# In[84]:


annual_df["score"] = ""
for i, new in enumerate(annual_df["translations"]):
    dict_score = sentiment_analyzer_scores(new)
    annual_df["score"][i] = dict_score["compound"]


# In[85]:


def sentiment(score):
    if score >= 0.05:
        # Positive sentiment
        return 3
    elif score > -0.05 and score < 0.05:
        # Neutral 
        return 2
    else:
        #Negative
        return 1


# In[86]:


annual_df["sentiment"] = annual_df.score.apply(lambda x: sentiment(x))
annual_df


# In[87]:


ml_df = annual_df.drop(columns = ["created", "url", "message", "link", "description",
                                  "translations", "page_name", "user_name", "score"])
ml_df


# In[88]:


ml_df.overperforming_score = ml_df.overperforming_score.apply(lambda x: x.replace(",", ""))


# In[89]:


ml_df.overperforming_score = pd.to_numeric(ml_df.overperforming_score, downcast = "float")
ml_df.score = pd.to_numeric(ml_df.score, downcast = "float")
ml_df.info()


# In[90]:


ml_df.dropna(inplace = True)


# In[91]:


ml_df.post_views = ml_df.post_views.replace(0, np.nan)
ml_df.total_views = ml_df.total_views.replace(0, np.nan)
ml_df.total_views_for_all_crossposts = ml_df.total_views_for_all_crossposts.replace(0, np.nan)


# In[92]:


post_views = ml_df.post_views.mean(skipna=True)
total_views = ml_df.total_views.mean(skipna=True)
total_views_for_all_crossposts = ml_df.total_views_for_all_crossposts.mean(skipna=True)


# In[93]:


ml_df.post_views = ml_df.post_views.replace(np.nan, post_views)
ml_df.total_views = ml_df.total_views.replace(np.nan, total_views)
ml_df.total_views_for_all_crossposts=ml_df.total_views_for_all_crossposts.replace(np.nan, total_views_for_all_crossposts)


# In[94]:


ml_df = pd.get_dummies(ml_df, drop_first = True)
ml_df.head()


# In[95]:


ml_df = ml_df.drop(columns = ["score"])
sns.heatmap(ml_df.corr())


# In[277]:


X = ml_df.drop(columns = ["overperforming_score"])
y = ml_df.overperforming_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[274]:


regr = RandomForestRegressor(max_depth=2, random_state=0,
                             n_estimators=100)

scores = cross_validate(regr, X, y, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
regr.fit(X_train, y_train)
y_hat_random = regr.predict(X_test)
rmse = (mean_squared_error(y_test, y_hat_random))**(0.5)
np.mean((scores["test_neg_mean_squared_error"]*-1)**0.5)
regr.score(X, y)
# rmse


# In[327]:


bagging_regressor = BaggingRegressor(random_state = 82)
bagging_regressor.fit(X_train, y_train)
y_hat_bagging = bagging_regressor.predict(X_test)
rmse = (mean_squared_error(y_test, y_hat_bagging))**(0.5)
print(rmse)
bagging_regressor.score(X, y)

