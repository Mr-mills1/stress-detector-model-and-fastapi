#!/usr/bin/env python
# coding: utf-8

# # STRESS DETECTION MACHINE<br>LEARNING MODEL

# # Problem statement<br>
# ***Predicting various human stress behaviour patterns***

# In[1]:


# upload the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# load dataset
df = pd.read_csv(r'C:\Users\HP\Desktop\Data Science\Datasets\Stress.csv')


# In[3]:


df


# In[4]:


# check general information of dataset
df.info()


# In[5]:


df.shape


# # Exploratory Data Analysis

# In[6]:


df.describe().T


# In[7]:


# print columns features
df.columns.tolist()


# In[8]:


# print unique features of subreddit
df['subreddit'].unique().tolist()


# In[9]:


# check and visualize the count of values
df['subreddit'].value_counts()


# In[10]:


# plot the value count visuals
plt.figure(figsize=(15,10))
sns.countplot('subreddit', data = df, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[11]:


# value count of label
df['label'].value_counts()


# In[12]:


# plot the value count visuals
plt.figure(figsize=(10,8))
sns.countplot('label', data = df, palette = 'hls')
plt.xticks(rotation = 90)
plt.show()


# In[13]:


# import text cleaning libraries
import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words('english'))


# In[14]:


# define a function to clean text
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["text"] = df["text"].apply(clean)


# In[15]:


# plot a wordcloud for more text visualization
# import the libraries
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(i for i in df.text)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords,
background_color="red").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[16]:


# apply map to categorical variables in label
# let 0 be equal to stress and 1 be unstress
df["label"] = df["label"].map({0: "Stress", 1: "Unstress"})
df = df[["text", "label"]]
print(df.head())


# # Features Selection

# In[ ]:


# transform text to vector
# import the libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle

# choose dependent and independent variable
x = np.array(df["text"])
y = np.array(df["label"])

# transforming text to vectors
cv = CountVectorizer()
X = cv.fit_transform(x)

# save the pickle file
pickle.dump(X, open("Xcv.pkl", "wb"))
# split into test and train set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# # Build the model

# In[24]:


# import the algorithm library
# model selection
#from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score
# using naive bayes 
model = MultinomialNB()
# fit the model to the set
model.fit(x_train, y_train)


# In[25]:


# make y prediction
MNB_pred = model.predict(x_test)


# # Model Evaluation and Validation

# In[26]:


# print classification report
print(classification_report(MNB_pred, y_test))


# In[27]:


# print and plot confusion matrix
cm = confusion_matrix(y_test, MNB_pred)
sns.heatmap(cm, annot=True)


# In[22]:


# check model accuracy
accuracy = accuracy_score(y_test, MNB_pred)
print("accuracy score is", accuracy)


# # Make Predictions with the Model

# In[29]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[30]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[31]:


user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[32]:


# save or dump the model in pickle
# load the library
import pickle
pickle.dump(model, open("stress_model.pkl", "wb"))


# In[ ]:




