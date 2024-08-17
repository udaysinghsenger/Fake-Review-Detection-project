#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


# In[2]:


nltk.download('omw-1.4')


# In[3]:


df = pd.read_csv(r'C:\fake reviews dataset.csv')
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df['rating'].value_counts()


# In[8]:


plt.figure(figsize=(15,8))
labels = df['rating'].value_counts().keys()
values = df['rating'].value_counts().values
explode = (0.1,0,0,0,0)
plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')
plt.title('Proportion of each rating',fontweight='bold',fontsize=25,pad=20,color='crimson')
plt.show()


# In[9]:


def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])


# In[10]:


df['text_'][0], clean_text(df['text_'][0])


# In[11]:


df['text_'].head().apply(clean_text)


# In[12]:


df.shape


# In[ ]:





# In[13]:


df['text_'] = df['text_'].astype(str)


# In[14]:


def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])


# In[15]:


preprocess(df['text_'][4])


# In[16]:


df['text_'][:10000] = df['text_'][:10000].apply(preprocess)


# In[17]:


df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)


# In[18]:


df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)


# In[19]:


df['text_'][30001:40000] = df['text_'][30001:40000].apply(preprocess)


# In[20]:


df['text_'][40001:40432] = df['text_'][40001:40432].apply(preprocess)


# In[21]:


df['text_'] = df['text_'].str.lower()


# In[22]:


stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])
df['text_'] = df['text_'].apply(lambda x: stem_words(x))


# In[23]:


lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))


# In[24]:


df['text_'].head()


# In[ ]:





# In[25]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings, string
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[26]:


df = pd.read_csv(r'C:\Preprocessed Fake Reviews Detection Dataset.csv')
df.head()


# In[27]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[28]:


df.head()


# In[29]:


df.dropna(inplace=True)


# In[30]:


df['length'] = df['text_'].apply(len)


# In[31]:


df.info()


# In[32]:


import matplotlib.pyplot as plt

# Plotting a line plot for the 'length' column of the DataFrame
plt.plot(df['length'], color='blue')
plt.xlabel('Index')
plt.ylabel('Length')
plt.title('Length Variation')
plt.grid(True)
plt.show()


# In[33]:


import matplotlib.pyplot as plt

# Creating a box plot for the 'length' column of the DataFrame
plt.boxplot(df['length'])
plt.ylabel('Length')
plt.title('Box Plot of Length')
plt.grid(True)
plt.show()


# In[34]:


df.groupby('label').describe()


# In[35]:


import matplotlib.pyplot as plt

# Creating histograms for the 'length' column grouped by 'label' with customizations
df.hist(column='length', by='label', bins=30, color='green', figsize=(10, 6), grid=False, alpha=0.7)
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.suptitle('Distribution of Length by Label')
plt.show()


# In[36]:


df[df['label']=='OR'][['text_','length']].sort_values(by='length',ascending=False).head().iloc[0].text_


# In[37]:


df.length.describe()


# In[38]:


def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[39]:


bow_transformer = CountVectorizer(analyzer=text_process)
bow_transformer


# In[40]:


bow_transformer.fit(df['text_'])
print("Total Vocabulary:",len(bow_transformer.vocabulary_))


# In[41]:


review4 = df['text_'][3]
review4


# In[42]:


bow_msg4 = bow_transformer.transform([review4])
print(bow_msg4)
print(bow_msg4.shape)


# In[ ]:





# In[43]:


bow_reviews = bow_transformer.transform(df['text_'])


# In[44]:


print("Shape of Bag of Words Transformer for the entire reviews corpus:",bow_reviews.shape)
print("Amount of non zero values in the bag of words model:",bow_reviews.nnz)


# In[45]:


print("Sparsity:",np.round((bow_reviews.nnz/(bow_reviews.shape[0]*bow_reviews.shape[1]))*100,2))


# In[46]:


tfidf_transformer = TfidfTransformer().fit(bow_reviews)
tfidf_rev4 = tfidf_transformer.transform(bow_msg4)
print(bow_msg4)


# In[47]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['mango']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['book']])


# In[48]:


tfidf_reviews = tfidf_transformer.transform(bow_reviews)
print("Shape:",tfidf_reviews.shape)
print("No. of Dimensions:",tfidf_reviews.ndim)


# In[49]:


review_train, review_test, label_train, label_test = train_test_split(df['text_'],df['label'],test_size=0.35)


# In[50]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',MultinomialNB())
])


# # Naive Bayes Algorithm

# In[51]:


pipeline.fit(review_train,label_train)


# In[52]:


predictions = pipeline.predict(review_test)
predictions


# In[53]:


print('Classification Report:',classification_report(label_test,predictions))
print('Confusion Matrix:',confusion_matrix(label_test,predictions))
print('Accuracy Score:',accuracy_score(label_test,predictions))


# In[54]:


# Calculate the accuracy score and print it
NB_accuracy = accuracy_score(label_test, predictions)
print('Model Prediction Accuracy: {:.2f}%'.format(NB_accuracy * 100))


# In[55]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',RandomForestClassifier())
])


# In[56]:


pipeline.fit(review_train,label_train)


# In[57]:


rfc_pred = pipeline.predict(review_test)
rfc_pred


# In[58]:


print('Classification Report:',classification_report(label_test,rfc_pred))
print('Confusion Matrix:',confusion_matrix(label_test,rfc_pred))
print('Accuracy Score:',accuracy_score(label_test,rfc_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')


# In[59]:


RF_accuracy = accuracy_score(label_test, rfc_pred)

# Print the accuracy score
print('Accuracy Score:', RF_accuracy)


# In[60]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])


# In[61]:


pipeline.fit(review_train,label_train)


# In[62]:


dtree_pred = pipeline.predict(review_test)
dtree_pred


# In[63]:


print('Classification Report:',classification_report(label_test,dtree_pred))
print('Confusion Matrix:',confusion_matrix(label_test,dtree_pred))
print('Accuracy Score:',accuracy_score(label_test,dtree_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')


# In[64]:


DT_accuracy = accuracy_score(label_test, dtree_pred)

# Print the accuracy score
print('Accuracy Score:', DT_accuracy)


# In[65]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC())
])


# In[66]:


pipeline.fit(review_train,label_train)


# In[67]:


svc_pred = pipeline.predict(review_test)
svc_pred


# In[68]:


print('Classification Report:',classification_report(label_test,svc_pred))
print('Confusion Matrix:',confusion_matrix(label_test,svc_pred))
print('Accuracy Score:',accuracy_score(label_test,svc_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')


# In[69]:


SVC_accuracy = accuracy_score(label_test, svc_pred)

# Print the accuracy score
print('Accuracy Score:', SVC_accuracy)


# In[70]:


pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_process)),
    ('tfidf',TfidfTransformer()),
    ('classifier',LogisticRegression())
])


# In[71]:


pipeline.fit(review_train,label_train)


# In[72]:


lr_pred = pipeline.predict(review_test)
lr_pred


# In[73]:


print('Classification Report:',classification_report(label_test,lr_pred))
print('Confusion Matrix:',confusion_matrix(label_test,lr_pred))
print('Accuracy Score:',accuracy_score(label_test,lr_pred))
print('Model Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')


# In[74]:


LR_accuracy = accuracy_score(label_test, lr_pred)

# Print the accuracy score
print('Accuracy Score:', LR_accuracy)


# In[75]:


print('Performance of various ML models:')
print('Logistic Regression Prediction Accuracy:',str(np.round(accuracy_score(label_test,lr_pred)*100,2)) + '%')
print('Decision Tree Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,dtree_pred)*100,2)) + '%')
print('Random Forests Classifier Prediction Accuracy:',str(np.round(accuracy_score(label_test,rfc_pred)*100,2)) + '%')
print('Support Vector Machines Prediction Accuracy:',str(np.round(accuracy_score(label_test,svc_pred)*100,2)) + '%')
print('Multinomial Naive Bayes Prediction Accuracy:',str(np.round(accuracy_score(label_test,predictions)*100,2)) + '%')


# In[76]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Function to plot confusion matrix
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Confusion matrix for Logistic Regression
lr_cm = confusion_matrix(label_test, lr_pred)
plot_confusion_matrix(lr_cm, ['CG', 'OR'])

# Confusion matrix for Decision Tree Classifier
dtree_cm = confusion_matrix(label_test, dtree_pred)
plot_confusion_matrix(dtree_cm, ['CG', 'OR'])

# Confusion matrix for Random Forest Classifier
rfc_cm = confusion_matrix(label_test, rfc_pred)
plot_confusion_matrix(rfc_cm, ['CG', 'OR'])

# Confusion matrix for Support Vector Machines (SVM)
svc_cm = confusion_matrix(label_test, svc_pred)
plot_confusion_matrix(svc_cm, ['CG', 'OR'])

# Confusion matrix for Multinomial Naive Bayes
nb_cm = confusion_matrix(label_test, predictions)
plot_confusion_matrix(nb_cm, ['CG', 'OR'])


# In[80]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D

# Load the data
data = pd.read_csv(r'C:\fake reviews dataset.csv')

# Preprocess the text data
data['text_'] = data['text_'].apply(lambda x: x.lower() if isinstance(x, str) else '')

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text_'])
sequences = tokenizer.texts_to_sequences(data['text_'])

# Pad sequences to ensure uniform length
MAX_SEQUENCE_LENGTH = 100  # Adjust as needed
X_pad = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Convert labels to numerical format (CG: 0, OR: 1)
label_dict = {'CG': 0, 'OR': 1}
data['label'] = data['label'].map(label_dict)
y = data['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Define constants
EMBEDDING_DIM = 100  # Dimension of word embeddings
VOCAB_SIZE = len(tokenizer.word_index) + 1  # Vocabulary size

# Define the LSTM model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # Binary classification, so using sigmoid activation

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


# In[81]:


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:





# In[82]:


from sklearn.model_selection import StratifiedKFold
from keras.layers import Conv1D, MaxPooling1D, Flatten
# Define the number of folds for cross-validation
num_folds = 5

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Define lists to store validation results
val_losses = []
val_accuracies = []

# Iterate over each fold
for fold, (train_indices, val_indices) in enumerate(skf.split(X_pad, y)):
    print(f"Fold {fold + 1}/{num_folds}")
    X_train, X_val = X_pad[train_indices], X_pad[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    # Define and compile the model
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM))
    model.add(Conv1D(64, 5, activation='relu'))  # 64 filters of size 5 with ReLU activation
    model.add(MaxPooling1D(pool_size=4))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val))

    # Evaluate the model on validation data
    loss, accuracy = model.evaluate(X_val, y_val)
    val_losses.append(loss)
    val_accuracies.append(accuracy)

# Calculate and print average validation loss and accuracy across folds
avg_val_loss = np.mean(val_losses)
avg_val_accuracy = np.mean(val_accuracies)
print(f"Average Validation Loss: {avg_val_loss}")
print(f"Average Validation Accuracy: {avg_val_accuracy}")


# In[ ]:





# In[83]:


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:





# In[84]:


# Assuming X_test contains the new data for prediction

# Make predictions on the test data
predictions = model.predict(X_test)

# Convert the probabilities to class labels (0 or 1)
predicted_labels = (predictions > 0.5).astype(int)

# Print the predicted labels
print("Predicted Labels:", predicted_labels)


# In[ ]:





# In[85]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




