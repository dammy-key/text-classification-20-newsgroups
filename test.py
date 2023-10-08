from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Accessing the 20 Newsgroups Dataset

# Download and load the dataset
newsgroups = fetch_20newsgroups(subset='all', shuffle=True, remove=('headers', 'footers', 'quotes'))

# Step 2: Exploring the Dataset

# List the categories (classes)
categories = newsgroups.target_names
print("Categories:")
print(categories)

# Number of categories
num_categories = len(categories)
print("Number of Categories:", num_categories)

# Number of documents
num_documents = len(newsgroups.data)
print("Number of Documents:", num_documents)

# Step 3: Building a Text Classification Model

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)

# Convert text data to TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = nb_classifier.predict(X_test_tfidf)

# Step 4: Evaluate the Model

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))
