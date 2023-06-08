import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import joblib

# Download NLTK stopwords
nltk.download('stopwords')

# Read the CSV data file
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/datasets/fake_news.csv')

# Display the first few rows of the data
data.head()

# Statistical description of the data
data.describe()

# Information about the data columns
data.info()

# Check for missing values
data.isnull().sum()

# Fill missing values with empty strings
data = data.fillna('')

# Check for missing values again
data.isnull().sum()

# Drop unnecessary columns
data = data.drop(['id', 'title', 'author'], axis=1)

# Display the modified data
data.head()

# Create a PorterStemmer object
port_stem = PorterStemmer()

# Perform stemming on a sample word
port_stem.stem('praveen')

# Define a stemming function
def stemming(content):
  cont = re.sub('[^a-zA-Z]', ' ', content)
  cont = cont.lower()
  cont = cont.split()
  cont = [port_stem.stem(word) for word in cont if not word in stopwords.words('english')]
  cont = ' '.join(cont)
  return cont

# Apply stemming to the 'text' column of the data
data['text'] = data['text'].apply(stemming)

# Separate the features (x) and labels (y)
x = data['text']
y = data['label']

# Check the shape of x and y
x.shape
y.shape

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Create a TfidfVectorizer object
vector = TfidfVectorizer()

# Fit and transform the training data
x_train = vector.fit_transform(x_train)

# Transform the testing data
x_test = vector.transform(x_test)

# Check the shape of the transformed data
x_train.shape
x_test.shape

# Create a DecisionTreeClassifier model
model = DecisionTreeClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Make predictions on the testing data
prediction = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = model.score(x_test, y_test)
print("Model accuracy:", accuracy)

# Save the trained model to a file
joblib.dump(model, 'fake_news_classifier_model.pkl')

# Load the saved model from the file
loaded_model = joblib.load('fake_news_classifier_model.pkl')

# Use the loaded model for prediction
new_prediction = loaded_model.predict(x_test)
new_accuracy = loaded_model.score(x_test, y_test)
print("Loaded model accuracy:", new_accuracy)
