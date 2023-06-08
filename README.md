# Fake_News_Classifier

This project is a fake news classifier that aims to distinguish between fake and real news articles using machine learning techniques. The classification is based on the textual content of the articles.

## Prerequisites
Make sure you have Python installed on your system. You can use the requirements.txt file to install the necessary packages. Run the following command to install the required packages:

pip install -r requirements.txt

## Usage
Read the CSV data file containing the news articles.
Perform data preprocessing steps such as handling missing values, dropping unnecessary columns, and applying stemming to the text.
Split the data into training and testing sets.
Perform feature extraction using TF-IDF vectorization.
Train a decision tree classifier model on the training data.
Make predictions on the testing data using the trained model.
Evaluate the model's accuracy.
Save the trained model to a file.
Load the saved model and use it for prediction.
Note: Adjust the file paths and names according to your specific environment.

For detailed code and instructions, refer to the code comments and documentation within the script.

## Dataset
The project assumes the existence of a CSV data file containing the news articles. Ensure that the file path is correctly specified.

Conclusion
The fake news classifier project provides a foundation for identifying fake news articles using machine learning algorithms. By following the provided code and instructions, you can build and train your own fake news classification model.
