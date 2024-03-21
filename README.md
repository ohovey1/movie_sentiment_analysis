# Sentiment Analysis of IMDB Movie Reviews Using Classification Models

*See below notebook for code*

https://github.com/ohovey1/movie_sentiment_analysis/blob/main/ml_models.ipynb

## Introduction:
This project involved sentiment analysis of IMDB movie reviews using various machine
learning classification models. A dataset of IMDB reviews and their associated sentiments
were used to train the models and make predictions about natural language sentiment in the
context of movie reviews. This project was broken down into the following sections:

1. Data Exploration (EDA)
2. Data Preprocessing
3. Data Modelling
4. Results

We’ll take a deep look at each of these sections to get a full scope of the project
implementation, as well as explore the real world use cases.

## Data Exploration:
The first section of the project involved performing some preliminary EDA on the provided
dataset. This is a crucial starting point for any ML project as it gives important insights on
how the data needs to be processed for model training. The first step taken in this section was
to import necessary packages and read the csv file into a pandas dataframe. We start by
printing key information about the data using pandas methods like info(), describe(), etc. This
helps us understand the structure of the data. Luckily, the dataset was already very organized
and easy to work with. From the analysis, there are a few key takeaways:

1. Size: Dataset contains 50,000 entries, each with a review and a sentiment
2. No missing values: Both columns contain equal number of values -- so no need to
handle any missing entries
3. Unique reviews: There are 49,582 unique reviews, meaning some of the reviews
are duplicates but most are unique
4. Balance: The dataset is perfectly balanced, with 25,000 positive and 25,000
negative reviews

In summary, the dataset seems to be clean and well balanced. Next, we will move onto
preprocessing the text data for model training.

## Data Preprocessing:
After importing the necessary modules, we will now begin preprocessing our data for feature
extraction. For stemming, we will use PorterStemmer from the NLTK package. We defined a
function to handle the basic preprocessing steps. After applying it to our dataframe, we see
that all text is lowercased, there are no apparent html tags or special characters, and the text
has been stemmed.
From here, we will use TfidVectorizer to fit and transform the preprocessed reviews. This
converts the text data into a matrix of TF-IDF features. This matrix reflects the importance of
a specific word is to a document in a collection or corpus (search). Fitting the vectorizer on
our review column means it is learning the vocabulary of the dataset. It is then transformed
into a numerical vector which reflects the importance of a word in a particular document. The
importance of using vectorization is that it converts the unstructured text data into a
structured form that can be used as input to a machine learning model. Machine learning
algorithms work with numerical data, and vectorization methods convert text into numerical
data while preserving the semantic relationship between words. Finally, the data is split into
training and test sets which will be used to train the ML models.

## Data Modelling
The first machine learning models that were implemented were Logistic Regression,
LinearSVC, and KNeighbors Classifier. These three models were pretty straightforward to
implement – especially since our dataset is so cleanly organized. However, our modelling got
a bit more complex when implementing fully connected layers and CNN. These models are
more advanced deep learning algorithms and require more detailed parameter tuning.
Starting with fully connected layers, we used grid search to identify the optimal parameters
for our model. Grid search is a hyperparameter tuning technique to find the optimal
parameters for a model. It involves defining a grid of hyperparameters and training a model
on every combination of these hyperparameters. Each model is then evaluated on a validation
set using cross validation. We found that the best performing model had the following
parameters: {'activation': 'relu', 'hidden_layer_sizes': (100, 50)}, with an accuracy score of
.873.

To implement CNN, we'll use numpy, TensorFlow and Keras to test the model with different
combinations of convolutional and fully connected layers. This will require some further data
preprocessing steps to prepare it for a CNN model. Previously, we converted the raw text data
to a matrix of TF-IDF features. This is suitable for the other models we trained but will not
work for CNN. The previous models treated each document as a "bag of words", but CNN
considers sequence of the words as well. CNN’s are typically used for image, so we had to
adjust it for text classification. Several different models were tested by adjusting the number
of convolutional layers and fully connected layers, activation functions, and number of
sequence inputs. The best performing model found had 1 convolutional layer, 1 dense layer,
and 500 sequences – predicting sentiment with .885 accuracy.

## Results
Here are the accuracy results for each of the models:

![image](https://github.com/ohovey1/movie_sentiment_analysis/assets/89608419/432d3ccf-e990-4048-b7ca-da5830656b80)

With the top 5 performing similarly, followed by a drop off in performance for the KNN
models.

## User Testing:
To allow for user testing, we allow the user to input their own reviews. The following user
inputted prompts were tested on the following models:

![image](https://github.com/ohovey1/movie_sentiment_analysis/assets/89608419/8f9643a9-a74f-4af5-8a42-297e8d6993cf)

Here, we can test different reviews and see how the models react. We see some shortcomings
in the KNN model in the last prompt when the review contradicts itself, but each of the other
models are still able to decipher the overall sentiment. For straightforward reviews, all
models are able to accurately determine whether or not the review is positive or negative.
