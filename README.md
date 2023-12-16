# Machine-Learning-for-Movie-Sentiment-Analysis


This project introduces a novel approach to movie sentiment analysis, utilizing advanced 
machine-learning techniques to interpret the emotional content of user-generated movie
reviews. We deploy two supervised learning models: Random Forest (RF) and Logistic 
Regression (LR), and rigorously compare their performance in terms of Accuracy, Precision, 
Recall, and F1 Score. Our findings indicate that the Logistic Regression model demonstrates 
superior performance with an 85% accuracy rate, compared to 83% for Random Forest. This 
highlights Logistic Regression's effectiveness in sentiment analysis tasks, enhancing the 
potential of machine learning in understanding user emotions related to movies.

Data Collection:
Our dataset, obtained from Kaggle, consists of 50,000 movie reviews. Each review is 
assigned a sentiment class label, either "positive" or "negative," based on the content 
analysis.
Data Pre-processing:
Text preprocessing is a foundational and critical step in the preparation of a substantial 
dataset containing 50,000 movie reviews for sentiment analysis. This process is pivotal in 
enhancing the quality and relevance of the data for subsequent analysis. 

Model Selection :
In the process of model selection, we evaluated and compared two classification models: 
Logistic Regression and Random Forest.
- Logistic Regression: Logistic Regression is a type of linear model characterized by 
one or more independent variables describing a relationship with a dependent 
response variable. It involves mapping qualitative or quantitative input features to a 
target variable in an attempt to predict the outcome.

Random Forest: The Random Forest is an ensemble learning algorithm consisting of 
n sets of uncorrelated decision trees. It is grounded in the concept of bootstrap 
aggregation, a resampling technique with replacement aimed at minimizing variance. 
In the Random Forest framework, multiple trees are employed to calculate the 
average (for regression) or determine majority votes (for classification) within the 
terminal leaf nodes when making predictions .

Model Training and Evaluation:
In our sentiment analysis task, the model training process begins with data preprocessing 
using natural language processing (NLP) techniques. We employ the NLTK library to 
tokenize, remove stop words, and apply lemmatization and stemming to the movie reviews. 
The pre-processed data is then transformed using the Term Frequency-Inverse Document 
Frequency (TF-IDF) technique, limiting the features to the top 500.
To ensure robust model evaluation, we split the dataset into training and testing sets, 
allocating 80% for training and 20% for testing. The training set is further divided into 
subsets for actual training and validation purposes.
Two classification models, namely Random Forest and Logistic Regression, are selected for 
sentiment analysis. The models are trained using the training data and subsequently evaluated 
on the testing set. The evaluation metrics include accuracy, precision, recall, and the F1 
score, providing a comprehensive understanding of each model's performance.
In the subsequent section, "Results and Determine the Best Model," detailed classification
reports for both models will be provided, outlining precision, recall, and F1 score for each 
sentiment class. These metrics serve as crucial indicators of the models' effectiveness in 
distinguishing between positive and negative sentiments in movie reviews.
These training and evaluation steps were executed using Python in a Jupyter Notebook 
environment. The ultimate goal is to determine the best-performing model for sentiment 
analysis based on these comprehensive evaluation metrics.

Conclusion:
 The study is dedicated to classifying reviews into positive or negative sentiments, employing two algorithms—
Random Forest and Logistic Regression. The use of both algorithms aims to identify the most 
effective approach for handling the diverse range of reviews, considering the inherent 
variability in review content. The experimental results highlight that the Logistic Regression 
Algorithm exhibits slightly better accuracy than Random Forest in the context of sentiment 
analysis. Moving forward, potential avenues for further research include improving the 
accuracy of sentiment analysis, particularly in discerning sarcastic or ironic reviews. 
Additionally, expanding the analysis to include reviews in languages other than English and 
refining classification based on user preferences are areas of interest.
