# Email_Spam_Detection
📩 SMS Spam Detection using Machine Learning
This project is a complete end-to-end solution for detecting spam messages using classic machine learning techniques and natural language processing. The objective is to classify SMS messages as spam or ham (not spam) based on their textual content.

🔍 Overview
Spam messages are not only annoying but can also pose serious privacy and security risks. In this project, I’ve used Python and several machine learning algorithms to build a classification system that can accurately identify and filter out spam messages.

The project covers everything from raw data preprocessing to model evaluation and comparison, with an interactive and intuitive pipeline.

📁 Dataset
Source: UCI SMS Spam Collection Dataset

File Used: spam.csv

Size: ~5,500 messages labeled as ham or spam

⚙️ Steps Implemented
1. Data Cleaning
Removed irrelevant columns

Renamed columns for clarity

Encoded target labels (ham = 0, spam = 1)

Removed duplicate entries

2. Exploratory Data Analysis (EDA)
Class distribution visualizations

Analysis of message length, word count, and sentence structure

Most common words in spam vs ham messages

Word clouds for spam and ham text

3. Text Preprocessing
Lowercasing text

Tokenization

Removing punctuation and stopwords

Stemming (PorterStemmer)

Final cleaned text stored in Transformed_Text

4. Feature Extraction
TF-IDF vectorization (top 300 features)

5. Model Building
Compared multiple classifiers:

Naive Bayes (Gaussian, Multinomial, Bernoulli)

SVC

Logistic Regression

KNN

Decision Trees

Random Forest

AdaBoost, Bagging, Extra Trees, Gradient Boosting, XGBoost

Evaluation based on Accuracy and Precision

6. Model Evaluation
Plotted comparison bar charts

Precision was prioritized due to the cost of false positives

Multinomial Naive Bayes + TF-IDF performed best overall

7. Ensemble Techniques
Voting Classifier: Combined SVM, Naive Bayes, and Extra Trees

Stacking Classifier: Used Random Forest as meta-classifier

📊 Results
Model	Accuracy	Precision
MultinomialNB + TF-IDF	0.97	0.94
SVC (sigmoid kernel)	0.96	0.92
Ensemble (Voting)	0.97	0.95
Stacking Classifier	0.97	0.95

The ensemble models slightly improved the balance between accuracy and precision.

🛠️ Tech Stack
Python 3.9+

Pandas, NumPy – Data handling

Matplotlib, Seaborn – Visualizations

NLTK – Natural language processing

scikit-learn – Machine Learning models

XGBoost – Advanced boosting
