# Natural-Language-Processing-to-build-a-Fake-News-Stacking-Classifier
Natural Language Processing creates token vectors of the training dataset and feeds it to a Stacking Classifier model, incorporating K Nearest Neighbours Classifier, Random Forest Classifer, Support Vector Classifier, Linear Regression and Naive Bayes Probabilistic Classifier, with a Logistic Regression model as the meta classifier.

# Software and Python Libraries Used: 
Python, NumPy, Pandas, NLTK (Natural Language Toolkit), Scikit-learn, TfidfVectorizer, train_test_split, LogisticRegression, RandomForestClassifier, LinearSVC, KNeighborsClassifier, GaussianNB, StackingClassifier, plot_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score.

# Mechanisms and Algorithms Used: 
1. Data Loading and Preprocessing
Mechanism:
The dataset is loaded using Pandas, which reads the CSV file containing the news data. During preprocessing, any missing values in the dataset are filled with empty strings, ensuring that there are no gaps that could disrupt model training.

Key Steps:

Tokenization: The text is split into tokens (words or subwords) using NLTK. This breaks down the sentences into individual words for further analysis.

Stopword Removal: Common words that do not contribute significant meaning (like "the", "is", "in") are removed using NLTK's stopwords list. This reduces noise in the dataset.

Stemming: Stemming reduces words to their base forms. For example, "running" becomes "run." The Porter Stemmer from NLTK is used for this. This process helps in normalizing words to their root form, making text analysis more efficient.

Text Cleaning: Non-alphabetic characters are removed using regular expressions to ensure the text only contains words, making it easier to process.

2. Feature Extraction with TF-IDF
Mechanism:
The cleaned text data is converted into numerical features using TfidfVectorizer from Scikit-learn. TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure that evaluates how important a word is to a document in a collection of documents.

How it works:

Term Frequency (TF): Measures how frequently a term appears in a document.
Inverse Document Frequency (IDF): Measures how important a term is across multiple documents (reduces the weight of common terms).
Output: The text is transformed into a matrix where each word is represented by a numerical value, allowing machine learning algorithms to process it.

3. Model Building and Training
Mechanism:
Several machine learning models are trained using the preprocessed and transformed data.

Train-Test Split: The dataset is split into training and testing sets using train_test_split from Scikit-learn. This helps in evaluating the model's performance on unseen data.

Models Used:

Logistic Regression: A linear model used for binary classification (in this case, to predict fake or real news). It outputs the probability of a news article being fake.

Random Forest Classifier: An ensemble model that builds multiple decision trees and averages their predictions to improve accuracy and prevent overfitting.

Support Vector Classifier (LinearSVC): A model that tries to find the best hyperplane to separate the data into two classes (fake or real news).

K-Nearest Neighbors (KNN): A non-parametric method used to classify data based on the majority vote of its neighbors.

Gaussian Naive Bayes: A probabilistic model based on Bayes’ Theorem that assumes independence among features.

Stacking Classifier: A Stacking Classifier is used to combine the predictions of multiple models (Logistic Regression, Random Forest, LinearSVC, KNN, and GaussianNB). The idea is that by combining different models, the overall performance improves. Meta-classifier (Logistic Regression): After each model generates predictions, the Logistic Regression model acts as a meta-classifier to combine these predictions and make the final decision.

4. Evaluation and Results
Mechanism:
The performance of each model is evaluated using several metrics:

Accuracy: Measures the proportion of correct predictions (both true positives and true negatives) out of the total predictions.
Precision: The proportion of true positive predictions among all positive predictions.
Recall: The proportion of true positive predictions among all actual positives.
F1-score: The harmonic mean of precision and recall, providing a balance between the two.
Confusion Matrix: Displays the distribution of true positives, true negatives, false positives, and false negatives. This gives insight into how well the model is distinguishing between real and fake news.

The Stacking Classifier, which combines the predictions of the base classifiers, achieved the highest accuracy of 98.31%, outperforming the individual models.







