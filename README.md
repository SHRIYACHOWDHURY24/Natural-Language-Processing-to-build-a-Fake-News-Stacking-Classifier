# Natural-Language-Processing-to-build-a-Fake-News-Stacking-Classifier
Natural Language Processing creates token vectors of the merged column of the title and author of each article of the training dataset. The merged column is first cleaned of the stopwords present in the nltk toolkit and then stemmed using PortStemmer to reduce the words to their original form and removing any prefix or suffix in the word. The tfidfvectorizer converts the textual data into numerical vector where each decimal point at the indices signify the importance of the word in the text. 
The training vector is given as input to the 5 training models : k-nearest neighbours classifier model, Random Forest Classifier model, Support Vector Classifier model, Gaussian Naïve Baye’s Probability model and Logistic Regression Classifier model. The Stacking Classifier model with a Logistic Regression model as the meta classifier, incorporates these 5 classifiers to predict whether the news is fake or real.

1.	Testing data accuracy after training the k-nearest neighbours classifier model ( classifier1 ) = 58.7019 % 
2.	Testing data accuracy after training the Random Forest Classifier model ( classifier 2 )= 94.0385 % 
3.	Testing data accuracy after training the Support Vector Classifier model ( classifier 3 ) = 99.1346 % 
4.	Testing data accuracy after training the Gaussian Naïve Baye’s Probability model ( classifier 4 ) = 80.3846 % 
5.	Testing data accuracy after training the Logistic Regression Classifier model ( classifier 5 ) = 97.9086 % 
6.	Accuracy of the Stacked Classifier model = 98.3173 % 

 
 
 
 
 
 
 
  
