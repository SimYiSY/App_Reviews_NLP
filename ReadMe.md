# Shopping Apps, Rating for Google Play Store and Apple AppStore Users

<img src="https://image.freepik.com/free-vector/cartoon-delivery-man-brings-goods-customer-from-laptop-vector-illustration-concept-with-online-shopping-services_46527-344.jpg" />

## Introduction

Users download apps for various purposes. Given that there is a rise in the usage of online shopping due to the Covid-19 pandemic, improvement of shopping experience has become more important then before. With that in mind, what are the important features we have to look out for to improve a shopping app?

## Problem Statement

- How do the app ratings differ across different shopping apps?
- Is there any specific group of users we can look out for to improve the app?
- Are there any specific improvement we can work on to further improve user satisfaction of the app?

To explore and answer the above questions, we will scrap reviews from Google Play Store and Apple AppStore and conduct analysis and modelling.

## Executive Summary

The data is webscrapped from the Shopping category in Google Play Store and Apple Appstore, 8 apps reviews were chosen for this project (Amazon, Wish, ASOS, Lazada, Ebay, Shoppee, AliExpress, Carousell). The data used was exclusive dated in 2020 only as majority of the data scrapped are from in 2020. Data cleaning was done by removing stopwords, lemmatized and Vectorized to the raw data to create bag-of-words. 

There will be 2 steps to our modelling process, with the first step classifying whether the text is a good or bad review, followed by classifying the reviews into categories created through topic modelling to group them into different subgroups.

A few classification model were used, namely LogisticRegression, MultinomialNB, SGDClassifier, RandomForest, ADABoost. LogisticRegression give us the best results in classifying our data and thus used as the final model. 

As the data set is quite big, RandomizedSearch was used instead of Gridsearch to find the best hyperparameter.



### Content Summary
- Webscrapped reviews of 8 apps from Google Play Store & Apple App Store
- Data Cleaning 
  - Removing data not in year 2020
  - Removing emoji and punctuations
  - Removing non english words
  - Lemmatization
  - Compound score calculation using VaderSentiment
- EDA
- - Check for wrongly Rated reviews
  - Plotting distribution of features
  - Topic modelling of good reviews
  - Topic modelling of bad reviews
- Machine Learning Model 
  - LogisticRegression
  - MultinomialNB
  - SGDClassifier
  - RandomForest
  - ADABoost
- Deep Learning Model
  - Convolutional Neural Network
  
### Data Dictionary 

|Feature|Type|range|comment| 
|-----|-------|------|-|
|rating|int64|1 - 5|Ratings of apps by user|
|date|datetimens|Year 2020|Year of the data|
|app|object|Apple / Google|Which store was the review from|
|store|object|8 apps|Which app was the review from|
|review|object|raw reviews|Review of user|
|clean_content|object|cleaned reviews|Cleaned review of user|
|adj|object|nil|Adjectives of reviews|
|noun|object|nil|Noun of Reviews|
|verb|object|nil|Verb of Reviews|
|emoji|object|nil|Emoji of Reviews
|neg_score|float64|0 - 1|Negative score of review using VaderSentiment|
|neu_score|float64|0 - 1|Neutral score of review using VaderSentiment|
|pos_score|float64|0 - 1|Positive score of review using VaderSentiment|
|compound_score|float64|(-1) - 1|Compound score of review using VaderSentiment|
|language|object|nil|Language type of review|
|month|int64|1 - 12|Month review was posted|
|dayofweek|int64|1 - 7|Day of the week review was posted|
|hour|int64|0 - 23|Hour review was posted|
|minute|int64|1 - 60|minute review was posted|
|text_len|int64|nil|total number or letters in review|
|word_count|int64|nil|total number of words in reivew|
|category|object|20 topics|Categories generated from topic modelling|
|rate|int64|0 / 1|Rate of Good/Bad Reviews|

### Key Findings

- How do the app ratings differ across different shopping apps?
    - Users the wants to give a bad review will tend to just give a 1 rating
    - Users that wants to give a good review will tend to just give a 5 rating
    - Amazon seems to take the lead in having more bad reviews as compared to the other apps, especially App Issues , User interface and Account issues
    - Wish has slightly higher negative delivery issue reviews
    - AliExpress has High Refund issues. 
    - Shoppee takes the lead in Convenient app while ASOS takes the lead for User Interface
    - Bad reviews tend to have more word count as compared to Good reviews
    - More negative reviews are seen in 9am - 3pm period
    - More negative reviews on Tuesdays
    - Good Reviews are mostly on Convenient App and User Interface
    - Bad reviews are mostly on User Interface, App Issues and Purchase Experience
    - There is more negative reviews in 9am - 3pm period, and on Tuesdays
    - There is quite a number of reviews being 1 word, or otherwise rated wrongly by the user, (e.g. review: Excellent, Rating: 1)
- Is there any specific group of users we can look out for to improve the app?
    - The categories for Good Reviews are:  Convenient App, User Interface, Variety & Price, User Experience, Shopping Experience, Delivery, Consumer Satisfaction, In-App Actitives, Recommendations, Customer Service
    - The categories for Bad Reviews are:  Account Issues, Poor Seller Feedback, User Interface, Payment Issue, Poor Customer Service, Product Issue, App Issues, Delivery Issue, Refund, Product Listing Issues
- Are there any specific improvement we can work on to further improve user satisfaction of the app?
    - Ratings for Refund seems to take a dip in March
    - Ratings for Refund and Poor Seller Feedback is consistently low across the week.
    - Consumer Satisfaction is always going lower as compared to other topics in the good reviews sector
    - Customer Service and Consumer satisfaction is consistently lower in the week compared to other categories in good reviews
    - Delivery, Consumer Satisfaction and Recommendations are scored lower in good reviews.
    - Poor Seller feedback and refunds are scored lower in bad reviews
    - Base on compound scores, Delivery and Product tend to be very low as compared to other categories.
    - To summarise, it is important to look into customer satisfaction for the app, where Customer Service, Refund, Delivery and Recommendations are import factors to look out for. 
    - It is also important to put some focus on sellers that are doing bad on the platform.

### Metrics
Using the following metrics to evaluate the models:
- ROC AUC curve(for Binary Classification)
  -  The ROC AUC curve is able to tell how much the model is capable of distinguishing between 0 and 1, with 1 being perfectly classified.
- MCC Score
  - The Matthews correlation coefficient (MCC), instead, is a more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset.
- f1 score weighted
  - The F1 Scores are calculated for each label and then their average is weighted by support - which is the number of true instances for each label. It can result in an F-score that is not between precision and recall

### Final Results
**Classification (Good & Bad Reviews)**
- LogisticRegression
  - Train data AUC: 0.966
  - Test data AUC: 0.965
  - MCC Score: 0.773
- Convolutional Neural Network
  - Train data AUC: 0.977
  - Test data AUC: 0.961
  - MCC Score: 0.767
  
**Multi Classification (Bad Review categories)**
- LogisticRegression
    - Train Data f1 weighted score: 0.731
    - Test Data f1 weighted score: 0.741	
    - MCC Score: 0.705	
- Convolutional Neural Network
  - Train data Acciracy: 0.914
  - Test data Accuracy: 0.717
  - MCC Score: 0.679

**Multi Classification (Good Review categories)**
- LogisticRegression
    - Train Data f1 weighted score: 0.868	
    - Test Data f1 weighted score: 0.878	
    - MCC Score: 0.851	
- Convolutional Neural Network
  - Train data Acciracy: 0.941
  - Test data Accuracy: 0.841
  - MCC Score: 0.805

**Model Remarks**
- From the misclassified post we can see the some comments are rated wrongly if we were to just look at the reviews directly.
- Some of the reviews are predicted wrongly. After looking at some of the reviews, it is clear that there are some misclassification by the topic modelling.
- It can be seen that the model is actually predicting better then what was classified in the first place.
- Some of the topics are very closely related to one another, which make it harder for the model to predict correctly
- Deep Learning Models are performing worst then Machine learning models, which could be due to the lack of complexity of the data for Neural Network to work well.

### Limitations
- The data set is mostly collected in the month of August and September, which means the model is able to predict this period better, but not in predicting past data. 
- More data could be collected, as there is a major lack of Apple Appstore reviews compared to Google Play Store

### Further research
- Try to use Compound score gathered from VaderSentiment to do the classification instead, as we know there is some misclassified post by users. which hopefully give us a better accuracy.
- Try different categories, not just shopping app category apps, do create a more complete review prediction model

### Content
1. Webscrap data
2. Data Cleaning
3. EDA
4. Model Part 1, Classification (Good & Bad Reviews)
5. Model Part 2, Multi Classification (Bad Review categories) 
6. Model Part 3, Multi Classification (Good Review categories)
7. Deep Learning Model