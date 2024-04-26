# TCSiON
RIO-125: Automate sentiment analysis of textual comments and feedback
Project Description:
Project Objective & Brief: To develop a deep learning algorithm to detect different types of sentiment contained in a collection of English Sentences or a large paragraph.

Project Guidelines:
Identify & finalize a collection of English sentences or a large paragraph which will also cover contradictory statements.
Develop a deep learning model for detection & segmentation of sentiments whether positive, negative, or neutral from the paragraph.
Test the model accuracy.
Expected Project Outcome:
Algorithms to detect different types of sentiment from a paragraph.
Detailed presentation with proof of reasonable accuracy.
Software Requirements to Implement the Project:
Google Collab
Python, Java (opt.), Eclipse, Ubuntu OS (opt.)
In Sentiment analysis, the data mainly focuses on whether the feedback to a product is :

negative
positive
neutral
This provides a top level review of the product and its acceptance in the market.

Emotion Analysis provides a deeper insight of consumer emotions.
Analyzing textual feedback to the level of reading between lines.
Categorizing feedback and analyzing its emotion by picking up words, contexts, patterns, behaviors. This can be even taken to the level of individual’s expressive capability of a particular situation.
Approaches to Text Classification
Rule Based Systems:
Rule-based approaches classify text into organized groups by using a set of linguistic rules.
Each rule comprises of a pattern based on semantics and its predicted category.
Machine Learning based Systems:
Text classification based on past observations.
By using training data, the algorithm can learn the different associations between pieces of text and that a particular output (i.e. tags) is expected for a particular input (i.e. text).
Feature extraction: Transforms each text into a numerical representation in the form of a vector. E.g. bag of words [a vector represents the frequency in a predefined dictionary of words ]
The algorithm is fed with training data consisting of feature sets.
Once trained with enough training samples, the machine learning model can begin to make accurate predictions on unseen text with similar feature sets.
Text Classification Algorithms
Some of the most popular machine learning algorithms for creating text classification models include the naive bayes family of algorithms, support vector machines, Regressions, and deep learning algorithms with CNN and RNN. Metrics and Evaluation Cross-validation is a common method to evaluate the performance of a text classifier.

It consists in splitting the training dataset randomly into equal-length sets.
For each set, a text classifier is trained with the remaining samples (e.g. 75% of the samples).
The classifiers make predictions on their respective sets and the results are compared against the human-annotated tags.
With these results, a performance metrics is built, that are useful for a quick assessment on how well a classifier works.
Performance metrics normally includes:
Accuracy: the percentage of texts that were predicted with the correct tag.
Precision: the percentage of examples the classifier got right out of the total number of examples that it predicted for a given tag.
Recall: the percentage of examples the classifier predicted for a given tag out of the total number of examples it should have predicted for that given tag.
F1 Score: the harmonic mean of precision and recall.
Data Sources to solve problem
Data Sets from any open source data sources can be used. Model building and solution engineering is NOT restricted to data.
Example:
https://ai.stanford.edu/~amaas/data/sentiment/
https://data.world/crowdflower/sentiment-analysis-in-text
https://www.kaggle.com/crowdflower/twitter-airline-sentiment
Test Accuracy achieved with IMDB dataset using following algorithms:
RNN- 2 LSTM Layers: 85.24% : https://colab.research.google.com/drive/1LR0vlfJLbw_e2ECeICkKOqGY7PtiKm7z
RNN- 1 LSTM Layer: 85.49% : https://colab.research.google.com/drive/1peyWG5zP9R3eJdnkw9TZihc6imyenqXo
Pre-processed Text: 85.5% : https://colab.research.google.com/drive/1PulJf43YSeGEI0FHb5pAU5nMkvZtcFwF
TF (Keras) Hub Layer: 84.9%
