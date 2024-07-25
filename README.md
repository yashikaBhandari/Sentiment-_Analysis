 followingg libraries have been used :-

numpy (imported as np): NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. It's widely used in scientific computing and data analysis.

pandas (imported as pd): Pandas is a data manipulation and analysis library for Python. It offers data structures and operations for manipulating numerical tables and time series. It's particularly useful for working with structured data, such as CSV files or SQL tables.

re: The re module provides support for regular expressions (regex) in Python. Regular expressions are sequences of characters that define search patterns. They can be used for searching, extracting, and replacing patterns in text data.

nltk.corpus and nltk.stem.porter: These are modules from the Natural Language Toolkit (NLTK), which is a library for natural language processing (NLP) in Python. nltk.corpus provides access to various corpora and lexical resources, while nltk.stem.porter is a module for stemming, which is the process of reducing words to their root or base form.

sklearn.feature_extraction.text.TfidfVectorizer: This is a module from scikit-learn, a machine learning library in Python. TfidfVectorizer is used for converting a collection of raw documents into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. TF-IDF is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents.

sklearn.model_selection.train_test_split: This function is used for splitting arrays or matrices into random train and test subsets. It's commonly used in machine learning for evaluating model performance on unseen data.

sklearn.linear_model.LogisticRegression: This is a module for logistic regression, a popular statistical method for binary classification problems. Logistic regression models the probability of a binary outcome based on one or more predictor variables.

sklearn.metrics.accuracy_score: This function computes the accuracy classification score, which is the fraction of correctly classified samples. It's a common metric for evaluating the performance of classification models






To use this first you need to login to keggle so that you get your own API or you can dirstly download the dataset then use the code.

This dataset is imported from kaggle which has 1.5 million datasets , having both positive and negative sentiment.

Accuracy of this model is around 79%
