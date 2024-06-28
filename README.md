# Sentiment Analysis of IMDB Movie Reviews using Naive Bayes and TF-IDF

## Overview

This project focuses on sentiment analysis of IMDB movie reviews using Natural Language Processing (NLP) techniques. 
The goal is to build a classifier that can predict whether a movie review expresses positive or negative sentiment 
based on its text content. We employed the TF-IDF vectorization technique and the Multinomial Naive Bayes classifier 
for this task.

## Project Structure

The project is structured as follows:

- **Data:** The dataset consists of three main parts: training, validation, and test sets. Each set contains IMDB
            movie reviews labeled with their corresponding sentiment (positive or negative).

- **Preprocessing:** 
  - **Loading Data:** The data is loaded using Pandas' `read_csv` function from CSV files (`imdb_train.csv`,
                      `imdb_valid.csv`, `imdb_test.csv`).
  - **Vectorization:** Text data is converted into numerical features using TF-IDF (Term Frequency-Inverse Document
                       Frequency) vectorization. We experimented with both unigram features (`ngram_range=(1, 1)`) and
                       a combination of unigrams and bigrams (`ngram_range=(1, 2)`).

- **Modeling:**
  - **Naive Bayes Classifier:** We employed the Multinomial Naive Bayes classifier, suitable for classification tasks
                                with discrete features like word counts in text data. Naive Bayes assumes that features
                                are conditionally independent given the class, which aligns well with the bag-of-words
                                model in text classification.

- **Training and Evaluation:**
  - **Model Training:** Two models were trained: one using only unigram features and another using both unigrams and bigrams.
  - **Validation:** Models were evaluated on a separate validation set to tune hyperparameters and select the best-performing
                    model based on validation accuracy.
  - **Testing:** The best model was evaluated on the test set to assess its generalization performance. We reported the accuracy
                 achieved on the test set.

- **Feature Analysis:**
  - **Top Predictive Features:** We identified the top-10 most predictive features for sentiment analysis using the difference in
                                 log probabilities between positive and negative sentiment classes. These features represent the
                                 words or word combinations that strongly influence the classification decision.

## Results and Interpretation

- **Validation Results:** The model using both unigrams and bigrams consistently outperformed the model using only unigrams, indicating
                          that incorporating bigrams improves sentiment analysis accuracy.
  
- **Test Accuracy:** The best model achieved an accuracy of approximately 89% on the test set, demonstrating its effectiveness in
                     predicting sentiment from movie reviews.

- **Top Predictive Features:** The top-10 most predictive features included phrases such as "worst movie", "highly recommended",
                               and "loved it", which strongly correlated with either positive or negative sentiment in reviews.

## Conclusion

In conclusion, this project showcases the application of NLP techniques for sentiment analysis using a classic approach: TF-IDF 
vectorization and Naive Bayes classification. The results highlight the importance of feature selection (unigrams vs. bigrams) 
in improving sentiment prediction accuracy. Future work could explore more advanced techniques such as deep learning models or 
ensemble methods for further enhancing performance.

## Usage

To replicate or extend this project:

1. Clone the repository.
2. Ensure Python and necessary libraries (scikit-learn, pandas) are installed.
3. Modify the dataset paths in the code (`imdb_train.csv`, `imdb_valid.csv`, `imdb_test.csv`) as per your data location.
4. Run the provided scripts or adapt them for your specific experiments and analyses.

## Acknowledgments

This project was completed as part of the CS589 Natural Language Processing course at San Francisco Bay University. 
Special thanks to Prof. Arun Jagota for guidance and support.

## Author

Natnael Haile
