### Overview
This repository contains 50k IMDB movie reviews, collected by [Maas et al. (2011)](https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf). The files have not been modified from their original form, except for the name of the file. Ambiguous reviews (original score is 5 or 6 out of 10) are not included.

### Splits
They are split into train (n=40k), validation (n=5k), and test (n=5k) with a ratio 8:1:1. Splits are evenly stratified with regard to length (number of whitespaces) and sentiment.

### Labels
The `_labels.csv` files have two columns:
 - `review`: path to the review text file. The ID has no purpose other than to distinguish reviews.
 - `sentiment`: 1 for positive and 0 for negative.

### Output format
A complete submission for this assignment will include a file called `test_predictions.csv` that is formatted like `*_labels.csv`. The `sentiment` column should only contain the integers 0 for negative and 1 for positive.
