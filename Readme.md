Naive Bayes Sentiment Classifier

Introduction

This project implements a Naive Bayes classifier for sentiment analysis on IMDb movie reviews. The classifier uses a bag-of-words model with binary features indicating the presence or absence of words in a review. The model is trained on labeled data and evaluated on a validation set. Predictions are also generated for a test set.



How to Run

Create a Virtual Environment
```angular2html
python3 -m venv myenv
```

Activate the Virtual Environment
```angular2html
source myenv/bin/activate
```
Install Dependencies in the Virtual Environment
```angular2html
pip install numpy pandas
```
run 
```angular2html
python main.py --data_src /Users/kevingerges/Desktop/School/csci444/hw1-imdb/code/

```
if not working try 
```angular2html
python3 main.py --data_src /Users/kevingerges/Desktop/School/csci444/hw1-imdb/code/
 
or
/Users/kevingerges/Desktop/School/csci444/hw1-imdb/code/myenv/bin/python main.py --data_src /Users/kevingerges/Desktop/School/csci444/hw1-imdb/code/\n

replace the path with the first with your pwd

```

Dependencies
This project requires Python 3.x and the following Python packages:
```angular2html
NumPy
Pandas

```

Installation
You can install the required packages using pip:
```angular2html
pip install -r requirements.txt

```

Output Files
The script will generate the following output files in the current directory:

val_predictions.csv: Contains the predicted labels for the validation set.
test_predictions.csv: Contains the predicted labels for the test set.
Both files are formatted as CSV files with a single column named prediction, including a header.


Contact Information
For any questions or issues, please contact:

Name: Kevin Gerges
Email: kgerges@usc.edu
