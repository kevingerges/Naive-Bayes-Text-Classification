import os
import numpy as np
import pandas as pd
from collections import Counter
import string
import argparse

# this didn't work for me , so pretty useless
# it just didn't increase the evaluation score
# would love to know why it didn't work


# stopWords = set([
#     'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'and', 'or', 'but',
#     'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have', 'had', 'do',
#     'does', 'did', 'of', 'with', 'as', 'by', 'this', 'that', 'these', 'those',
#     'it', 'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'from',
#     'not', 'no', 'so', 'if', 'then', 'out', 'up', 'down', 'what', 'which',
#     'who', 'whom', 'why', 'how', 'when', 'where', 'all', 'any', 'some', 'more',
#     'most', 'other', 'such', 'only', 'own', 'same', 'can', 'will', 'just',
#     'should', 'now'
# ])



def tokenize(text):
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = text.split()
    return tokens

# prof said that it will take too long to run 10000 words, but thanks to m3 chips it's fine
def buildVocab(docs, vocabSize=10000):
    wordCounter = Counter()
    for doc in docs:
        tokens = tokenize(doc)
        wordCounter.update(tokens)
    mostCommonWords = wordCounter.most_common(vocabSize)
    vocab = [word for word, count in mostCommonWords]
    return vocab

def wordToIndex(vocab):
    # i wanna make an index for each word in the vocab hereee
    wordToIndex = {word: i for i, word in enumerate(sorted(vocab))}
    return wordToIndex

def textToFeatureVector(tokens, wordToIndex):
    featureVector = np.zeros(len(wordToIndex), dtype=int)
    for token in tokens:
        if token in wordToIndex:
            # was confused about this part, but it's just setting the index of the word to 1
            featureVector[wordToIndex[token]] = 1
    return featureVector



def trainNaiveBayes(featureVectors, labels, alpha=1):
    numberOfDocuments, vocabSize = featureVectors.shape
    classes = np.unique(labels)

    priorProbabilities = {}
    conditionalProbabilities = {}

    for cls in classes:
        # here i wanna find indices of documents belonging to the current class

        clsIndices = np.where(labels == cls)[0]
        numClsDocuments = len(clsIndices)

        #prior probability
        priorProbabilities[cls] = numClsDocuments / numberOfDocuments
        wordCounts = featureVectors[clsIndices].sum(axis=0)
        #Laplace
        smoothedWordCounts = wordCounts + alpha
        smoothedTotal = numClsDocuments + 2 * alpha


        conditionalProbabilities[cls] = smoothedWordCounts / smoothedTotal

    model = {
        'priorProbabilities': priorProbabilities,
        'conditionalProbabilities': conditionalProbabilities
    }

    return model

def predictNaiveBayes(model, featureVectors):
    numberOfDocuments = featureVectors.shape[0]
    classes = list(model['priorProbabilities'].keys())
    predictions = np.zeros(numberOfDocuments)

    #prior probabilities
    logPriorProbabilities = {}
    for cls, prob in model['priorProbabilities'].items():
        logPriorProbabilities[cls] = np.log(prob)

    logConditionalProbabilities = {}
    for cls, probs in model['conditionalProbabilities'].items():
        # here im just trying to avoid log(0) by clipping probabilities to be in [1e-10, 1-1e-10]
        # saw this online but i forgot where :( sorry for not citing
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        logProbs = np.log(probs)
        logComplementProbs = np.log(1.0 - probs)
        logConditionalProbabilities[cls] = {
            'present': logProbs,
            'absent': logComplementProbs
        }

    for i in range(numberOfDocuments):
        logProbabilities = {}

        for cls in classes:
            # ima start with the log prior probability
            logProbability = logPriorProbabilities[cls]

            # then i will add the log conditional probabilities for each word is present or absent
            logProbability += np.dot(featureVectors[i], logConditionalProbabilities[cls]['present'])
            logProbability += np.dot(1 - featureVectors[i], logConditionalProbabilities[cls]['absent'])


            logProbabilities[cls] = logProbability
        # then just pick the highest total log probability
        predictions[i] = max(logProbabilities, key=logProbabilities.get)

    return predictions

def accuracyScore(yTrue, yPred):
    correctPredictions = np.sum(yTrue == yPred)
    totalPredictions = len(yTrue)
    return correctPredictions / totalPredictions

def precisionScore(yTrue, yPred):
    tp = np.sum((yPred == 1) & (yTrue == 1))
    fp = np.sum((yPred == 1) & (yTrue == 0))
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recallScore(yTrue, yPred):
    tp = np.sum((yPred == 1) & (yTrue == 1))
    fn = np.sum((yPred == 0) & (yTrue == 1))
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1Score(yTrue, yPred):
    prec = precisionScore(yTrue, yPred)
    rec = recallScore(yTrue, yPred)
    if prec + rec == 0:
        return 0
    return 2 * (prec * rec) / (prec + rec)

def main():
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--data_src', type=str, required=True, help='Path to the data directory')
    args = parser.parse_args()

    dataDir = args.data_src

    # load training data
    trainLabelsDf = pd.read_csv(os.path.join(dataDir, 'train_labels.csv'))
    trainReviewPaths = trainLabelsDf['review'].values
    trainLabels = trainLabelsDf['sentiment'].values

    reviewTexts = []
    for reviewPath in trainReviewPaths:
        fullPath = os.path.join(dataDir, reviewPath)
        # read in from the review text file
        # i had to add encoding='utf-8' because it was giving me an error
        # i saw this on stackoverflow
        # but it's a werid set up of the data, i had to change the path to the data
        with open(fullPath, 'r', encoding='utf-8') as file:
            reviewText = file.read()
            reviewTexts.append(reviewText)

    #  build the vocab here
    vocab = buildVocab(reviewTexts, vocabSize=10000)
    wordToIdx = wordToIndex(vocab)

    # texts to binary feature vectors
    featureVectors = np.array([
        textToFeatureVector(tokenize(doc), wordToIdx) for doc in reviewTexts
    ])

    # trainningggggggg
    model = trainNaiveBayes(featureVectors, trainLabels, alpha=0.1)

    # validation data
    valLabelsDf = pd.read_csv(os.path.join(dataDir, 'val_labels.csv'))
    valReviewPaths = valLabelsDf['review'].values
    valLabels = valLabelsDf['sentiment'].values

    valTexts = []
    for reviewPath in valReviewPaths:
        fullPath = os.path.join(dataDir, reviewPath)
        # Read the validation review text file
        with open(fullPath, 'r', encoding='utf-8') as file:
            reviewText = file.read()
            valTexts.append(reviewText)

    # texts to binary feature vectors
    valFeatureVectors = np.array([
        textToFeatureVector(tokenize(doc), wordToIdx) for doc in valTexts
    ])

    # predictingggggg
    valPredictions = predictNaiveBayes(model, valFeatureVectors)
    valPredictions = valPredictions.astype(int)

    # Evaluatinggggggg
    acc = accuracyScore(valLabels, valPredictions)
    prec = precisionScore(valLabels, valPredictions)
    rec = recallScore(valLabels, valPredictions)
    f1 = f1Score(valLabels, valPredictions)

    print(f'Validation Results with alpha=0.1:')
    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('---')

    # csv files
    valPredictionsDf = pd.DataFrame({'prediction': valPredictions})
    valPredictionsDf.to_csv('val_predictions.csv', index=False)
    assert len(valPredictionsDf) == len(valLabelsDf), "Mismatch in number of validation predictions and labels"

    # loading againnn test data
    testReviewPathsDf = pd.read_csv(os.path.join(dataDir, 'test_paths.csv'), header=None, names=['review'])
    testReviewPaths = testReviewPathsDf['review'].values

    testTexts = []
    for reviewPath in testReviewPaths:
        normalizedPath = os.path.normpath(os.path.join(dataDir, reviewPath))
        # reading in test review text file
        with open(normalizedPath, 'r', encoding='utf-8') as file:
            reviewText = file.read()
            testTexts.append(reviewText)

    # texts to binary feature vectors
    testFeatureVectors = np.array([
        textToFeatureVector(tokenize(doc), wordToIdx) for doc in testTexts
    ])

    # predictinnggggg
    testPredictions = predictNaiveBayes(model, testFeatureVectors)
    testPredictions = testPredictions.astype(int)

    # csv filleee
    testPredictionsDf = pd.DataFrame({'prediction': testPredictions})
    testPredictionsDf.to_csv('test_predictions.csv', index=False)
    assert len(testPredictionsDf) == 5000, "Incorrect number of test predictions"

if __name__ == '__main__':
    main()