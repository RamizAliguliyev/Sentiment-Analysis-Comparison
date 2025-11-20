#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=1

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):
    """
    Reads all raw data files from disk and populates the dictionaries
    passed as arguments.
    - sentimentDictionary: Fills with words and their scores (1 or -1).
    - sentencesTrain: Fills with ~90% of film reviews for training.
    - sentencesTest: Fills with ~10% of film reviews for testing.
    - sentencesNokia: Fills with all Nokia reviews for out-of-domain testing.
    """

   # --- 1. Read movie review files ---
    # reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    # --- 2. Read Nokia review files ---
    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
    # --- 3. Read sentiment dictionary files ---
    
    # Use a list comprehension to read positive words.
    # .strip() removes whitespace/newlines.
    # 'if line.strip()' filters out empty blank lines.
    # 'not line.startswith(";")' filters out comment lines.
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = [line.strip() for line in posDictionary.readlines() if line.strip() and not line.startswith(';')]
    posDictionary.close()

    # Do the same for the negative words list.
    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = [line.strip() for line in negDictionary.readlines() if line.strip() and not line.startswith(';')] 
    negDictionary.close()

    # --- 4. Populate the master sentiment dictionary with scores ---
    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # --- 5. Create Training and Test Datsets for Film Reviews ---
    # We want to test on sentences we haven't trained on, 
    # to see how well the model generalises to previously unseen sentences.
    
    # Create ~90% training / 10% test split of training and test data
    for i in posSentences:
        # random.randint(1,10) picks a number between 1 and 10.
        # '< 2' is true 1 out of 10 times (i.e., when it picks '1').
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

   # --- 6. Create Nokia (Out-of-Domain) Datset ---
    # No split needed, this is only for testing.
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

# calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    """
    Trains the Naive Bayes model.
    This function iterates over the training data to count word occurrences
    and then calculates the conditional probabilities P(Word|Positive)
    and P(Word|Negative) for every word in the vocabulary.
    
    Args:
        sentencesTrain (dict): The training data (sentence: sentiment).
        pWordPos (dict): An empty dict to be filled with P(W|Pos) probabilities.
        pWordNeg (dict): An empty dict to be filled with P(W|Neg) probabilities.
        pWord (dict): An empty dict to be filled with P(W) probabilities.
    """
    # Dictionaries to store the raw counts of each word  [hash function]
    freqPositive = {} 
    freqNegative = {}
    # A set of all unique words in the training data (our vocabulary)
    dictionary = {}
    # Counters for the total number of words in each class
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    # --- STAGE 1: COUNTING ---
    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        # Tokenize the sentence into a list of words
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: # calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                # keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

# implement naive bayes algorithm
# INPUTS:
#   sentencesTest is a dictonary with sentences associated with sentiment 
#   dataName is a string (used only for printing output)
#   pWordPos is dictionary storing p(word|positive) for each word
#      i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#   pWordNeg is dictionary storing p(word|negative) for each word
#   pWord is dictionary storing p(word)
#   pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):

    print("Naive Bayes classification for " + dataName)
    pNeg=1-pPos

    # These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: # calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0; 
        # Normalize the scores into a 0-1 probability
        # prob = P(Pos|W) / (P(Pos|W) + P(Neg|W))           
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)

        total+=1
        # 'sentiment' is the "ground truth" (correct answer)
        if sentiment=="positive":
            totalpos+=1
            # 'prob > 0.5' is the model's prediction
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:# Ground truth is negative
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
 
 
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
    # Accuracy = (Correct Predictions) / (Total Predictions)
    accuracy = (correct / float(total)) * 100 if total >0 else 0

    # --- Positive Class Metrics ---

    # Precision (Positive): TP / (TP + FP) -> correctpos / totalpospred
    # "Of all sentences we PREDICTED positive, how many were ACTUALLY positive?"
    pos_precision = (correctpos / float(totalpospred)) * 100 if totalpospred >0 else 0

    # Recall (Positive): TP / (TP + FN) -> correctpos / totalpos
    # "Of all ACTUALLY positive sentences, how many did we FIND?"
    pos_recall = (correctpos / float(totalpos)) * 100 if totalpos >0 else 0

    # F1-Score (Positive): 2 * (Precision * Recall) / (Precision + Recall)
    # The harmonic mean, balancing precision and recall. Good for uneven classes.
    pos_f1 = (2 * pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) >0 else 0

    # --- Negative Class Metrics ---
    
    # Precision (Negative): TN / (TN + FN) -> correctneg / totalnegpred
    # "Of all sentences we PREDICTED negative, how many were ACTUALLY negative?"
    neg_precision = (correctneg / float(totalnegpred)) * 100 if totalnegpred > 0 else 0

    # Recall (Negative): TN / (TN + FP) -> correctneg / totalneg
    neg_recall = (correctneg / float(totalneg)) * 100 if totalneg > 0 else 0

    # F1-Score (Negative): 2 * (Precision * Recall) / (Precision + Recall)
    neg_f1 = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0

    # --- Print all the results in a clean format ---
    print(f"\n--- Results for {dataName} ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("\n--- Positive Class ---")
    print(f"Precision: {pos_precision:.2f}% ({correctpos}/{totalpospred})")
    print(f"Recall:    {pos_recall:.2f}% ({correctpos}/{totalpos})")
    print(f"F1-Score:  {pos_f1:.2f}")
    print("\n--- Negative Class ---")
    print(f"Precision: {neg_precision:.2f}% ({correctneg}/{totalnegpred})")
    print(f"Recall:    {neg_recall:.2f}% ({correctneg}/{totalneg})")
    print(f"F1-Score:  {neg_f1:.2f}")
    print("--------------------------------\n")
    # --- END of Step 2 TODO ---
# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):

    print("Dictionary-based classification")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # Iterate over each sentence and its correct label
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0

        # Sum the scores of all known words in the sentence
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1

        # --- Tallying Results ---
        # 'sentiment' is the correct answer
        if sentiment=="positive":
            totalpos+=1
            # 'score >= threshold' is the model's prediction
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:# Predict negative
                correct+=0
                totalnegpred+=1
        else:
            totalneg+=1
            # STEP 6: Check for printing errors
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1


# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
     # Accuracy = (Correct Predictions) / (Total Predictions)
    accuracy = (correct / float(total)) * 100 if total > 0 else 0
    
    # --- Positive Class Metrics ---
    # Precision (Positive): TP / (TP + FP) -> correctpos / totalpos
    pos_precision = (correctpos / float(totalpospred)) * 100 if totalpospred > 0 else 0

    # Recall (Positive): TP / (TP + FN) -> correctpos / totalpos
    pos_recall = (correctpos / float(totalpos)) * 100 if totalpos > 0 else 0

    # F1-Score (Positive): 2 * (Precision * Recall) / (Precision + Recall)
    pos_f1 = (2 * pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0
    
    # --- Negative Class Metrics ---
    # Precision (Negative): TN / (TN + FN) -> correctneg / totalneg
    neg_precision = (correctneg / float(totalnegpred)) * 100 if totalnegpred > 0 else 0

    # Recall (Negative): TN / (TN + FP) -> correctneg / totalneg
    neg_recall = (correctneg / float(totalneg)) * 100 if totalneg > 0 else 0

    # F1-Score (Negative): 2 * (Precision * Recall) / (Precision + Recall)
    neg_f1 = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0

    # --- Print all the results in a clean format ---
    print(f"\n--- Results for {dataName} ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("\n--- Positive Class ---")
    print(f"Precision: {pos_precision:.2f}% ({correctpos}/{totalpospred})")
    print(f"Recall:    {pos_recall:.2f}% ({correctpos}/{totalpos})")
    print(f"F1-Score:  {pos_f1:.2f}")
    print("\n--- Negative Class ---")
    print(f"Precision: {neg_precision:.2f}% ({correctneg}/{totalnegpred})")
    print(f"Recall:    {neg_recall:.2f}% ({correctneg}/{totalneg})")
    print(f"F1-Score:  {neg_f1:.2f}")
    print("--------------------------------\n")
    # --- END of Step 5 TODO ---

#For step 5.3: Improved dictionary-based classifier with negation handling
#Negotiation handling: If a negation word is found, invert the sentiment scores of the next three words.
def testDictionaryImproved(sentencesTest, dataName, sentimentDictionary, threshold):
    """
    Performs rule-based classification (like testDictionary), but includes
    a simple negation-handling rule.
    
    This function extends the base classifier by adding a "negation window".
    When a negation word (e..g, "not", "n't") is found, the sentiment
    score of the next N words (e.g., 3) is inverted. This is a form of
    linguistic generalization, as required by Step 5.3.
    """
    print(f"IMPROVED Dictionary-based classification for: {dataName}")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    
    # Define negation words
    negation_words = {"not", "n't", "no", "never", "cannot", "can't", "don't", "doesn't", "didn't"}

    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence.lower()) # process in lowercase
        score=0
        negation_window = 0 # How many words forward the negation applies

        for word in Words:
            if word in negation_words:
                negation_window = 3 # Apply negation to next 3 words
            
            if word in sentimentDictionary:
                word_score = sentimentDictionary[word]
                if negation_window > 0:
                    word_score = -word_score # Invert the score
                score += word_score
            
            # Decrement the window
            if negation_window > 0:
                negation_window -= 1
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print (f"ERROR (pos classed as neg, score {score}): {sentence}")
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print (f"ERROR (neg classed as pos, score {score}): {sentence}") 

    # --- Metrics calculations ---

    # Accuracy = (Correct Predictions) / (Total Predictions)
    accuracy = (correct / float(total)) * 100 if total > 0 else 0

    # --- Positive Class Metrics ---
    # Precision (Positive): TP / (TP + FP) -> correctpos / totalpos
    pos_precision = (correctpos / float(totalpospred)) * 100 if totalpospred > 0 else 0

    # Recall (Positive): TP / (TP + FN) -> correctpos / totalpos
    pos_recall = (correctpos / float(totalpos)) * 100 if totalpos > 0 else 0

    # F1-Score (Positive): 2 * (Precision * Recall) / (Precision + Recall)
    pos_f1 = (2 * pos_precision * pos_recall) / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0
    
    # --- Negative Class Metrics ---
    # Precision (Negative): TN / (TN + FN) -> correctneg / totalneg
    neg_precision = (correctneg / float(totalnegpred)) * 100 if totalnegpred > 0 else 0

    # Recall (Negative): TN / (TN + FP) -> correctneg / totalneg
    neg_recall = (correctneg / float(totalneg)) * 100 if totalneg > 0 else 0

    # F1-Score (Negative): 2 * (Precision * Recall) / (Precision + Recall)
    neg_f1 = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall) if (neg_precision + neg_recall) > 0 else 0

    # --- Print all the results in a clean format ---
    print(f"\n--- Results for {dataName} ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("\n--- Positive Class ---")
    print(f"Precision: {pos_precision:.2f}% ({correctpos}/{totalpospred})")
    print(f"Recall:    {pos_recall:.2f}% ({correctpos}/{totalpos})")
    print(f"F1-Score:  {pos_f1:.2f}")
    print("\n--- Negative Class ---")
    print(f"Precision: {neg_precision:.2f}% ({correctneg}/{totalnegpred})")
    print(f"Recall:    {neg_recall:.2f}% ({correctneg}/{totalneg})")
    print(f"F1-Score:  {neg_f1:.2f}")
    print("--------------------------------\n")
# --- END of Step 5.3 ---
# Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower[word]=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)




#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

#run naive bayes classifier on datasets
#testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
#testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)



# run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)

print("\nRUNNING IMPROVED RULE-BASED CLASSIFIER (STEP 5.3)\n")
testDictionaryImproved(sentencesTrain,  "Films (Train Data, Improved Rule-Based)\t", sentimentDictionary, 1)
#testDictionaryImproved(sentencesTest,  "Films  (Test Data, Improved Rule-Based)\t",  sentimentDictionary, 1)
#testDictionaryImproved(sentencesNokia, "Nokia   (All Data, Improved Rule-Based)\t",  sentimentDictionary, 1)

# print most useful words
#mostUseful(pWordPos, pWordNeg, pWord, 100)