# Assignment 2 - Text classification benchmarks

## 1. Contributions
This code was written independently by me. 
## 2. Assignment description by instructor
This is the repository for the first assignment in the course Language Analytics from the bachelors elective course Cultural Data Science at Aarhus University

This assignment is about using ```scikit-learn``` to train simple (binary) classification models on text data. For this assignment, we'll continue to use the Fake News Dataset that we've been working on in class.

For this exercise, you should write *two different scripts*. One script should train a logistic regression classifier on the data; the second script should train a neural network on the same dataset. Both scripts should do the following:

- Be executed from the command line
- Save the classification report to the folder called ```out```
- Save the trained models and vectorizers to the folder called ```models```

## 2.1 Objective

This assignment is designed to test that you can:

1. Train simple benchmark machine learning classifiers on structured text data;
2. Produce understandable outputs and trained models which can be reused;
3. Save those results in a clear way which can be shared or used for future analysis

## 2.2 Some notes

- Saving the classification report to a text file can be a little tricky. You will need to Google this part!
- You might want to challenge yourself to create a third script which vectorizes the data separately, and saves the new feature extracted dataset. That way, you only have to vectorize the data once in total, instead of once per script. Performance boost!

## 2.3 Additional comments

Your code should include functions that you have written wherever possible. Try to break your code down into smaller self-contained parts, rather than having it as one long set of instructions.

For this assignment, you are welcome to submit your code either as a Jupyter Notebook, or as ```.py``` script. If you do not know how to write ```.py``` scripts, don't worry - we're working towards that!

Lastly, you are welcome to edit this README file to contain whatever informatio you like. Remember - documentation is important!

## 3. Methods


## 4. Usage
The script is written in python 3.10.7. Make sure this is the python version you have installed on your device.
From the command line clone this GitHub repository to your current location (this can be changed with the `cd path/to/desired/location`) by running the command `git clone https://github.com/NiGitaMyrGit/Lang_assignment1.git".` 
### 4.1 install required packages
From the command line, located in the main directory, run the command `bash setup.sh`. This will install the packages found in the ```requirements.txt``` file, which is required to run the script.

### 4.2 Get the data 
The data is located in the folder ```in```. THe data
### 4.3 run the script 

## 5. Results
The models and vectorizers are saved in the folder ```models```
The **logistic regression model** is saved as ```Logreg_model.joblib```
and the **neural network model** as ```NN_model.joblib```. 
The **logistic regression vectorizer** is saved as ```Count_vectorizer```
while the **neural network vectorizer** is saved as ```TF-IDF_vectorizer.joblib```

The **classification reports** are saved in the folder ```out```
The logistics regression classifier is saved as ```log_reg_classification_report.txt```
and the neural network classifier as ```NN_classification_report.txt```


**Classification report for the Logical Regression classifier**
              precision    recall  f1-score   support

        FAKE       0.79      0.86      0.83       628
        REAL       0.85      0.78      0.81       639

    accuracy                           0.82      1267
   macro avg       0.82      0.82      0.82      1267
weighted avg       0.82      0.82      0.82      1267

**Classification report for the Neural Network classifier**

              precision    recall  f1-score   support

        FAKE       0.90      0.87      0.88       628
        REAL       0.87      0.90      0.89       639

    accuracy                           0.89      1267
   macro avg       0.89      0.89      0.89      1267
weighted avg       0.89      0.89      0.89      1267

The accuracy of the logical regression classifier is 0.82 and for the neural network 0.89. The Neural Network is performing better than the logical regression, which can ebe explained by the fact that the logistic regression classifier assumes a linear relationship between the features while a neural network is better at learning more complex and non-linear relationship between the text-data. 
