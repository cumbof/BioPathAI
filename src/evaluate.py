#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)',
                'Giovanni Felici (giovanni.felici@iasi.cnr.it)' )
__version__ = '0.01'
__date__ = 'Feb 15, 2021'

import sys, os
import argparse as ap
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, classification_report

def read_params():
    p = ap.ArgumentParser( description = ( "The evaluate.py script tests multiple classification algorithms over an input data matrix" ),
                           formatter_class = ap.ArgumentDefaultsHelpFormatter )
    p.add_argument( '--input', 
                    type = str,
                    help = "Input file matrix" )
    p.add_argument( '--folds', 
                    type = int,
                    default = 5,
                    help = "Number of folds for the cross-validation step" )
    p.add_argument( '--classifiers', 
                    type = str,
                    default = None,
                    help = "List of comma separated classifiers (case sensitive)" )
    p.add_argument( '--output_prefix', 
                    type = str,
                    default = None,
                    help = "Output file prefix" )
    p.add_argument( '--verbose',
                    action = 'store_true',
                    default = False,
                    help = "Print results" )
    p.add_argument( '-v', 
                    '--version', 
                    action = 'version',
                    version = 'evaluate.py version {} ({})'.format( __version__, __date__ ),
                    help = "Print the current evaluate.py version and exit" )
    return p.parse_args()

SEED = 0

NAMES = ["Logistic Regression", "Linear Discriminant Analysis", "Nearest Neighbors", "Linear SVM", "RBF SVM", 
         "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]

CLASSIFIERS = [
    LogisticRegression(random_state=SEED),                                                     # Logistic Regression
    LinearDiscriminantAnalysis(),                                                              # Linear Discriminant Analysis
    KNeighborsClassifier(3),                                                                   # Nearest Neighbors
    SVC(kernel="linear", C=0.025, random_state=SEED),                                          # Linear SVM
    SVC(gamma=2, C=1, random_state=SEED),                                                      # RBF SVM
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=SEED),                              # Gaussian Process
    DecisionTreeClassifier(max_depth=5, random_state=SEED),                                    # Decision Tree
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=SEED),   # Random Forest
    MLPClassifier(alpha=1, max_iter=1000, random_state=SEED),                                  # Neural Net
    AdaBoostClassifier(random_state=SEED),                                                     # AdaBoost
    GaussianNB(),                                                                              # Naive Bayes
    QuadraticDiscriminantAnalysis()                                                            # QDA
]

def process(dataframe, classifiers, folds):
    # Build matrix content X and classes vector y
    X, y = dataframe.drop('Class', axis=1), dataframe['Class']
    # Dummify our data to make sklearn happy
    labels = list(set(y))
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns)
    y = y.map(lambda x: labels.index(x))
    folds = 10
    # Manually implement cross-validation
    X_folds = np.array_split(X, folds)
    y_folds = np.array_split(y, folds)
    matrix = dict()
    for k in range(folds):
        # Use 'list' to copy, in order to 'pop' later on
        X_train = list(X_folds)
        X_test  = X_train.pop(k)
        X_train = np.concatenate(X_train)
        y_train = list(y_folds)
        y_test  = y_train.pop(k)
        y_train = np.concatenate(y_train)
        # Define classifier and fit dataset
        for classifier in NAMES:
            if classifier not in matrix:
                matrix[classifier] = dict()
            clf = CLASSIFIERS[ NAMES.index(classifier) ]
            clf.fit(X_train, y_train)
            # Get misclassified samples
            prediction = clf.predict(X_test)
            y_test = np.asarray(y_test)
            samples = list(X_test.index)
            for sample in samples:
                misclassified = int( y_test[samples.index(sample)] != prediction[samples.index(sample)] )
                if sample not in matrix[classifier]:
                    matrix[classifier][sample] = list()
                matrix[classifier][sample].append(misclassified)
    predictions = dict()
    for classifier in matrix:
        samples = sorted(list(matrix[classifier]))
        Y = list()
        Y_pred = list()
        for sample in samples:
            Y.append( y[ list(X.index).index( sample ) ] )
            Y_pred.append( max(set(matrix[classifier][sample]), key=matrix[classifier][sample].count) )
        predictions[ classifier ] = ( Y, Y_pred )
    return matrix, predictions, labels

if __name__ == '__main__':
    print( 'Evaluate v{} ({})'.format( __version__, __date__ ) )
    # Load command line parameters
    args = read_params()

    # Check if args.input does not exist
    if not os.path.exists(args.input):
        if args.verbose:
            print("Input file does not exist")
        sys.exit(1)

    # Use supported classifiers only
    classifiers = args.classifiers.strip().split(",") if args.classifiers else NAMES
    classifiers = list(set(classifiers).intersection(set(NAMES)))
    if not classifiers:
        if args.verbose:
            print("Unknown classifier")
        sys.exit(1)

    # Check if outputs already exist
    for output_suffix in [ '__evaluate.csv', '__confusion.csv' ]:
        if os.path.exists('{}{}'.format(args.output_prefix, output_suffix)):
            if args.verbose:
                print("Output file already exists")
            sys.exit(1)

    # Process the args.input matrix
    if args.verbose:
        print("Building evaluation matrix")
        print("\tInput: {}".format(args.input))
    # Load matrix as pandas dataframe
    dataframe = pd.read_csv(args.input, index_col=0)
    # Fix class column name
    dataframe.set_axis([*dataframe.columns[:-1], 'Class'], axis=1, inplace=True)
    evaluation_matrix, predictions, labels = process(dataframe, classifiers, args.folds)

    if args.verbose:
        print('Dumping evaluation and confusion matrices')
        print('\t{}__evaluation.csv'.format(args.output_prefix))
        print('\t{}__confusion.csv'.format(args.output_prefix))
    # Retrieve processed samples
    samples = set(evaluation_matrix[list(evaluation_matrix.keys())[0]].keys())
    # Dump evaluation matrix
    with open('{}__evaluation.csv'.format(args.output_prefix), "w+") as outfile:
        # Write header line
        outfile.write("SampleID,{},Class\n".format(",".join(classifiers)))
        for sample in samples:
            outfile.write(sample)
            for clf in classifiers:
                outfile.write(",{}".format(max(set(evaluation_matrix[clf][sample]), key=evaluation_matrix[clf][sample].count)))
            rowData = dataframe.loc[sample, :]
            outfile.write(",{}\n".format(rowData.Class))
    # Dump confusion matrices
    with open('{}__confusion.csv'.format(args.output_prefix), "w+") as outfile:
        # Write header line
        header = ""
        for class_label in labels:
            header += ',{} (predicted)'.format(class_label)
        outfile.write("#{},Algorithm\n".format(header))
        for classifier in predictions:
            Y, Y_pred = predictions[classifier]
            conf = confusion_matrix(Y, Y_pred)
            if args.verbose:
                print(classifier)
                print(classification_report(Y, Y_pred, target_names=labels, digits=3))
            for idx, val in enumerate(labels):
                content = ','.join( [ str(v) for v in conf[idx] ] )
                outfile.write('{} (true),{},{}\n'.format( val, content, classifier ))
