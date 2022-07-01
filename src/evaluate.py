#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)',
                'Giovanni Felici (giovanni.felici@iasi.cnr.it)' )
__version__ = '0.01'
__date__ = 'Jun 07, 2021'

import sys, os, time
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
from sklearn.model_selection import train_test_split, KFold, cross_val_predict

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
    p.add_argument( '--nproc', 
                    type = int,
                    default = 1,
                    help = "Number of parallel jobs" )
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

NAMES = [
    "Nearest Neighbors K=5", 

    #"SVM Linear C=0.05", 
    #"SVM Linear C=1",
    #"SVM Linear C=3",

    #"SVM Poly gamma=2 C=0.05",
    #"SVM Poly gamma=2 C=1",
    #"SVM Poly gamma=2 C=3",

    #"SVM Poly gamma=3 C=0.05",
    #"SVM Poly gamma=3 C=1",
    #"SVM Poly gamma=3 C=3",

    "SVM RBF C=0.05",
    "SVM RBF C=1",
    "SVM RBF C=3",

    "Gaussian Process", 
    
    "Decision Tree min_samples_split=2",
    "Decision Tree min_samples_split=5",
    "Decision Tree min_samples_split=10",
    "Decision Tree min_samples_split=20",

    "Random Forest min_samples_split=2 max_features=auto", 
    "Random Forest min_samples_split=5 max_features=auto",
    "Random Forest min_samples_split=10 max_features=auto",
    "Random Forest min_samples_split=2 max_features=1",
    "Random Forest min_samples_split=5 max_features=1",
    "Random Forest min_samples_split=10 max_features=1",
    
    "Neural Net (10, 10, 10, 10)",
    "Neural Net (20, 20, 20, 20)",
    "Neural Net (30, 30, 30, 30)",
    "Neural Net (30, 50, 50, 30)"
]

CLASSIFIERS = [    
    KNeighborsClassifier(5),                                                                                  # Nearest Neighbors
    
    #SVC(kernel="linear",        C=0.05, random_state=SEED),                                                   # Linear SVM
    #SVC(kernel="linear",        C=1,    random_state=SEED),                                                   # Linear SVM
    #SVC(kernel="linear",        C=3,    random_state=SEED),                                                   # Linear SVM
    
    #SVC(kernel="poly", gamma=2, C=0.05, random_state=SEED),                                                   # RBF SVM
    #SVC(kernel="poly", gamma=2, C=1,    random_state=SEED),                                                   # RBF SVM
    #SVC(kernel="poly", gamma=2, C=3,    random_state=SEED),                                                   # RBF SVM
    
    #SVC(kernel="poly", gamma=3, C=0.05, random_state=SEED),                                                   # RBF SVM
    #SVC(kernel="poly", gamma=3, C=1,    random_state=SEED),                                                   # RBF SVM
    #SVC(kernel="poly", gamma=3, C=3,    random_state=SEED),                                                   # RBF SVM
    
    SVC(kernel="rbf",           C=0.05, random_state=SEED),                                                   # RBF SVM
    SVC(kernel="rbf",           C=1,    random_state=SEED),                                                   # RBF SVM
    SVC(kernel="rbf",           C=3,    random_state=SEED),                                                   # RBF SVM
    
    GaussianProcessClassifier(1.0 * RBF(1.0), random_state=SEED),                                             # Gaussian Process
    
    DecisionTreeClassifier(min_samples_split=2,  random_state=SEED),                                          # Decision Tree
    DecisionTreeClassifier(min_samples_split=5,  random_state=SEED),                                          # Decision Tree
    DecisionTreeClassifier(min_samples_split=10, random_state=SEED),                                          # Decision Tree
    DecisionTreeClassifier(min_samples_split=20, random_state=SEED),                                          # Decision Tree
    
    RandomForestClassifier(min_samples_split=2,  n_estimators=100, max_features="auto", random_state=SEED),   # Random Forest
    RandomForestClassifier(min_samples_split=5,  n_estimators=100, max_features="auto", random_state=SEED),   # Random Forest
    RandomForestClassifier(min_samples_split=10, n_estimators=100, max_features="auto", random_state=SEED),   # Random Forest
    RandomForestClassifier(min_samples_split=2,  n_estimators=100, max_features=1,      random_state=SEED),   # Random Forest
    RandomForestClassifier(min_samples_split=5,  n_estimators=100, max_features=1,      random_state=SEED),   # Random Forest
    RandomForestClassifier(min_samples_split=10, n_estimators=100, max_features=1,      random_state=SEED),   # Random Forest
    
    MLPClassifier(hidden_layer_sizes=(10,10,10,10), max_iter=1000, random_state=SEED),                        # Neural Net
    MLPClassifier(hidden_layer_sizes=(20,20,20,20), max_iter=1000, random_state=SEED),                        # Neural Net
    MLPClassifier(hidden_layer_sizes=(30,30,30,30), max_iter=1000, random_state=SEED),                        # Neural Net
    MLPClassifier(hidden_layer_sizes=(30,50,50,30), max_iter=1000, random_state=SEED)                         # Neural Net
]

if __name__ == '__main__':
    # Load command line parameters
    args = read_params()
    if args.verbose:
        print( 'Evaluate v{} ({})'.format( __version__, __date__ ) )

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
    pathsize = len(list(dataframe.columns))-1 # Exclude the 'Class' column
    if args.verbose:
        print("\tPathway Size: {}".format(pathsize))
    
    # Select a set of algorithms
    models = dict()
    selected_algorithms = list()
    for n in range(len(NAMES)):
        models[NAMES[n]] = CLASSIFIERS[n]
        selected_algorithms.append(NAMES[n])

    # Build matrix content X and classes vector y
    x_df, y_df = dataframe.drop('Class', axis=1), dataframe['Class']
    # Dummify our data to make sklearn happy
    labels = sorted(list(set(y_df)))
    x = pd.get_dummies(x_df, columns=x_df.select_dtypes('object').columns)
    y = y_df.map(lambda v: labels.index(v))
    samples_profiles_real = dict()
    samples_list_sorted = list(y.index)
    for sample, c in zip(list(y.index), list(y.values)):
        samples_profiles_real[sample] = c

    # Init confusion
    with open('{}__{}__confusion.csv'.format(args.output_prefix, pathsize), "a+") as outfile:
        outfile.write("# Pathway size: {}\n".format(pathsize))
        outfile.write("# {} (predicted),{} (predicted),Algorithm,Time (seconds)\n".format(labels[0], labels[1]))

    if args.verbose:
        print("\tRunning algorithms:")
    sample_profiles_predicted = dict()
    for name in selected_algorithms:
        model = models[name]
        if args.verbose:
            print("\t\t{}".format(name))
        kfold = KFold(n_splits=args.folds)
        t0 = time.time()
        y_pred = cross_val_predict(model, x, y, cv=kfold, n_jobs=args.nproc)
        t1 = time.time()
        for i in range(len(y_pred)):
            sample = samples_list_sorted[i]
            if sample not in sample_profiles_predicted:
                sample_profiles_predicted[sample] = list()
            sample_profiles_predicted[sample].append( "0" if y_pred[i] == samples_profiles_real[sample] else "1" )
        conf_mat = confusion_matrix(y, y_pred, labels=list(range(len(labels))))
        # Dump confusion matrix
        with open('{}__{}__confusion.csv'.format(args.output_prefix, pathsize), "a+") as outfile:
            row_count = 0
            for row in conf_mat:
                outfile.write("{} (true),{},{},{},{}\n".format(labels[row_count], row[0], row[1], name, float(t1-t0)))
                row_count += 1
    
    # Dump evaluation matrix
    with open('{}__{}__evaluate.csv'.format(args.output_prefix, pathsize), "a+") as outfile:
        outfile.write("# Pathway size: {}\n".format(pathsize))
        outfile.write("# Sample,{}\n".format(",".join(selected_algorithms)))
        for sample in sample_profiles_predicted:
            outfile.write("{},{}\n".format(sample, ",".join(sample_profiles_predicted[sample])))
