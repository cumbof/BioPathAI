#!/usr/bin/env python3

__authors__ = ( 'Fabio Cumbo (fabio.cumbo@unitn.it)' )
__version__ = '0.01'
__date__ = 'Jan 31, 2021'

import os, re, sys, copy
import argparse as ap
import pandas as pd
import numpy as np
import wittgenstein as lw
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score

'''
# Test Breast cancer pathway
python ripper.py --input Breast_cancer___Homo_sapiens__human___path_hsa05224___KEGG_.csv \
                 --output Breast_cancer___Homo_sapiens__human___path_hsa05224___KEGG_.txt \
                 --test_size            0.2 \
                 --folds                10  \
                 --cv_threshold         1.0 \
                 --min_accuracy         0.0 \
                 --min_accuracy_eval    0.8 \
                 --nproc                2   \
                 --verbose
'''

def read_params():
    p = ap.ArgumentParser( description = ( "The ripper.py script builds alternative rule-based classification models" ),
                           formatter_class = ap.ArgumentDefaultsHelpFormatter )
    p.add_argument( '--input', 
                    type = str,
                    help = "Input file matrix" )
    p.add_argument( '--test_size', 
                    type = float,
                    default = 0.2,
                    help = "Size of the test set in percentage" )
    p.add_argument( '--min_size', 
                    type = int,
                    default = 1,
                    help = "Minimum number of columns that a matrix must contain in order to be processed" )
    p.add_argument( '--max_size', 
                    type = int,
                    default = None,
                    help = "Maximum number of columns that a matrix must contain in order to be processed" )
    p.add_argument( '--folds', 
                    type = int,
                    default = 5,
                    help = "Number of folds for the cross-validation step" )
    p.add_argument( '--cv_threshold',
                    type = float,
                    default = 1.0,
                    help = "Coefficient of variation threshold" )
    p.add_argument( '--max_iterations', 
                    type = int,
                    default = None,
                    help = "Max number of iterations for producing alternative classification rules" )
    p.add_argument( '--min_accuracy', 
                    type = float,
                    default = 0.8,
                    help = "Stop iterating if the accuracy of the classification model is lower than --min_accuracy" )
    p.add_argument( '--min_accuracy_eval', 
                    type = float,
                    default = 0.8,
                    help = "Minimum accuracy used during the evaluation of the dataset" )
    p.add_argument( '--output', 
                    type = str,
                    default = None,
                    help = "Output file" )
    p.add_argument( '--nproc', 
                    type = int,
                    default = 1,
                    help = "Number of jobs to run in parallel for the cross-validation" )
    p.add_argument( '--verbose',
                    action = 'store_true',
                    default = False,
                    help = "Print results" )
    p.add_argument( '-v', 
                    '--version', 
                    action = 'version',
                    version = 'ripper.py version {} ({})'.format( __version__, __date__ ),
                    help = "Print the current ripper.py version and exit" )
    return p.parse_args()

def test_matrix(dataframe, test_size=.20, folds=5, nproc=1, verbose=False):
    # Split dataset into train and test sets
    train, test = train_test_split(dataframe, test_size=test_size, random_state=42)
    # Fixing folds
    class_labels = sorted(list(set(dataframe["class"])))
    samples_per_class = {}
    for class_label in class_labels:
        samples_per_class[class_label] = len(list(train[ train["class"] == class_label ].index))
    min_samples_in_class = min(samples_per_class.values())
    fixed_folds = args.folds
    while min_samples_in_class / fixed_folds < 1:
        fixed_folds -= 1
    if fixed_folds < folds:
        if verbose:
            print("\t\tNon enough samples. Reducing number of folds to {}".format( fixed_folds ))
    X_train, y_train = train.drop('class', axis=1), train['class']
    # Dummify our data to make sklearn happy
    X_train = pd.get_dummies(X_train, columns=X_train.select_dtypes('object').columns)
    y_train = y_train.map(lambda x: class_labels.index(x))
    # Init RIPPER classifier
    clf = lw.RIPPER()
    # Silence stderr to avoid package warning prints (temporary solution)
    # Redirect stderr to devnull
    sys.stderr = open(os.devnull, "w")  # silence stderr
    # Compute cross-validation scores
    cross_scores = cross_val_score(clf, X_train, y_train, cv=fixed_folds, n_jobs=nproc)
    # Unsilence stderr (temporary solution)
    sys.stderr = sys.__stderr__
    # Compute the mean of the cross-validation scores
    scores_mean = np.mean(cross_scores)
    # Compute the standard deviation of the cross-validation scores
    scores_stdev = np.std(cross_scores)
    # Compute the coefficient of variation of the cross-validation scores
    scores_coeff_variation = scores_stdev/scores_mean
    return train, test, cross_scores, scores_mean, scores_stdev, scores_coeff_variation, fixed_folds

def process_matrix(train_baseline, test_baseline, class_labels, out, 
                   class2exclude, min_size, max_size, iteration, prev_iter_cols, 
                   prev_accuracies, min_accuracy, verbose=False):
    if verbose:
        print("\t\tIteration {}".format(iteration))
    stops = 0
    accuracies = { }
    rules = { }
    for class_label in class_labels:
        train = copy.deepcopy(train_baseline)
        test = copy.deepcopy(test_baseline)
        if class_label in class2exclude:
            if class2exclude[class_label]:
                train.drop(class2exclude[class_label], axis=1, inplace=True)
                test.drop(class2exclude[class_label], axis=1, inplace=True)
        if len(list(train.columns)) < prev_iter_cols[class_label] or iteration == 1 or (prev_accuracies[class_label] >= min_accuracy and len(list(train.columns)) < prev_iter_cols[class_label]):
            prev_iter_cols[class_label] = len(list(train.columns))
            # Retrieve the number of columns
            cols = len(list(test.columns))
            if cols >= min_size and cols <= max_size:
                # Init RIPPER classifier
                clf = lw.RIPPER()
                if args.verbose:
                    print("\t\t\tColumns: {}".format(cols))
                with open(out, 'a+') as outfile:
                    outfile.write("# Iteration {}\n".format(iteration))
                    outfile.write("# Columns: {}\n".format(cols))
                if args.verbose:
                    print("\t\t\tSearching for rules for the class \"{}\"".format(class_label))
                # Silence stderr to avoid package warning prints (temporary solution)
                # Redirect stderr to devnull
                sys.stderr = open(os.devnull, "w")  # silence stderr
                # Fit with train subset
                clf.fit(train, class_feat='class', pos_class=class_label, random_state=42)
                # Take classes out of the test set
                X_test = test.drop('class', axis=1)
                y_test = test['class']
                rules_actors = [ ]
                with open(out, 'a+') as outfile:
                    outfile.write("# class: {}\n".format(class_label))
                    # Compute accuracy
                    accuracy = clf.score(X_test, y_test)
                    accuracies[class_label] = accuracy
                    if args.verbose:
                        print("\t\t\t\tAccuracy: {0:.0%}".format(accuracy))
                    outfile.write("# accuracy: {}\n".format( accuracy ))
                    # Compute precision
                    outfile.write("# precision: {}\n".format( clf.score(X_test, y_test, precision_score) ))
                    # Compute recall
                    outfile.write("# recall: {}\n".format( clf.score(X_test, y_test, recall_score) ))
                    # For each rule
                    concat_rule = ""
                    for idx, rule in enumerate(clf.ruleset_):
                        # Structure rule in a human readable format
                        human_readable_rule = "{}{}".format( str(rule).replace("[", "").replace("]", "").replace("^", " AND "), " OR" if idx<len(clf.ruleset_)-1 else "" )
                        concat_rule += human_readable_rule.strip()
                        actors = []
                        # Identify rules actors
                        for actor in human_readable_rule.split(" "):
                            actor = actor.strip()
                            if actor != "AND" and actor != "OR":
                                actors.append(actor.split("=")[0])
                        rules_actors.extend(actors)
                        outfile.write("{}\n".format( human_readable_rule ))
                        if args.verbose:
                            print("\t\t\t\t\t{}".format(human_readable_rule))
                    outfile.write("\n")
                    rules[class_label] = concat_rule
                # Unsilence stderr (temporary solution)
                sys.stderr = sys.__stderr__
                if class_label in class2exclude:
                    rules_actors.extend( class2exclude[class_label] )
                class2exclude[class_label] = list(set(rules_actors))
            else:
                stops += 1
        else:
            stops += 1
    stop_iterating = stops == len(class_labels)
    if stop_iterating:
        if args.verbose:
            print("\t\t\tNothing to do here")
    return class2exclude, stop_iterating, prev_iter_cols, accuracies, rules

if __name__ == '__main__':
    print( 'RIPPER v{} ({})'.format( __version__, __date__ ) )
    # Load command line parameters
    args = read_params()

    if os.path.exists(args.output):
        if args.verbose:
            print("Output file already exists")
        sys.exit(1)
    if not os.path.exists(args.input):
        if args.verbose:
            print("Input file does not exist")
        sys.exit(1)
    
    if args.verbose:
        print("Processing {}".format(args.input))
    # Load dataset as a pandas dataframe
    dataframe = pd.read_csv(args.input, index_col=0)
    # Retrieve the number of columns
    # Exclude "class" column
    cols = len(list(dataframe.columns)) - 1
    if not args.max_size:
        max_size = cols + 1
    if cols < args.min_size or cols > max_size:
        if args.verbose:
            print("Not enough columns")
        sys.exit(1)
    
    if args.verbose:
        print("\tDetected columns: {}".format(cols))
    # Cross-validate
    train, test, cross_scores, scores_mean, scores_stdev, scores_coeff_variation, folds = test_matrix(dataframe, 
                                                                                                      test_size=args.test_size, 
                                                                                                      folds=args.folds,
                                                                                                      nproc=args.nproc,
                                                                                                      verbose=args.verbose)
    if scores_coeff_variation >= args.cv_threshold:
        if args.verbose:
            print("Coefficient of variation >= {}".format(args.cv_threshold))
    
    class_labels = sorted(list(set(dataframe["class"])))
    # Proceed only if the coefficient of variation of the cross-validation scores
    # is lower than the coefficient of variation threshold defined above
    with open(args.output, 'a+') as outfile:
        outfile.write("# cross-validation scores ({}-folds)\n".format(folds))
        outfile.write("{}\n".format(", ".join([ str(score) for score in cross_scores ])))
        outfile.write("# scores mean: {}\n".format(scores_mean))
        outfile.write("# scores standard deviation: {}\n".format(scores_stdev))
        outfile.write("# scores coefficient of variation: {}\n".format(scores_coeff_variation))
        outfile.write("\n")    
    
    class2exclude = { }
    prev_iter_cols = { }
    prev_accuracies = { }
    for class_label in class_labels:
        prev_iter_cols[class_label] = cols
        prev_accuracies[class_label] = 100.0
    max_iterations = cols
    if args.max_iterations != None:
        max_iterations = args.max_iterations

    if args.verbose:
        print("\tRunning RIPPER")
    stats = { }
    iteration = 1
    stop = False
    while iteration <= max_iterations and not stop:
        class2exclude, stop, prev_iter_cols, accuracies, rules = process_matrix(train, test, 
                                                                                class_labels, args.output, class2exclude, 
                                                                                args.min_size, max_size, iteration, 
                                                                                prev_iter_cols, prev_accuracies, args.min_accuracy,
                                                                                verbose=args.verbose)
        stats[ iteration ] = { }
        for class_label in class_labels:
            if class_label in accuracies:
                prev_accuracies[class_label] = accuracies[class_label]
            stats[ iteration ][ class_label ] = {
                "accuracy": accuracies[ class_label ] if class_label in accuracies else None,
                "rules": rules[ class_label ] if class_label in rules else None
            }
        iteration += 1

    if args.verbose:
        print("\tEvaluating dataset")
    scores = { }
    for iteration in stats:
        for class_label in stats[ iteration ]:
            if class_label not in scores:
                scores[ class_label ] = { }
            if stats[ iteration ][ class_label ][ "rules" ]:
                rules = [ r.strip() for r in stats[ iteration ][ class_label ][ "rules" ].split( "OR" ) ]
                prev_iters = 1
                while prev_iters < iteration:
                    for rule in [ r.strip() for r in stats[ prev_iters ][ class_label ][ "rules" ].split( "OR" ) ]:
                        if rule in rules:
                            rules.remove( rule )
                    prev_iters += 1
                for rule in rules:
                    actors = [ actor.strip().split("=")[0] for actor in rule.split(" ") if actor.strip() != "AND" and actor.strip() != "OR" ]
                    actors = list(set(actors))
                    scores[ class_label ] = { **scores[ class_label ], 
                                              **{ actor: (1/cols)/int(iteration) 
                                                    if stats[ iteration ][ class_label ][ "accuracy" ] >= args.min_accuracy_eval else 0.0 
                                                        for actor in actors } }
    for class_label in scores:
        score = sum(scores[class_label].values())
        with open(args.output, 'a+') as outfile:
            outfile.write("# score for class \"{}\": {}\n".format(class_label, score))
        if args.verbose:
            print("\t\tScore for class \"{}\": {}".format(class_label, score))
