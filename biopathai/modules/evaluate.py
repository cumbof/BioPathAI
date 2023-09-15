"""Run multiple machine learning algorithms over an input dataset in cross validation and report the confusion matrices.
"""

import errno
import os
import time
from io import StringIO
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, cross_val_predict

SUPPORTED_CLASSIFIERS = [
    "decisiontree",
    "gaussianprocess",
    "nearestneighbors",
    "neuralnet",
    "randomforest",
    "svm"
]


def get_classifier(
    classifier_name: str,
    min_samples_split: int = 5,
    gaussian_process_val: float = 1.0,
    gaussian_process_rbf: float = 1.0,
    k_neighbors: int = 5,
    hidden_layer_sizes: Tuple[int] = (30, 50, 50, 30),
    max_iter: int = 1000,
    n_estimators: int = 100,
    max_features: Union[str, int] = "auto",
    svc_kernel: str = "rbf",
    svc_c: int = 1,
    seed: int = 0
):
    """
    Get a sklearn classifier

    :param classifier_name:     Name of the classifier
    :param seed:                Set a seed for reproducibility
    :return:                    The classifier object
    """

    if classifier_name not in SUPPORTED_CLASSIFIERS:
        raise ValueError("Unable to find \"{}\" in sklearn".format(classifier_name))

    # TODO Add classifiers' arguments

    if classifier_name.lower() == "decisiontree":
        return DecisionTreeClassifier(
            min_samples_split=min_samples_split,
            random_state=seed
        )

    elif classifier_name.lower() == "gaussianprocess":
        return GaussianProcessClassifier(
            gaussian_process_val * RBF(gaussian_process_rbf),
            random_state=seed
        )

    elif classifier_name.lower() == "nearestneighbors":
        return KNeighborsClassifier(k_neighbors)

    elif classifier_name.lower() == "neuralnet":
        return MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=seed
        )

    elif classifier_name.lower() == "randomforest":
        return RandomForestClassifier(
            min_samples_split=min_samples_split,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=seed
        )

    elif classifier_name.lower() == "svm":
        return SVC(kernel=svc_kernel, C=svc_c, random_state=seed)


def evaluate(
    input_id: str,
    input_file: Union[str, StringIO],
    output_prefix: str,
    classifiers: Optional[List[str]] = None,
    folds: int = 5,
    in_memory: bool = False,
    nproc: int = 1,
    verbose: bool = False
) -> Tuple[str, Union[Dict[str, Dict[str, str]], Dict[str, Dict[str, StringIO]]], Dict[str, float]]:
    """Run multiple machine learning algorithms over an input dataset in cross validation
    and report the confusion matrices

    :param input_id:        Input ID
    :param input_file:      Path to the input matrix or StringIO object
    :param output_prefix:   Prefix of output file names
    :param classifiers:     List of classifiers
    :param folds:           Number of folds for the cross validation
    :param in_memory:       Keep results in memory
    :param nproc:           Make the execution of classifiers parallel when possible
    :param verbose:         Pront messages on the stdout
    :return:                Dictionary with paths to the "confusion" and "evaluate" files
                            Return a dictionary with StringIO objects in case of in_memory
                            Also return the accuracy for each of the classification models
    """

    evaluations = dict()

    if isinstance(input_file, str):
        # Check if input_file exists
        if not os.path.isfile(input_file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_file)

    else:
        # input_file is a StringIO object
        input_file.seek(0)

    # Use supported classifiers only
    if not classifiers:
        classifiers = SUPPORTED_CLASSIFIERS

    # In case it has been selected only one classifier
    if isinstance(classifiers, str):
        classifiers = [classifiers]

    classifiers = list(set(classifiers).intersection(set(SUPPORTED_CLASSIFIERS)))

    if not classifiers:
        raise ValueError("Unknown classifier")

    if not in_memory:
        # Check if outputs already exist
        for output_suffix in ["__evaluate.csv", "__confusion.csv"]:
            if os.path.isfile('{}{}'.format(output_prefix, output_suffix)):
                raise Exception("The output file already exists")

    # Process the input_file matrix
    if verbose:
        print("Building evaluation matrix")

        if isinstance(input_file, str):
            print("\tInput: {}".format(input_file))

    # Load matrix as pandas dataframe
    dataframe = pd.read_csv(input_file, index_col=0)

    # Fix class column name
    dataframe.set_axis([*dataframe.columns[:-1], "Class"], axis=1, inplace=True)
    pathsize = len(list(dataframe.columns)) - 1  # Exclude the 'Class' column

    if verbose:
        print("\tPathway Size: {}".format(pathsize))

    # Select a set of classifiers
    models = dict()
    selected_algorithms = list()

    for classifier in classifiers:
        classifier_obj = get_classifier(classifier)

        if classifier_obj is not None:
            models[classifier] = classifier_obj
            selected_algorithms.append(classifier)

    # Build matrix content X and classes vector y
    X_df, y_df = dataframe.drop("Class", axis=1), dataframe["Class"]

    labels = sorted(list(set(y_df)))

    # Dummify our data to make sklearn happy
    x = pd.get_dummies(X_df, columns=X_df.select_dtypes("object").columns)
    y = y_df.map(lambda v: labels.index(v))

    samples_profiles_real = dict()
    samples_list_sorted = list(y.index)

    for sample, class_label in zip(list(y.index), list(y.values)):
        samples_profiles_real[sample] = class_label

    confusion_filepath = "{}__{}__confusion.csv".format(output_prefix, pathsize)
    confusion_outfile = None

    if in_memory:
        confusion_outfile = StringIO()

    else:
        # Open output file
        confusion_outfile = open(confusion_filepath, "w+")

    # Define output file headers
    confusion_outfile.write("# Pathway size: {}\n".format(pathsize))
    confusion_outfile.write("# {} (predicted),{} (predicted),Algorithm,Time (seconds)\n".format(labels[0], labels[1]))

    if verbose:
        print("\tRunning algorithms:")

    sample_profiles_predicted = dict()

    accuracies = dict()

    for model_name in selected_algorithms:
        model = models[model_name]

        if verbose:
            print("\t\t{}".format(model_name))

        # Initialize KFold
        kfold = KFold(n_splits=folds)

        t0 = time.time()

        # Run current classification model in cross validation
        # Keep track of running time
        y_pred = cross_val_predict(model, x, y, cv=kfold, n_jobs=nproc)

        t1 = time.time()

        for predicted_class_idx in range(len(y_pred)):
            sample = samples_list_sorted[predicted_class_idx]

            if sample not in sample_profiles_predicted:
                sample_profiles_predicted[sample] = list()

            sample_profiles_predicted[sample].append( "0" if y_pred[predicted_class_idx] == samples_profiles_real[sample] else "1" )

        conf_matrix = confusion_matrix(y, y_pred, labels=list(range(len(labels))))

        # Dump the confusion matrix into the output file
        row_count = 0
        for row in conf_matrix:
            confusion_outfile.write("{} (true),{},{},{},{}\n".format(labels[row_count], row[0], row[1], model_name, float(t1-t0)))
            row_count += 1
        
        # Take track of the model accuracy
        accuracies[model_name] = accuracy_score(y, y_pred)

    if in_memory:
        evaluations["confusion"] = confusion_outfile

    else:
        confusion_outfile.close()
        evaluations["confusion"] = confusion_filepath

    evaluate_filepath = "{}__{}__evaluate.csv".format(output_prefix, pathsize)

    if in_memory:
        evaluate_outfile = StringIO()

    else:
        evaluate_outfile = open(evaluate_filepath, "w+")

    # Dump also the evaluation matrix
    evaluate_outfile.write("# Pathway size: {}\n".format(pathsize))
    evaluate_outfile.write("# Sample,{}\n".format(",".join(selected_algorithms)))

    for sample in sample_profiles_predicted:
        evaluate_outfile.write("{},{}\n".format(sample, ",".join(sample_profiles_predicted[sample])))

    if in_memory:
        evaluate_outfile.seek(0)
        evaluations["evaluate"] = evaluate_outfile

    else:
        evaluations["evaluate"] = evaluate_filepath

    return input_id, evaluations, accuracies
