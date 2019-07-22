from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu import evaluate
from rasa_nlu import utils

import logging
import re

import numpy.random as rd
import numpy as np
import random
import string

import matplotlib.pylab as plt

# to get the logging stream into a string
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

"""
This script investigates whether adding ngrams improves the robustness of the 
NLU model.  It is run simply as

    python run_ngrams.py

and uses the company dataset for the testings.
Durig the run, noise is added to the dataset.  
With some probability 'noise', each of the company entity characters are 
flipped to a random value.  
This is averaged over several runs per value of 'noise'.
"""


def train_model(td_file, config_file, model_dir):
    """trains a model using the training data and config
       creates model and returns the path to this model for evaluation"""
    td = load_data(td_file)
    trainer = Trainer(config.load(config_file))
    trainer.train(td)
    model_loc = trainer.persist(model_dir)

    return model_loc


def train_test(td_file, config_file, model_dir, key="company", noise=0.1):
    """trains a model using the training data
       (split into train-test) and config"""
    td = load_data(td_file)

    trainer = Trainer(config.load(config_file))
    train, test = td.train_test_split(train_frac=0.8)
    test = add_noise(test, key, noise=noise)

    trainer.train(train)
    tmp_fname = "data/tmp/temp_test.json"
    model_loc = trainer.persist(model_dir)
    with open(tmp_fname, "w", encoding="utf8") as f:
        f.write(test.as_json())
    evaluate_model(tmp_fname, model_loc)


def add_noise(td, key, noise=0.2):
    """with probability 'noise', randomizes each character of company entities"""
    entity = key  # the entity is just the key.
    for ex in td.training_examples:
        entities = ex.get("entities")
        text = ex.text
        if entities:
            for e in entities:
                if e["entity"] == entity:
                    value = e["value"]
                    new_value = list(value)
                    for i in range(len(value)):
                        if rd.random() < noise and new_value[i] != " ":
                            new_value[i] = random.choice(string.ascii_lowercase)
                    new_value = "".join(new_value)
                    e["value"] = new_value
                    old_text = ex.as_dict()["text"]
                    if old_text:
                        new_text = list(old_text)
                        new_text[e["start"] : e["end"] + 1] = new_value
                        new_value = "".join(new_text)
                        ex.text = "".join(new_text)
        if entities:
            ex.set("entities", entities)
    return td


def CV_eval(td_file, config_file, Nfolds=10):
    """trains a model with crossvalidation using the training data and config"""
    td = load_data(td_file)
    configuration = config.load(config_file)
    evaluate.run_cv_evaluation(td, Nfolds, configuration)


def evaluate_model(td_file, model_loc):
    """evaluates the model on the training data."""
    evaluate.run_evaluation(td_file, model_loc)


def get_path_dicts(key):
    """gets the right training data and model directory given the demo"""
    td_dict = {"company": "data/company/company_full.json"}
    td_lookup_dict = {"company": "data/company/company_full_lookup.json"}
    td_ngrams_dict = {"company": "data/company/company_full_ngrams.json"}
    td_both_dict = {"company": "data/company/company_full_both.json"}
    model_dir_dict = {"company": "data/models"}

    td = td_dict[key]
    td_lookup = td_lookup_dict[key]
    td_ngrams = td_ngrams_dict[key]
    td_both = td_both_dict[key]
    model_dir = model_dir_dict[key]

    return td, td_lookup, td_ngrams, td_both, model_dir


def run_demo(key, noise=0.1):
    """runs the demo specified by key"""
    td, td_lookup, td_ngrams, td_both, model_dir = get_path_dicts(key)
    config_file = "configs/config.yaml"

    print("running normal...")
    train_test(td, config_file, model_dir, noise=noise)
    print("running lookups...")
    train_test(td_lookup, config_file, model_dir, noise=noise)
    print("running ngrams...")
    train_test(td_ngrams, config_file, model_dir, noise=noise)
    print("running both...")
    train_test(td_both, config_file, model_dir, noise=noise)

    metric_list = strip_metrics(key)
    return metric_list


def parse_metrics(match, key):
    """Gets the metrics out of the parsed logger stream"""
    elements = match.split(" ")[1:]
    elements = filter(lambda x: len(x) > 2, elements)
    elements = [float(e) for e in elements]
    metrics = dict(zip(["key", "precision", "recall", "f1"], [key] + elements))
    return metrics


def strip_metrics(key):
    # steals the logger stream and returns the metrics associated with key
    stream_string = log_stream.getvalue()
    stream_literal = repr(stream_string)
    p_re = re.compile(key + "[ ]+\d.\d\d[ ]+\d.\d\d[ ]+\d.\d\d")
    matches = p_re.findall(stream_literal)
    metric_list = [parse_metrics(m, key) for m in matches]
    return metric_list


def print_metrics(metric_list):
    """Prints the metrics for each training data"""
    if metric_list:
        key = metric_list[-4]["key"]
        print("before adding lookup table(s), demo '{}' had:".format(key))
        display_metrics(metric_list[-4])
        print("after adding lookup table(s),  demo '{}' had:".format(key))
        display_metrics(metric_list[-3])
        print("after adding ngrams,  demo '{}' had:".format(key))
        display_metrics(metric_list[-2])
        print(
            "after adding both lookup table(s) and ngrams,  demo '{}' had:".format(key)
        )
        display_metrics(metric_list[-1])
    else:
        raise ValueError("metrics were not parsed correctly.")


def display_metrics(metrics):
    """Prints the metrics"""
    for key, val in metrics.items():
        print("\t{}:\t{}".format(key, val))


if __name__ == "__main__":

    # capture logging to string
    log_stream = StringIO()
    logging.basicConfig(stream=log_stream, level=logging.INFO)

    # number to average over
    N_avg = 7

    # number of noise points
    N_noise = 20

    # probability of randomizing each character of key entity
    noises = list(np.linspace(0, 0.5, N_noise))

    # store the F1 scores
    f1s = {
        "normal": [[] for _ in range(N_avg)],
        "lookup": [[] for _ in range(N_avg)],
        "ngram": [[] for _ in range(N_avg)],
        "both": [[] for _ in range(N_avg)],
    }

    # cases to examine
    cases = ["normal", "lookup", "ngram", "both"]

    count = 1
    for avg_index in range(N_avg):

        for noise in noises:

            print("working on run {} of {}".format(count, N_avg * N_noise))
            print("    noise = {}".format(noise))

            count += 1

            log_stream.flush()

            # run the demo on company
            metric_list = run_demo("company", noise=noise)
            for i, e in enumerate(cases):
                f1s[e][avg_index].append(metric_list[-4 + i]["f1"])

    for case, f1_list in f1s.items():

        # compute the average over all runs for each noise level
        f1s_T = list(map(list, zip(*f1_list)))
        avgs = [sum(f) / N_avg for f in f1s_T]

        plt.plot(noises, avgs)

    plt.legend(cases)
    plt.xlabel("noise")
    plt.ylabel("f1s")
    plt.show()
