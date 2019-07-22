from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu import evaluate
from rasa_nlu import utils

import logging
import re

import matplotlib.pylab as plt

# to get the logging stream into a string
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

"""
This script demonstrates the improvment of entity extraction recall by
use of lookup tables.  A new feature in rasa_nlu.

The demo can by run by 

	python run_lookup.py <demo>

where <demo> is one of {food, company}.  
If <demo> is omitted, then it will both demos back to back.
See the README.md for more information.
"""

DEMO_KEYS = ["food", "company"]


def train_model(td_file, config_file, model_dir):
    # trains a model using the training data and config

    td = load_data(td_file)
    trainer = Trainer(config.load(config_file))
    trainer.train(td)

    # creates model and returns the path to this model for evaluation
    model_loc = trainer.persist(model_dir)

    return model_loc


def train_test(td_file, config_file, model_dir):
    # helper function to split into test and train and evaluate on results.

    td = load_data(td_file)
    trainer = Trainer(config.load(config_file))
    train, test = td.train_test_split(train_frac=0.6)
    trainer.train(train)
    model_loc = trainer.persist(model_dir)
    with open("data/tmp/temp_test.json", "w", encoding="utf8") as f:
        f.write(test.as_json())
    with open("data/temp_train.json", "w", encoding="utf8") as f:
        f.write(train.as_json())
    evaluate_model("data/tmp/temp_test.json", model_loc)


def CV_eval(td_file, config_file, Nfolds=10):
    # trains a model with crossvalidation using the training data and config

    td = load_data(td_file)
    configuration = config.load(config_file)
    evaluate.run_cv_evaluation(td, Nfolds, configuration)


def evaluate_model(td_file, model_loc):
    # evaluates the model on the training data
    # wrapper for rasa_nlu.evaluate.run_evaluation

    evaluate.run_evaluation(td_file, model_loc)


def get_path_dicts(key):
    # gets the right training data and model directory given the demo
    training_data_dict = {
        "food": "data/food/food_train.md",
        "company": "data/company/company_train.json",
    }
    training_data_lookup_dict = {
        "food": "data/food/food_train_lookup.md",
        "company": "data/company/company_train_lookup.json",
    }
    test_data_dict = {
        "food": "data/food/food_test.md",
        "company": "data/company/company_test.json",
    }
    model_dir_dict = {"food": "models/food", "company": "models/company"}

    training_data = training_data_dict[key]
    training_data_lookup = training_data_lookup_dict[key]
    test_data = test_data_dict[key]
    model_dir = model_dir_dict[key]

    return training_data, training_data_lookup, test_data, model_dir


def run_demo(key, disp_bar=True):
    # runs the demo specified by key

    # get the data for this key and the configs
    training_data, training_data_lookup, test_data, model_dir = get_path_dicts(key)
    config_file = "configs/config.yaml"
    config_baseline = "configs/config_no_features.yaml"

    # run a baseline
    model_loc = train_model(training_data, config_baseline, model_dir)
    evaluate_model(test_data, model_loc)

    # run with more features in CRF
    model_loc = train_model(training_data, config_file, model_dir)
    evaluate_model(test_data, model_loc)

    # run with the lookup table
    model_loc = train_model(training_data_lookup, config_file, model_dir)
    evaluate_model(test_data, model_loc)

    # get the metrics
    metric_list = strip_metrics(key)

    # either print or plot them
    if disp_bar:
        plot_metrics(metric_list)
    else:
        print_metrics(metric_list)


def parse_metrics(match, key):
    # from the regex match, parse out the precision, recall, f1 scores

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
    # prints out the demo preformance

    key = metric_list[0]["key"]
    print("baseline, demo '{}' had:".format(key))
    display_metrics(metric_list[0])
    print("before adding lookup table(s), demo '{}' had:".format(key))
    display_metrics(metric_list[1])
    print("after adding lookup table(s),  demo '{}' had:".format(key))
    display_metrics(metric_list[2])


def display_metrics(metrics):
    # helper function for print_metrics

    for key, val in metrics.items():
        print("\t{}:\t{}".format(key, val))


def plot_metrics(metric_list, save_path=None):
    # runs through each test case and adds a set of bars to a plot.  Saves

    f, (ax1) = plt.subplots(1, 1)
    plt.grid(True)

    print_metrics(metric_list)

    bar_metrics(metric_list[0], ax1, index=0)
    bar_metrics(metric_list[1], ax1, index=1)
    bar_metrics(metric_list[2], ax1, index=2)

    if save_path is None:
        save_path = "img/bar_" + key + ".png"

    plt.savefig(save_path, dpi=400)


def bar_metrics(metrics, ax, index=0):
    # adds a set of metrics bars to the axis 'ax' of the plot

    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    title = metrics["key"]
    width = 0.2
    shift = index * width
    indeces = [r + shift for r in range(3)]
    metric_list = [precision, recall, f1]
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.bar(indeces, metric_list, width)
    ax.set_xticks([r + width / 2 for r in range(3)])
    ax.set_xticklabels(("precision", "recall", "f1"))
    ax.legend(["baseline", "no lookup", "with lookup"])
    ax.set_ylabel("score")


if __name__ == "__main__":

    # capture logging to string
    log_stream = StringIO()
    logging.basicConfig(stream=log_stream, level=logging.INFO)

    # whether to create and save a bar plot
    disp_bar = True

    import sys

    argv = sys.argv

    if len(argv) < 2:
        # run all of the demos
        for key in DEMO_KEYS:
            run_demo(key, disp_bar=disp_bar)
    else:
        key = argv[1]
        if key not in DEMO_KEYS:
            raise ValueError(
                "first argument to run_demo.py must be one of {'food','company'}"
            )
        else:
            run_demo(key, disp_bar=disp_bar)
