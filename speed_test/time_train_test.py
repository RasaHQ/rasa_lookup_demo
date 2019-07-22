from time import time
import random
import string
import os, shutil
import numpy as np
import matplotlib.pylab as plt

from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config, evaluate

"""
Trains a bunch of models with differently sized lookup tables
Times the results of training and evaluation
Creates pretty plots.
"""


def clear_model_dir():
    # clears old models
    folder = "tmp/models"
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def construct_lookup(n_words, w_len):
    # makes a random lookup table
    lookup_fname = "tmp/lookup.txt"
    open(lookup_fname, "w").close()
    with open(lookup_fname, "w") as f:
        word = "".join(random.choices(string.ascii_letters + string.digits, k=w_len))
        words = "\n".join(n_words * [word])
        f.write(words)


def train_model():
    # trains a model and times it
    t = time()
    # training_data = load_data('demo_train.md')
    training_data = load_data("data/company_train_lookup.json")
    td_load_time = time() - t
    trainer = Trainer(config.load("config.yaml"))
    t = time()
    trainer.train(training_data)
    train_time = time() - t
    clear_model_dir()
    t = time()
    model_directory = trainer.persist(
        "./tmp/models"
    )  # Returns the directory the model is stored in
    persist_time = time() - t
    return td_load_time, train_time, persist_time


def evaluate_model():
    # evaluates a model and times it
    model_name = os.listdir("./tmp/models/default")[0]  # get first (and only) model
    t = time()
    evaluate.run_evaluation("data/demo_test.md", "./tmp/models/default/" + model_name)
    eval_time = time() - t
    return eval_time


def plot_results(
    words,
    lookup_construct_time,
    load_time,
    train_time,
    persist_time,
    eval_time,
    total_time,
    plt_times=True,
):
    f, ax = plt.subplots()
    ax.plot(words, load_time)
    ax.plot(words, train_time)
    ax.plot(words, persist_time)
    ax.plot(words, eval_time)
    ax.plot(words, total_time)
    if plt_times:
        a = 0.6
        ax.plot([words[0], words[-1]], [1, 1], "k:", alpha=a)
        ax.plot([words[0], words[-1]], [60, 60], "k-.", alpha=a)
        ax.plot([words[0], words[-1]], [600, 600], "k--", alpha=a)

    ax.set_xlabel("number of words")
    ax.set_ylabel("time to evaluate (sec)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if plt_times:
        ax.legend(
            (
                "load data",
                "train",
                "persist",
                "evaluate",
                "total",
                "1 sec",
                "1 min",
                "10 min",
            ),
            loc="upper left",
        )
    else:
        ax.legend(
            ("load data", "train", "persist", "evaluate", "total"), loc="upper left"
        )
    return f, ax


def print_stats(
    count,
    num_words_list,
    num_words,
    lookup_construct_time,
    td_load_time,
    train_time,
    persist_time,
    eval_time,
    total_time,
):

    print("{0:.2f} % done".format((count + 1) / len(num_words_list) * 100))
    print("with {} words in lookup table:".format(num_words))
    print("    took {} sec. to construct lookup table".format(lookup_construct_time))
    print("    took {} sec. to load training data".format(td_load_time))
    print("    took {} sec. to train model".format(train_time))
    print("    took {} sec. to perist model".format(persist_time))
    print("    took {} sec. to evaluate on test set".format(eval_time))
    print("    took {} sec. total".format(total_time))


if __name__ == "__main__":

    N = 1
    word_length = 10
    verbose = True

    num_word_distribution = np.logspace(0, 7, num=N)

    # integerize and uniquify this
    num_words_list = [int(num_word_distribution[i]) for i in range(N)]
    num_words_list = list(set(num_words_list))
    num_words_list.sort()

    t_train = []
    t_eval = []
    t_total = []

    for count, num_words in enumerate(num_words_list):

        t_tot = time()
        construct_lookup(num_words, word_length)
        train_time = train_model()
        eval_time = evaluate_model()
        total_time = time() - t_tot

        if verbose:
            print_stats(
                count, num_words_list, num_words, train_time, eval_time, total_time
            )

        t_train.append(train_time)
        t_eval.append(eval_time)
        t_total.append(total_time)

    f, ax = plot_results(num_words_list, t_train, t_eval, t_total, plt_times=True)
    f.savefig("img/results_terminal.png", dpi=400)
    plt.show()
