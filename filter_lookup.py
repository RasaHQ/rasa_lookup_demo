import csv
import sys
from typing import Tuple, Set

"""
This script takes a list of startups and removes
any startups that have names that are also scrabble words
"""


def open_data(filename):
    out_list = []
    print("loading data from {}".format(filename))
    with open(filename, "rt") as f:
        reader = csv.reader(f)
        for row in reader:
            out_list += row
    print("found {} elements".format(len(out_list)))
    return out_list


def write_data(filename, filtered_startups):
    print("writing that to file at : {}".format(filename))
    with open(filename, "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows([filtered_startups])


def filter_list(in_list: Tuple, scrabble_set: Set):
    filtered_list = set()
    for i, s in enumerate(in_list):
        if i % 100 == 0:
            print(
                "percent done: {} % \t  elements removed: {}".format(
                    int(1000 * i / len(in_list)) / 10, i - len(filtered_list)
                )
            )
        if s.lower() not in scrabble_set:
            words = s.split(" ")
            if any([word.lower() not in scrabble_set for word in words]):
                filtered_list.add(s)
    filtered_list = list(filtered_list)
    print(
        "now have {} elements, removed {}".format(
            len(filtered_list), len(in_list) - len(filtered_list)
        )
    )
    return filtered_list


def parse_cmi():
    argv = sys.argv

    # defaults
    read_file = "data/company/startups.csv"
    scrabble_file = "data/company/english_scrabble.txt"
    write_file = "data/company/startups_filtered.csv"

    out_files = [read_file, scrabble_file, write_file]

    # read in command line args
    if len(argv) > 1:
        for i, filename in enumerate(argv):
            # ignore the script name
            if i == 0:
                continue
            out_files[i - 1] = filename

    return tuple(out_files)


if __name__ == "__main__":

    read_file, scrabble_file, write_file = parse_cmi()

    in_list = open_data(read_file)

    scrabble_list = open_data(scrabble_file)

    filtered_list = filter_list(tuple(in_list), set(scrabble_list))

    write_data(write_file, filtered_list)
