import pandas as pd
import os
from numpy.random import randint
import numpy as np


def load_dataset(fname):
    company_df = pd.read_csv(fname)
    num_open_addr = len(company_df.index)
    print("{} addresses in dataset {}".format(num_open_addr, fname))
    return company_df


if __name__ == "__main__":
    # files containing data
    company_fname = "data/companies.csv"
    english_fname = "data/english_scrabble.txt"
    names_fname = "data/names.txt"

    # load dataframes from files
    company_df = load_dataset(company_fname)[["Company Name"]].copy()
    english_df = load_dataset(english_fname)
    names_df = load_dataset(names_fname)

    # rename words to element
    company_df = company_df.rename(index=str, columns={"Company Name": "element"})
    english_df = english_df.rename(index=str, columns={"dripstone": "element"})
    names_df = names_df.rename(index=str, columns={"mcleese": "element"})

    num_words = 4
    other_df = pd.concat([english_df, names_df])
    num_elements = len(other_df.index)
    other_df = other_df.sample(frac=1).reset_index(drop=True)

    other_df_copy = other_df.copy()
    other_df = pd.DataFrame(columns=["element", "label"])

    print("creating new dataset with multiple words")
    count = 0
    while count < num_elements:
        ws = randint(1, num_words)
        new_words = []
        for i in range(count, count + ws):
            new_words.append(str(other_df_copy.at[i, "element"]))
        count += ws
        other_df = other_df.append({"element": " ".join(new_words)}, ignore_index=True)
        if count % 1000 == 0:
            print("wrote {} of {} examples".format(count, num_elements))

    # label elements
    company_df["label"] = 1
    other_df["label"] = 0

    # concatenate all datasets
    print("combining datasets")
    print("\tother dataset has {} examples".format(len(other_df.index)))
    print("\tcompany dataset has {} examples".format(len(company_df.index)))
    company_df = company_df.sample(frac=1 / 3)
    print("\twhich was reduced to {} examples".format(len(company_df.index)))

    combined_df = pd.concat([company_df, other_df])

    # shuffle the datasets
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # send to lower case
    combined_df["element"] = combined_df["element"].str.lower()

    # get rid of spaces
    # combined_df['element'] = combined_df['element'].str.replace(' ','')

    # get rid of company names that are too long
    # combined_df = combined_df.loc[combined_df['element'].str.len() < 15]

    # get rid of company names with numbers
    combined_df = combined_df.loc[
        combined_df["element"].str.replace(" ", "").str.isalpha()
    ]

    # reset index
    combined_df.reset_index(drop=True, inplace=True)

    # write to file
    print("writing to file")
    print(combined_df.head())
    combined_df.to_csv("data/combined.csv", header=False, index=False)
