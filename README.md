# Phrase Matcher Demo

This is a simple demo of the new lookup table feature in [rasa_nlu](https://rasa.com/docs/nlu/).  See the blog post accompanying this repository [here](https://medium.com/rasa-blog/entity-extraction-with-the-new-lookup-table-feature-in-rasa-nlu-94c6c30876a3)

The goal is to show how lookup tables may improve entity extraction under certain conditions and also give some advice on using this feature effectively.

This repo contains two demos:

1.  A simple restaurant example with very few training examples and only one entity.
2.  A medium-sized company name extraction example with a few thousand examples and several entities.


## Running the demo.

No installation is necessary although you must have [rasa_nlu](https://rasa.com/docs/nlu/) installed and version > 0.13.3 or above.

To run one or both of the demos:

	python run_lookup.py <demo_key>

where `<demo_key>` is one of `{food, company}`.  If `<demo_key>` is ommitted, it will run both of the demos.

### Code Structure

`data/` holds the training data and lookup tables for each of the demos.

`models/` is where the models are persisted.

`configs/` holds the rasa_nlu configs to do the baseline evaluation and the lookup table evaluation.

`img/` stores plots and outputs from the runs.

## Cleaning lookup tables

The script `filter_lookup.py` may be used to clean up lookup tables by removing any elements that match with a cross-list.

You can call this scripy by running

	python filter_lookup.py <lookup_in> <cross_list> <lookup_out>

`<lookup_in>` is a lookup table with newline-separated elements.

`<cross_list>` is either a comma or newline-separated list of elements that you'd like to remove from `<lookup_in>`

`<lookup_out>` is the name of the file that you'd like to write the filtered list to.

<img src="https://github.com/RasaHQ/rasa_lookup_demo/blob/master/img/filter_diagram.png?raw=True" width="300">

## Speed Testing

We include the directory `speed_test/` for testing the speed of training as a function of the lookup table size.

This generates random lookup tables and times each component of the training and evaluation process.  We use the company dataset `data/company/company_train_lookup.py`. 

	cd speed_test
	python time_lookups.py

See `speed_test/README.md` for more details.

<img src="https://github.com/RasaHQ/rasa_lookup_demo/blob/master/img/timings.png?raw=True" width="500">

## Ngrams

A simple ngrams tester is included and can be run by

	python run_ngrams.py

This loads two lookup tables, `data/company/pos_ngrams.txt` & `data/company/neg_ngrams.txt`, each containing ngrams that were found to be influential to classifying phrases as company names.  We then compute the f1 score as a function of random noise injected into the entities.  The 'noise' value is the probability of a character flip in each character of each company entity in the test set.

This gives the following plot

<img src="https://github.com/RasaHQ/rasa_lookup_demo/blob/master/img/ngram_robustness.png?raw=True" width="500">

