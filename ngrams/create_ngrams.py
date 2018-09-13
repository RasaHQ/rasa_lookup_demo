from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import cloudpickle


def load_file(fname='data/combined_new.csv'):
    print('\nloading file {}...'.format(fname))
    df = pd.read_csv(fname, sep=',', names=['word', 'label'])
    total_examples = len(df)
    street_examples = len(df[df.label == 1])
    print(df.head())
    print('we have {} company examples out of {} total examples'.format(
        street_examples, total_examples))
    return df


def prep_data(df):
    print('\nsplitting into test and train...')
    df_x = df['word'].values.astype('U')
    df_y = df['label'].values.astype('U')
    x_train, x_test, y_train, y_test = train_test_split(
        df_x, df_y, test_size=0.2, random_state=1110)
    y_train = [int(y) for y in y_train]
    y_test = [int(y) for y in y_test]
    return x_train, x_test, y_train, y_test, df_x


def transorm_ngrams(df_x, x_train, x_test):
    print('\ntransforming inputs with ngrams...')
    vectorizer = CountVectorizer(ngram_range=(1, 5), analyzer='char')
    X = vectorizer.fit_transform(df_x)
    X_train = vectorizer.transform(x_train)
    X_test = vectorizer.transform(x_test)
    names = vectorizer.get_feature_names()
    return X_train, X_test, names


def get_features(X_train, y_train, names, selection_threshold=0.2):
    print('\ngetting features with randomized logistic regression...')
    print('using a selection threshold of {}'.format(selection_threshold))
    randomized_logistic = RandomizedLogisticRegression(
        selection_threshold=selection_threshold)
    randomized_logistic.fit(X_train, y_train)
    mask = randomized_logistic.get_support()
    features = np.array(names)[mask]
    print('found {} ngrams:'.format(len([f for f in features])))
    print([f for f in features])
    return features


def run_logreg(X_train, y_train, selection_threshold=0.2):
    print('\nrunning logistic regression...')
    print('using a selection threshold of {}'.format(selection_threshold))
    pipe = Pipeline([
        ('feature_selection', RandomizedLogisticRegression(
            selection_threshold=selection_threshold)),
        ('classification', LogisticRegression())
    ])
    pipe.fit(X_train, y_train)
    print('training accuracy : {}'.format(pipe.score(X_train, y_train)))
    print('testing accuracy : {}'.format(pipe.score(X_test, y_test)))
    return pipe


def get_pos_neg(pipe, features, f_pos='./data/pos_ngrams.txt', f_neg='./data/neg_ngrams.txt', cutoff=0.5):
    print('\nseparating into positive and negative ngrams...')
    print('using a cutoff of {}'.format(cutoff))
    params = pipe.get_params()
    logistic = params['classification']
    coeffs = logistic.coef_[0]
    coef_dict = {f: c for f, c in zip(features, coeffs)}
    positive_features = [f for f, c in coef_dict.items() if abs(c) > cutoff and c > 0]
    negative_features = [f for f, c in coef_dict.items() if abs(c) > cutoff and c < 0]
    print('positive ngrams : {}\n{}'.format(len(positive_features), positive_features))
    print('')
    print('negative ngrams : {}\n{}'.format(len(negative_features), negative_features))    
    print('writing to files {} and {}'.format(f_pos, f_neg))
    with open(f_pos, 'w') as f:
        f.write('\n'.join([str(feat) for feat in positive_features]))
    with open(f_neg, 'w') as f:
        f.write('\n'.join([str(feat) for feat in negative_features]))


def write_all_ngrams(features, fname_base='./data/ngram_'):
    for i, f in enumerate(features):
        fname = fname_base + str(i) + '.txt'
        with open(fname, 'w') as f:
            f.write(str(f))      


if __name__ == '__main__':
    file_in = 'data/combined_new.csv'
    ST = 0.4
    cutoff = 0.9
    expand = True
    df = load_file()
    x_train, x_test, y_train, y_test, df_x = prep_data(df)
    X_train, X_test, names = transorm_ngrams(df_x, x_train, x_test)
    features = get_features(X_train, y_train, names, selection_threshold=ST)
    if expand:
        fname_base = './data/ngram_'
        write_all_ngrams(features, fname_base=fname_base)
    else:
        file_pos = '../phrase_match_test/regex_phrase_match_demo/company_data/data/pos_ngrams.txt'
        file_neg = '../phrase_match_test/regex_phrase_match_demo/company_data/data/neg_ngrams.txt'      
        pipe = run_logreg(X_train, y_train, selection_threshold=ST)
        get_pos_neg(pipe, features, cutoff=cutoff)



