import numpy as np
from time import time
from time_train_test import construct_lookup, train_model, evaluate_model, plot_results, print_stats
import matplotlib.pylab as plt

N = 100
word_length = 10
verbose = True
start_pow = 0    # 10^start_pow  is first point
end_pow = 7.0      # 10^end_pow    is last point

num_word_distribution = np.logspace(start_pow, end_pow, num=N)
num_words_list = [int(num_word_distribution[i]) for i in range(N)]
num_words_list = list(set(num_words_list))
num_words_list.sort()

t_train = []
t_eval = []
t_total = []
t_persist = []
t_load = []
t_lookup = []

for count, num_words in enumerate(num_words_list):

    t_tot = time()
    t = time()
    construct_lookup(num_words, word_length)
    lookup_construct_time = time() - t
    td_load_time, train_time, persist_time = train_model()
    eval_time = evaluate_model()    
    total_time = time() - t_tot

    if verbose:
        print_stats(count, num_words_list, num_words, lookup_construct_time,
                td_load_time, train_time, persist_time, eval_time, total_time)
        
    t_lookup.append(lookup_construct_time)
    t_load.append(td_load_time)        
    t_train.append(train_time)
    t_persist.append(persist_time)
    t_eval.append(eval_time)
    t_total.append(td_load_time+train_time+persist_time+eval_time)

print(t_train)
f, ax = plot_results(num_words_list, t_lookup,
                t_load, t_train, t_persist, t_eval, t_total, plt_times=True)


f.savefig('img/results.png', dpi = 400)
plt.show()

