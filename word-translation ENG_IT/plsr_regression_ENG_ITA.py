"""Word translation

Usage:
  plsr_regression_ENG_ITA.py [-v | --verbose] 
  plsr_regression_ENG_ITA.py (-h | --help)

Options:
  -h --help      Show this screen.
  --version      Show version.
  -v --verbose   Show verbose output.

"""

from docopt import docopt
import numpy as np
import utils
from sklearn.cross_decomposition import PLSRegression
from utils import run_PCA, dm_dict_en, dm_dict_it, dm_dict_en_pairs, dm_dict_it_pairs


def mk_training_matrices(pairs, en_dimension, ita_dimension, english_space, italian_space):
    en_mat = np.zeros((len(pairs),en_dimension)) 
    ita_mat = np.zeros((len(pairs),ita_dimension))
    c = 0
    for p in pairs:
        en_word,ita_word = p.split()
        en_mat[c] = english_space[en_word]   
        ita_mat[c] = italian_space[ita_word]   
        c+=1
    return en_mat,ita_mat


def PLSR(mat_english,mat_italian,ncomps):
    plsr = PLSRegression(n_components=ncomps)
    plsr.fit(mat_english,mat_italian)
    return plsr 

if __name__ == '__main__':
    args = docopt(__doc__, version='PLSR regression for word translation 1.1')
    verbose = False
    if args["--verbose"]:
        verbose = True

    #Read semantic spaces
    english_space = utils.readDM("data/english.subset.dm")
    italian_space = utils.readDM("data/italian.subset.dm")

    #Read all word pairs
    all_pairs = []
    f = open("data/pairs_ENG_ITA.txt")
    for l in f:
        l = l.rstrip('\n')
        all_pairs.append(l)
    f.close()

    #Make training/test pairs: 
    training_pairs = all_pairs[:53]
    test_pairs = all_pairs[54:]

    #Define the ranges of ncomps and nns to try
    ncomps_range = [5, 10, 15, 20, 25, 30, 40, 50]  
    nns_range = [1, 2, 3, 5, 8, 12]  

    for ncomps in ncomps_range:
        for nns in nns_range:
            print(f"Trying ncomps={ncomps}, nns={nns}:")
            '''Make training/test matrices and get PLSR model'''
            en_mat, ita_mat = mk_training_matrices(training_pairs, 400, 300, english_space, italian_space)
            plsr = PLSR(en_mat,ita_mat,ncomps)

            ''' Predict with PLSR'''
            score = 0
            for p in test_pairs:
                en, ita = p.split()
                predicted_vector = plsr.predict(english_space[en].reshape(1,-1))[0]
                #print(predicted_vector[:20])
                nearest_neighbours = utils.neighbours(italian_space,predicted_vector,nns)
                if ita in nearest_neighbours:
                    score+=1
                    if verbose:
                        print(en,ita,nearest_neighbours,"1")
                else:
                    if verbose:
                        print(en,ita,nearest_neighbours,"0")

            print("Precision PLSR:",score/len(test_pairs))
            print()

#Create a table with precision results
def results_table(ncomps_range, nns_range):
    results = []
    for ncomps in ncomps_range:
        for nns in nns_range:

            #Make training/test matrices and get PLSR model
            en_mat, ita_mat = mk_training_matrices(training_pairs, 400, 300, english_space, italian_space)
            plsr = PLSR(en_mat, ita_mat, ncomps)

            #Predict with PLSR
            score = 0
            for p in test_pairs:
                en, ita= p.split()
                predicted_vector = plsr.predict(english_space[en].reshape(1, -1))[0]
                nearest_neighbours = utils.neighbours(italian_space, predicted_vector, nns)
                if ita in nearest_neighbours:
                    score += 1

            results.append((ncomps, nns, score / len(test_pairs)))
    return results


results = results_table(ncomps_range, nns_range)
print("ncomps\tnns\tprecision")
for row in results:
    print(f"{row[0]}\t{row[1]}\t{row[2]}")


#run PCA to create a PNG space
run_PCA(dm_dict_en, 'data/english.subset_pca.png', figsize=(20, 20), xlim=(-1.5, 2.0), ylim=(-1.5, 2.0))  
run_PCA(dm_dict_it, 'data/italian.subset_pca.png', figsize=(20, 20), xlim=(-10, 10), ylim=(-10, 10))

#PNG spaces of the words present in pairs: uncomment them to use
run_PCA(dm_dict_en_pairs, 'data/eng_pairs_subset.png', figsize=(20, 20), xlim=(-2.0, 2.5), ylim=(-1.5, 3.0))  
run_PCA(dm_dict_it_pairs, 'data/ita_pairs_subsetpng', figsize=(20, 20), xlim=(-10, 10), ylim=(-10, 10))