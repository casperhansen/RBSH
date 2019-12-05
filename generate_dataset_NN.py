#import tensorflow as tf
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from scipy import sparse
#from graphs import make_graph_queue_simple_vae, matrix_for_embed, make_embedding, make_graph_queue_nash, make_importance_embedding
#from graphs_weak import make_graph_queue_SingleWeak
from gensim.models import KeyedVectors
#from generators import generator_vae_version, DataGenerator, \
#    DataGenerator_singleWeak, generator_singleWeak_version_eval
import time
#from generators import generator_vae_version_eval
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import datetime

def filter_weak_signals(weak_signals, indices, filter_indices):
    new_weak_signals = {}
    for index in indices:
        if index not in weak_signals:
            print("skipping", index)
            continue
        tmp = weak_signals[index]
        tmp = [tmp[i] for i in range(len(tmp)) if tmp[i][0] not in filter_indices]
        new_weak_signals[index] = tmp
    return new_weak_signals

def extract_indices(weak_signals, data_text, data_text_vect, labels, indices, filter_indices, topv):
    #new_data_text = [data_text[k] for k in indices]
    #new_labels = [labels[k] for k in indices]
    #new_data_text_vect = np.array([data_text_vect[k] for k in indices])
    new_weak_signals = filter_weak_signals(weak_signals, indices, filter_indices)

    # extract row combinations for weak signals of [doc, doc1, doc2], where doc is considered the query
    # extract top50: 50 * 50/2 - 50

    topSmallCombos = []
    topSmallScores = []
    print("extracting small")
    keys = list(new_weak_signals.keys())
    for index in new_weak_signals.keys():
        index_vals = new_weak_signals[index]
        for j in range(min(topv, len(index_vals))):
            j_val = index_vals[j]
            for k in range(j+1, min(topv, len(index_vals))):
                k_val = index_vals[k]
                topSmallCombos.append([index, j_val[0], k_val[0]])
                topSmallScores.append([j_val[2], k_val[2]])

    print("extracting big")
    # extract top1000: random 50 * 50/2 - 50
    pickNum = (topv * topv/2 - topv)
    topBigCombos = []
    topBigScores = []
    '''
    pre_combos = []
    for j in range(999):
        for k in range(j+1,999):
            pre_combos.append((j,k))
    len_pre_combos = len(pre_combos)


    for i in tqdm(range(len(keys))):
        index = keys[i]
        index_vals = new_weak_signals[index]
        perms = np.random.permutation(len_pre_combos)

        found = 0
        counter = 0
        while found < -1:#pickNum:
            j, k = pre_combos[perms[counter]]
            try:
                j_val = index_vals[j]
                k_val = index_vals[k]
                topBigCombos.append([index, j_val[0], k_val[0]])
                topBigScores.append([j_val[2], k_val[2]])
                found += 1
            except Exception as e:
                pass
            counter += 1
    '''
    combos = np.array(topSmallCombos + topBigCombos).astype(np.int32)
    scores = np.array(topSmallScores + topBigScores).astype(np.float32)

    return combos, scores, indices#, new_data_text, new_data_text_vect, new_labels

def make_collections(dataset_name):
    # load raw data
    data = pickle.load(open("datasets/" + dataset_name + ".pkl", "rb"))
    weak_signals = pickle.load(open("datasets/weakSignals/" + dataset_name + "/" + dataset_name + "_okapi_dic.pkl", "rb"))
    labels = [data[i][-1] for i in range(len(data))]

    # vectorize text data (for simple encoding using BoW)
    data_text = [data[i][1] for i in range(len(data))]
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
    vectorizer.fit(data_text)
    data_text_vect = vectorizer.transform(data_text)
    print(data_text_vect.shape)
    #data_text_vect = data_text_vect.todense()

    token2id = vectorizer.vocabulary_
    id2token = {token2id[key]:key for key in token2id}

    data_text_new = []
    for text in data_text:
        tokens = text.split(" ")
        tokens = [token2id[token] for token in tokens if token in token2id]
        data_text_new.append(tokens)

    data_text = data_text_new
    # split data_text into token-ids (for word embedding look ups)
    # data_text, token2id, id2token = unique_tokenizer(data_text)

    # split into train, val, and test 80/10/10
    perm = np.random.permutation(len(data))
    N = len(perm)
    train_idx = perm[:int(N * 0.8)]

    val_idx = perm[int(N * 0.8):int(N * 0.9)]
    test_idx = perm[int(N * 0.9):]

    train_collection = extract_indices(weak_signals, data_text, data_text_vect, labels, train_idx, test_idx, 20)
    train_collection_one_index = extract_indices(weak_signals, data_text, data_text_vect, labels, train_idx, test_idx, 2)

    val_collection = extract_indices(weak_signals, data_text, data_text_vect, labels, val_idx, test_idx, 2)
    val_collection_one_index = extract_indices(weak_signals, data_text, data_text_vect, labels, val_idx, test_idx, 2)

    test_collection = extract_indices(weak_signals, data_text, data_text_vect, labels, test_idx, [], 2)

    pickle.dump([train_collection, train_collection_one_index, val_collection, val_collection_one_index, test_collection, data_text, data_text_vect, labels, token2id, id2token],
                open(dataset_name + "_collections", "wb"))

    import scipy.io
    unique_labels = {}
    ii = 0
    for row in labels:
        for label in row:
            if label not in unique_labels:
                unique_labels[label] = ii
                ii += 1

    matlabels = np.zeros((len(labels), ii))
    for i, row in enumerate(labels):
        for label in row:
            matlabels[i, unique_labels[label]] = 1

    scipy.io.savemat(dataset_name + "_collections_MATLAB" + '.mat', mdict={'text_vect': data_text_vect,
                                                        'labels': matlabels, 'train_idx': train_idx.tolist(), 'val_idx': val_idx.tolist(),
                                                        'test_idx': test_idx.tolist()})
    exit()

def get_vectors_labels(gen, input_dict, sess, loss, rank_loss, sem_vector_bit, is_eval, kl_value, rank_value, cont_vec=None, cont_vec_cut=0.1):
    val_vectors = {}
    val_labels = {}
    losses = []
    rank_losses = []
    iii = 0
    for data in gen:
        #print(iii)
        iii += 1
        #print("------#####")
        doc, doc1, doc2, doc1weak, doc2weak, mask, labels, doc_idxs = data
        #tmp = input_dict(doc, mask)
        #tmp[is_eval] = False

        tmp = {test_placeholder_set[0]: doc, test_placeholder_set[1]: doc1, test_placeholder_set[2]: doc2,
               test_placeholder_set[3]: doc1weak, test_placeholder_set[4]: doc2weak,
               test_placeholder_set[5]: mask, is_eval: True, kl_weight: kl_value, rank_weight: rank_value,
               sigma_annealing: 0.0}

        if cont_vec is None:
            lossval, ranklossval, semvector = sess.run([loss, rank_loss, sem_vector_bit], feed_dict=tmp)
        else:
            lossval, ranklossval, semvector, semvector_cont = sess.run([loss, rank_loss, sem_vector_bit, cont_vec], feed_dict=tmp)

        #print("evalval:", evalll)
        rank_losses.append(ranklossval)
        losses.append(lossval)
        for i in range(len(labels)):
            if mask[i] > 0:
                val_vectors[doc_idxs[i]] = semvector[i]
                if args.cont_cut > 0.0001:
                    cont = semvector_cont[i]
                    lows = cont<cont_vec_cut
                    highs = cont>(1-cont_vec_cut)
                    tmp = semvector[i]
                    tmp[lows] = 0
                    tmp[highs] = 1
                    val_vectors[doc_idxs[i]] = tmp

                val_labels[doc_idxs[i]] = labels[i]
                #val_vectors.append(semvector[i])
                #val_labels.append(labels[i])


    keys = list(val_vectors.keys())

    return [val_vectors[key] for key in keys], [val_labels[key] for key in keys], losses, rank_losses


def as_matrix(config):
    return [[k, str(w)] for k, w in config.items()]

def eval_hashing_and_returnKNN(train_vectors, train_labels, val_vectors, val_labels, medianTrick=False):

    #train_vectors = np.array(train_vectors)
    #val_vectors = np.array(val_vectors)

    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)

    upto = 100
    knn = NearestNeighbors(n_neighbors=upto, n_jobs=-1)
    #knn = NearestNeighbors(n_neighbors=upto, n_jobs=-1)

    knn.fit(train_vectors)
    #print("fitted")

    nns = knn.kneighbors(val_vectors, upto, return_distance=False)
    print(len(nns))
    top100_precisions = []

    for i, nn_indices in enumerate(nns):
        #if i % 100 == 0:
        #print(i, len(nns))
        eval_label = val_labels[i]
        matches = np.zeros(upto)
        for j, idx in enumerate(nn_indices):
            if any([label in train_labels[idx] for label in eval_label]):
                matches[j] = 1
        top100_precisions.append(np.mean(matches))

    top100_precisions_train = []

    upto = 500
    dists, nns = knn.kneighbors(train_vectors, upto, return_distance=True)
    dists = -dists # we want largest to be most relevant

    #means = np.mean(dists, axis=0)
    #print(means.shape, means)

    #import matplotlib
    #import matplotlib.pyplot as plt
    #plt.plot(means)
    #plt.show()

    print(len(nns))
    knn_20s = []
    knns_20s_dists = []
    for i, nn_indices in enumerate(nns):
        knn_20s.append(nn_indices)
        knns_20s_dists.append(dists[i])
        #if i % 100 == 0:
        #print(i, len(nns))
        eval_label = train_labels[i]
        matches = np.zeros(upto)
        for j, idx in enumerate(nn_indices):
            if any([label in train_labels[idx] for label in eval_label]):
                matches[j] = 1
        top100_precisions_train.append(np.mean(matches))


    return knn_20s, knns_20s_dists, np.mean(top100_precisions), np.mean(top100_precisions_train)

import scipy.io
def make_STH_NNs(dataset_name, top=[200]):
    train_collection, train_collection_one_index, val_collection, val_collection_one_index, \
        test_collection, data_text, data_text_vect, labels, token2id, id2token = pickle.load(open(dataset_name + "_collections", "rb"))

    nn_file = "codes_" + dataset_name + "_collections_MATLAB.mat_STH"
    nn_path = "simple_baseline_results/" + nn_file

    vals = scipy.io.loadmat(nn_path)

    # 'codeTrain','codeTest','gndTrain','gndTest','fixed0_idx'
    codes = vals["codeTrain"][:, :64] # use 64 bits
    testcodes = vals["codeTest"][:, :64]
    idx = vals["fixed0_idx"]

    train_idx = np.concatenate((train_collection_one_index[-1], val_collection_one_index[-1]))
    org_train_labels = [labels[i] for i in train_idx]
    org_test_labels = [labels[i] for i in test_collection[-1]]

    knns, knns_dists, restest, resttrain = eval_hashing_and_returnKNN(codes, org_train_labels, testcodes, org_test_labels)

    # translate knns indices
    nns = {}
    nns_dists = {}
    for i, idx in enumerate(train_idx):
        nns[idx] = np.array([train_idx[validx] for validx in knns[i]])
        nns_dists[idx] = knns_dists[i]

    print(restest, resttrain)
    print(codes.shape)
    #print(knns_dists)

    def make_dataset(keys, extract=None):
        print(extract)
        topSmallCombos = []
        topSmallScores = []
        for index in keys:
            idx_scores = nns_dists[index]
            idx_nns = nns[index]

            if extract is None:
                extract = np.arange(2)

            idx_scores = idx_scores[extract]
            idx_nns = idx_nns[extract]

            for j in range( len(idx_nns)):
                for k in range(j+1, len(idx_nns)):
                    topSmallCombos.append([index, idx_nns[j], idx_nns[k]])
                    topSmallScores.append([idx_scores[j], idx_scores[k]])
        return topSmallCombos, topSmallScores

    train_indexes = train_collection_one_index[-1]
    val_indexes =  val_collection_one_index[-1]
    #test_indexes = test_collection[-1]

    print("train")
    for topval in top:
        print(topval)
        train_combos = make_dataset(train_indexes, extract=np.linspace(1,topval,10).astype(int) ) #np.arange(1,topval,10)
        train_combos_single = make_dataset(train_indexes)
        val_combos = make_dataset(val_indexes)

        print("train_combos avg:", len(train_combos[0])/len(train_indexes), len(train_indexes))
        print("train_combos_single avg:", len(train_combos_single[0])/len(train_indexes), len(train_indexes))
        print("val_combos avg:", len(val_combos[0])/len(val_indexes), len(val_indexes))

        train_u = set([train_combos[0][i][0] for i in range(len(train_combos[0]))])
        train_s_u = set([train_combos[0][i][0] for i in range(len(train_combos[0]))])
        val_u = set([val_combos[0][i][0] for i in range(len(val_combos[0]))])

        assert (train_u == set(train_indexes))
        assert(train_s_u == set(train_indexes))
        assert(val_u == set(val_indexes))

        pickle.dump([train_combos, train_combos_single, val_combos], open(dataset_name + "_STH_NNs", "wb"))

import sys
if __name__ == "__main__":

    # this script just creates the weak supervision NN pairs

    #dataset_name = "agnews"
    #dataset_name = "TMC"
    #dataset_name = "20news"
    dataset_name = "reuters"

    if True:
        for dataset_name in ["20news","TMC","reuters","agnews"]:#,"reuters","agnews"]:
            print(dataset_name)
            make_STH_NNs(dataset_name)
    
