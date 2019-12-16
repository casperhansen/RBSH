import tensorflow as tf
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from scipy import sparse
from graphs import matrix_for_embed, make_embedding, make_importance_embedding
from graphs_weak import make_graph_queue_SingleWeak
from gensim.models import KeyedVectors
from generators import DataGenerator_singleWeak, generator_singleWeak_version_eval
import time
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import datetime
import os

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
    #data_text_vect = data_text_vect.todense()
    print("### vocab", data_text_vect.shape)
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
                #print(semvector[i])
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


def eval_hashing(train_vectors, train_labels, val_vectors, val_labels, medianTrick=False):

    train_vectors = np.array(train_vectors)
    val_vectors = np.array(val_vectors)

    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)

    upto = 100
    knn = NearestNeighbors(n_neighbors=upto, metric="hamming", n_jobs=-1)
    #knn = NearestNeighbors(n_neighbors=upto, n_jobs=-1)

    knn.fit(train_vectors)

    nns = knn.kneighbors(val_vectors, upto, return_distance=False)
    top100_precisions = []
    for i, nn_indices in enumerate(nns):
        eval_label = val_labels[i]
        matches = np.zeros(upto)
        for j, idx in enumerate(nn_indices):
            if any([label in train_labels[idx] for label in eval_label]):
                matches[j] = 1
        top100_precisions.append(np.mean(matches))

    return np.mean(top100_precisions)

def eval_hashing_multi(train_vectors, train_labels, val_vectors, val_labels, medianTrick=False):

    train_vectors = np.array(train_vectors)
    val_vectors = np.array(val_vectors)

    if medianTrick:
        medians = np.median(train_vectors, 0)
        train_vectors = (train_vectors > medians).astype(int)
        val_vectors = (val_vectors > medians).astype(int)

    upto = 100
    knn = NearestNeighbors(n_neighbors=upto, metric="hamming", n_jobs=-1)
    #knn = NearestNeighbors(n_neighbors=upto, n_jobs=-1)

    knn.fit(train_vectors)

    mean_val_vectors = np.median(val_vectors, axis=0)
    nns = []
    nns.append(knn.kneighbors(mean_val_vectors, upto, return_distance=False))
    for i in range(len(val_vectors)):
        nns1 = knn.kneighbors(val_vectors[i], upto, return_distance=False)
        nns.append(nns1)

    #print([len(nn) for nn in nns])

    top100_precisions = []
    for i in range(len(nns1)):
        eval_label = val_labels[i]

        top100s = []
        for k in range(len(nns)):
            matches = np.zeros(upto)
            for j, idx in enumerate(nns[k][i]):
                if any([label in train_labels[idx] for label in eval_label]):
                    matches[j] = 1
            top100s.append(np.mean(matches))

        best_top100 = max(top100s) # first is avg vector, dont use this in here

        top100_precisions.append(top100s + [best_top100])

    ttt = np.array(top100_precisions)
    return ttt[:, 0], np.mean(top100_precisions, axis=0)

def as_matrix(config):
    return [[k, str(w)] for k, w in config.items()]

if __name__ == "__main__":

    #if False:
    #    make_collections(dataset_name)

    parser = argparse.ArgumentParser()
    # These 4 are the primary variables to change
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument("--rank_inc", default=10, type=float)
    parser.add_argument("--rank_val", default=0.5, type=float)
    parser.add_argument("--KL_inc", default=5, type=int) 

    # These are not model parameters, but needs to be changed for running experiments
    parser.add_argument('--filename', default="output1.txt", type=str)
    parser.add_argument("--pickle_add", default="newnew1", type=str) 
    parser.add_argument("--bits", default=32, type=int)
    parser.add_argument("--dname", default="reuters", type=str)
    parser.add_argument("--eval_every", default=5, type=float) # determines how often to evaluate the performance. vary this depending on convergence. Eg. for 20news it should be high ~20-30, but other are relatively fast to converge

    # You can keep these fixed
    parser.add_argument("--deterministic_eval", default=1, type=int)
    parser.add_argument("--test_repeats", default=1, type=int)
    parser.add_argument("--cont_cut", default=0.0, type=float)
    parser.add_argument("--noise_type", default="annealing", type=str) #none, learned, annealing (annealing is used in the paper)
    parser.add_argument("--annealing_min", default=0.0, type=float)
    parser.add_argument("--annealing_max", default=1.0, type=float)
    parser.add_argument("--annealing_decrement", default=1000000, type=float)
    parser.add_argument("--pretrain_emb", default=1, type=int) # not supported, 0 and 1 does the same.
    parser.add_argument("--input_emb", default=1, type=int)
    parser.add_argument("--different_embs", default=0, type=int)
    parser.add_argument("--use_sigma_directly", default=0, type=int)
    parser.add_argument("--max_iter_val", default=350000, type=int)
    parser.add_argument("--hinge_val", default=1.0, type=float)
    parser.add_argument("--kl_max_val", default=0.04, type=float)
    parser.add_argument("--drop_out", default=0.9, type=float)
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--embedding_size', default=300, type=int)

    #parser.add_argument("--NNs", default=None, type=int)

    args = parser.parse_args()
    args.deterministic_eval = args.deterministic_eval > 0.5
    args.pretrain_emb = args.pretrain_emb > 0.5

    no_ranking_loss = args.rank_inc < 0.001 and args.rank_val < 0.001
    print("#### do not use ranking", no_ranking_loss)

    dataset_name = args.dname
    train_collection, train_collection_one_index, val_collection, val_collection_one_index, \
        test_collection, data_text, data_text_vect, labels, token2id, id2token = pickle.load(open(dataset_name + "_collections", "rb"))
    val_collection = val_collection_one_index

    train_combos11, train_combos_single11, val_combos11 = pickle.load(open(dataset_name + "_STH_NNs", "rb"))

    train_collection = (train_combos11[0], train_combos11[1], train_collection[-1])
    train_collection_one_index = (train_combos_single11[0], train_combos_single11[1], train_collection_one_index[-1])
    val_collection = (val_combos11[0], val_combos11[1], val_collection[-1])

    train_combos11 = None
    train_combos_single11 = None
    val_combos11 = None

    diffs = 0
    eqs = 0
    for row in train_collection[1]:
        if abs(row[0]-row[1]) < 1e-10:
            eqs += 1
        else:
            diffs += 1

    print(diffs, eqs, diffs+eqs)
    print(diffs/(diffs+eqs), eqs/(diffs+eqs))

    pickle_file_name = "savefiles-our/" + "_".join([dataset_name, args.pickle_add, str(args.bits),
                                                str(args.annealing_decrement), str(args.annealing_min), str(args.annealing_max),str(args.pretrain_emb),
                                                str(args.rank_inc), str(args.rank_val), str(args.learning_rate)])
    tmpp_name = pickle_file_name
    kk = 1
    while os.path.exists(tmpp_name):
        tmpp_name = pickle_file_name + "_" + str(kk)
        kk += 1
    pickle_file_name = tmpp_name

    args.input_emb = args.input_emb > 0.5
    args.different_embs = args.different_embs > 0.5
    args.use_sigma_directly = args.use_sigma_directly > 0.5

    if args.pretrain_emb:
        args.embedding_size = 300
    else:
        args.embedding_size = 300 #args.bits

    args.annealing_decrement = 1.0/args.annealing_decrement
    if args.noise_type == "none":
        args.noise_type = 0
    elif args.noise_type == "learned":
        args.noise_type = 1
    elif args.noise_type == "annealing":
        args.noise_type = 2
    else:
        print("unknown:", args.noise_type)
        exit()

    output_file = args.filename
    print(args)
    with open(output_file, "a") as myfile:
        myfile.write(str(args) + "\n")
    weak_signals = 1

    # data specific parameters
    vocab_size = len(id2token)
    maskvalue = 0

    print(dataset_name ,"vocab_size" ,vocab_size)
    #print("------------", id2token)

    # embedding setup parameters
    embedding_dim = args.embedding_size
    trainable_embedding = True
    use_pretrain_emb = args.pretrain_emb

    if use_pretrain_emb or args.input_emb:
        w2v  = {}
        embInit = matrix_for_embed(w2v, [], vocab_size, 300) #just use randomly init embedding

    batch_size = args.batch_size
    initial_learning_rate = args.learning_rate
    dropout_keep = args.drop_out

    learning_rate = initial_learning_rate
    decay_rate = 0.97
    step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate,
                                    step,
                                    15000*20/batch_size,
                                    decay_rate,
                                    staircase=True, name="lr")
    optimizer = tf.train.AdamOptimizer(learning_rate=lr,name="AdamOptimizer") 

    max_epochs = 2000

    embedding_variable, embedding_placeholder, embedding_init = make_embedding(vocab_size, embedding_dim, name="word_embedding",
                                                                      trainable=trainable_embedding, init=use_pretrain_emb)

    importance_embedding = make_importance_embedding(vocab_size)

    input_embedding = None
    if args.input_emb:
        if args.different_embs:
            input_embedding, input_embedding_placeholder, input_embedding_init = make_embedding(vocab_size, 300, name="word_embedding_input",
                                                                      trainable=trainable_embedding, init=True)
        else:
            input_embedding = embedding_variable


    doc_bow = tf.placeholder(tf.float32, [None, vocab_size], name="doc_bow_test")
    #doc_words = tf.placeholder(tf.int32, [None, None], name="doc_words_test")

    doc1_bow = tf.placeholder(tf.float32, [None, vocab_size], name="doc1_bow_test")
    doc2_bow = tf.placeholder(tf.float32, [None, vocab_size], name="doc2_bow_test")

    doc1_weak_signals = tf.placeholder(tf.float32, [None], name="doc1weak_test")
    doc2_weak_signals = tf.placeholder(tf.float32, [None], name="doc2weak_test")

    masking = tf.placeholder(tf.float32, shape=[None], name="masking_test")  # 1 if used, 0 if filler sample

    is_eval = tf.placeholder(tf.bool, name="eval_bool") # used for evaluating in case of val and test sets
    kl_weight = tf.placeholder(tf.float32, name="KL_Weight")
    rank_weight = tf.placeholder(tf.float32, name="Rank_Weight")

    sigma_annealing = tf.placeholder(tf.float32, name="sigma_annealing")

    query_test_vec = np.zeros((1, vocab_size)).astype(float)
    doc1_test_vec = query_test_vec
    doc2_test_vec = query_test_vec
    doc1_weak_signals_test_vec = np.zeros(1).astype(float)
    doc2_weak_signals_test_vec = doc1_weak_signals_test_vec
    mask = np.ones(1).astype(float)

    train_dummy_dic = {doc_bow: query_test_vec, masking: mask, is_eval: False,
                       doc1_bow: doc1_test_vec, doc2_bow: doc2_test_vec, doc1_weak_signals: doc1_weak_signals_test_vec,
                       doc2_weak_signals: doc2_weak_signals_test_vec}


    test_placeholder_set = [doc_bow, doc1_bow, doc2_bow, doc1_weak_signals, doc2_weak_signals, masking]


    coord = tf.train.Coordinator()
    with tf.name_scope('create_inputs'):
        reader = DataGenerator_singleWeak(coord, batch_size, train_collection[0], train_collection[1], data_text_vect, vocab_size, maskval=0)
        input_batch = reader.dequeue()

    init, grad, loss, input_dict, sem_vector_bit, sem_vector_cont, \
    evalval, dist1, dist2, signpart, rank_loss, sum_merged, \
    loss_unweight, importance_embedding_val, recon_per_word = make_graph_queue_SingleWeak(sigma_annealing, rank_weight, kl_weight, args.bits, dropout_keep, vocab_size,
                                                                   args.embedding_size, embedding_variable, importance_embedding,
                                                                   optimizer, batch_size, input_batch, test_placeholder_set, is_eval, maskvalue,
                                                                   deterministic_eval=args.deterministic_eval, noise_type=args.noise_type,
                                                                   pretrained_emb=use_pretrain_emb, emb_input_embedding=input_embedding,
                                                                   use_sigma_directly=args.use_sigma_directly, use_ranking=(not no_ranking_loss),
                                                                   hinge_val=args.hinge_val)

    print("made graph")
    pickle_save = [] # each save should be [args, test_scores, train_loss, val_loss, test_loss, train_p100, test_p100]
    with tf.Session() as sess:

        if use_pretrain_emb:
            sess.run(embedding_init, feed_dict={embedding_placeholder: embInit})
            print("initialized embedding")

        print("session opened")
        sess.run(init)
        runned_epochs = 0
        saver = tf.train.Saver()

        threads = reader.start_threads(sess)

        iter = 0
        start = time.time()
        losses = []

        print("----starting epochs!")
        lowest_val = 100000
        no_better = 0
        lenTrain = len(train_collection_one_index[0])

        kl_weight_value = 0.
        kl_inc = 1 / (float(args.KL_inc) * 15000 * batch_size/20)  # set the annealing rate for KL loss

        rank_weight_value = args.rank_val
        if no_ranking_loss:
            rank_inc = 0
            rank_weight_value = 0
        else:
            rank_inc = 1 / (float(args.rank_inc) * 15000 * batch_size/20)

        annealing_value = args.annealing_max


        best_train = []
        best_val = []

        import os
        logs_dir = os.path.abspath(os.path.expanduser('logs/'))
        run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("logs dir:", logs_dir)
        summary_writer = tf.summary.FileWriter(logs_dir+"/"+run_time+"_"+str(np.random.random()), graph=sess.graph)
        config_runner_summary = tf.summary.text('config', tf.convert_to_tensor(as_matrix(vars(args))),
                                                collections=[])

        safe_names = ["train_p100", "val_p100", "test_p100", "test_loss", "val_loss"]
        pythonVars = []
        tensorboardVars = []
        sumrs = []
        for name in safe_names:
            pythonVar = tf.placeholder(tf.float32, [])
            tensorboardVar = tf.Variable(0, dtype=tf.float32, name=name)
            update_tensorboardVar = tensorboardVar.assign(pythonVar)
            v = tf.summary.scalar(name, tensorboardVar)

            pythonVars.append(pythonVar)
            tensorboardVars.append(update_tensorboardVar)
            sumrs.append(v)

        val_sum = tf.summary.merge(sumrs)

        summary_writer.add_summary(sess.run(config_runner_summary))

        #while runned_epochs < max_epochs:
        already_printed_best = True
        start = time.time()
        for _ in (range(args.max_iter_val)):
            train_dummy_dic[kl_weight] = kl_weight_value
            train_dummy_dic[rank_weight] = rank_weight_value
            train_dummy_dic[sigma_annealing] = annealing_value

            if iter % 200 == 0:
                #print("AVG time:", (time.time() - start) / 200)

                _, sum = sess.run([grad, sum_merged], feed_dict=train_dummy_dic)
                summary_writer.add_summary(sum, iter)
                #start = time.time()

            else:
                sess.run([grad], feed_dict=train_dummy_dic)

            used_kl_weight_value = kl_weight_value
            kl_weight_value = min(kl_weight_value + kl_inc, args.kl_max_val)

            used_rank_weight_value = rank_weight_value
            rank_weight_value = min(rank_weight_value + rank_inc, 30.0)
            if no_ranking_loss:
                rank_weight_value = 0

            annealing_value = max(annealing_value-args.annealing_decrement, args.annealing_min)


            iter += 1
            runned_epochs = (iter*batch_size)/lenTrain

            run_iters = int(args.eval_every * 15000 / 20)
            if iter % run_iters == 0 and iter >= 0:

                start = time.time()

                # val
                gen = generator_singleWeak_version_eval(labels, val_collection[0], val_collection[1], data_text_vect, batch_size, vocab_size)
                val_vectors, val_labels, val_losses, val_rank_losses = get_vectors_labels(gen, input_dict, sess, loss, rank_loss,
                                                                         sem_vector_bit, is_eval, used_kl_weight_value,
                                                                         used_rank_weight_value, cont_vec=sem_vector_cont)

                print("iter/epochs",iter, runned_epochs, "val loss:", np.mean(val_losses), "rank loss:", np.mean(val_rank_losses))
                print("used KL", used_kl_weight_value)
                if args.noise_type == 2:
                    print("used annealing", annealing_value)

                if np.mean(val_losses) < lowest_val:
                    test_repeats = []
                    test_repeats_losses = []
                    test_repeats_rank_losses = []
                    for _ in range(args.test_repeats): # args.test_repeats should just be = 1 for the traditional setting for 1 hash code.
                        gen = generator_singleWeak_version_eval(labels, test_collection[0], test_collection[1],
                                                                data_text_vect, batch_size, vocab_size)

                        test_vectors, test_labels, test_losses, test_rank_losses = get_vectors_labels(gen, input_dict, sess, loss, rank_loss,
                                                                                 sem_vector_bit, is_eval,
                                                                                 used_kl_weight_value,
                                                                                 used_rank_weight_value, cont_vec=sem_vector_cont)
                        test_repeats.append(test_vectors)
                        test_repeats_losses.append(test_losses)
                        test_repeats_rank_losses.append(test_rank_losses)

                    already_printed_best = False
                    best_val = val_vectors, val_labels, val_losses, val_rank_losses
                    best_test = test_repeats, test_labels, test_repeats_losses, test_repeats_rank_losses
                    no_better = 0

                    gen = generator_singleWeak_version_eval(labels, train_collection_one_index[0], train_collection_one_index[1],
                                                            data_text_vect, batch_size, vocab_size)
                    train_vectors, train_labels, train_losses, train_rank_losses = get_vectors_labels(gen, input_dict, sess, loss, rank_loss, sem_vector_bit,
                                                                                   is_eval, used_kl_weight_value, used_rank_weight_value, cont_vec=sem_vector_cont)
                    best_train = train_vectors, train_labels, train_losses, train_rank_losses

                    lowest_val = np.mean(val_losses)

                    print(iter, lowest_val)
                    with open(output_file, "a") as myfile:
                        myfile.write(str(iter) + " " + str(lowest_val) + "  rankloss" + str(np.mean(val_rank_losses)) + "\n")

                else:
                    no_better += 1

            if not already_printed_best: 
                already_printed_best = True
                print("###### eval!!!")
                train_vectors, train_labels, train_losses, train_rank_losses = best_train
                val_vectors, val_labels, val_losses, val_rank_losses = best_val
                test_repeats, test_labels, test_repeats_losses, test_repeats_rank_losses = best_test

                p100train = eval_hashing(train_vectors, train_labels, train_vectors, train_labels,
                                         medianTrick=False)
                p100val = eval_hashing(train_vectors, train_labels, val_vectors, val_labels, medianTrick=False)

                wholevals, p100test = eval_hashing_multi(train_vectors, train_labels, test_repeats, test_labels, medianTrick=False)
                test_loss = np.mean(test_repeats_losses)
                test_rank_loss = np.mean(test_repeats_rank_losses)
                print("test loss", test_loss)
                print("train loss", np.mean(train_losses), "train rank loss", np.mean(train_rank_losses))

                print(iter, "TRAIN avg P@100", p100train)
                print(iter, "VAL avg P@100", p100val)
                print(iter, "TEST avg P@100", p100test, np.mean(wholevals), wholevals.shape)

                sess.run([tensorboardVars[0]], feed_dict={pythonVars[0]: p100train})
                sess.run([tensorboardVars[1]], feed_dict={pythonVars[1]: p100val})

                sess.run([tensorboardVars[2]], feed_dict={pythonVars[2]: p100test[0]})
                sess.run([tensorboardVars[3]], feed_dict={pythonVars[3]: test_loss})

                sess.run([tensorboardVars[4]], feed_dict={pythonVars[4]: lowest_val})
                result = sess.run(val_sum)
                summary_writer.add_summary(result, iter)

                pickle_add = [args, wholevals, p100test, lowest_val, np.mean(val_rank_losses), p100train, np.mean(train_losses), np.mean(train_rank_losses)]
                pickle_save.append(pickle_add)
                pickle.dump(pickle_save, open(pickle_file_name, "wb"))

                with open(output_file, "a") as myfile:
                    myfile.write(str(iter) + " TRAIN avg P@100 " + str(p100train) + "\n")
                    myfile.write(str(iter) + " VAL avg P@100 " + str(p100val) + "\n")
                    myfile.write(str(iter) + " TEST avg P@100 " + str(p100test) + "\n")

            if no_better >= 20:
                print("break!")
                with open(output_file, "a") as myfile:
                    myfile.write("break!" + "\n")

                break