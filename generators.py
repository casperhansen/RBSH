import tensorflow as tf
import numpy as np
import pickle
import pandas as pd

import threading
import time

def generator_singleWeak_version_eval(labels, row_combos, row_weakSignals, text_matrix, batch_size, vocab_size):

    #perm = np.random.permutation(len(row_combos))
    row_combos = np.array(row_combos)#[perm]
    row_weakSignals = np.array(row_weakSignals)#[perm]

    N = len(row_combos)

    currentIndex = 0
    upto = batch_size

    while currentIndex < N:
        docs = []
        docs1 = []
        docs2 = []
        doc1_weak = []
        doc2_weak = []
        locallabels = []
        unique_docs = []

        for j in range(currentIndex, upto):
            #print(j)
            #print(row_combos[j])
            doc_idx, doc1_idx, doc2_idx = row_combos[j, 0], row_combos[j, 1], row_combos[j, 2]
            docs.append(text_matrix[doc_idx])
            docs1.append(text_matrix[doc1_idx])
            docs2.append(text_matrix[doc2_idx])
            doc1_weak.append(row_weakSignals[j, 0])
            doc2_weak.append(row_weakSignals[j, 1])
            locallabels.append(labels[doc_idx])
            unique_docs.append(doc_idx)

        docs_vector = np.zeros((batch_size, vocab_size))
        docs1_vector = np.zeros((batch_size, vocab_size))
        docs2_vector = np.zeros((batch_size, vocab_size))

        doc1_weak_vector = np.zeros(batch_size)
        doc2_weak_vector = np.zeros(batch_size)

        i0 = 0
        for i in range(upto-currentIndex):
            if i < N:
                docs_vector[i0] = docs[i].todense()
                docs1_vector[i0] = docs1[i].todense()
                docs2_vector[i0] = docs2[i].todense()

                doc1_weak_vector[i0] = doc1_weak[i]
                doc2_weak_vector[i0] = doc2_weak[i]

                i0 += 1

        currentIndex += batch_size
        upto += batch_size
        upto = np.min([upto, N])

        mask_filler_samples = np.ones(batch_size) # used when calculating loss
        mask_filler_samples[i0:] = 0

        yield docs_vector, docs1_vector, docs2_vector, doc1_weak_vector, doc2_weak_vector, mask_filler_samples, locallabels, unique_docs

def generator_singleWeak_version(row_combos, row_weakSignals, text_matrix, batch_size, vocab_size):

    perm = np.random.permutation(len(row_combos))
    row_combos = np.array(row_combos)[perm]
    row_weakSignals = np.array(row_weakSignals)[perm]

    N = len(row_combos)

    currentIndex = 0
    upto = batch_size

    while currentIndex < N:
        docs = []
        docs1 = []
        docs2 = []
        doc1_weak = []
        doc2_weak = []

        for j in range(currentIndex, upto):
            #print(j)
            #print(row_combos[j])
            doc_idx, doc1_idx, doc2_idx = row_combos[j, 0], row_combos[j, 1], row_combos[j, 2]
            switch = False
            if np.random.random() < 0.5:
                switch = True
                tmp = doc1_idx
                doc1_idx = doc2_idx
                doc2_idx = tmp
            docs.append(text_matrix[doc_idx])
            docs1.append(text_matrix[doc1_idx])
            docs2.append(text_matrix[doc2_idx])

            if not switch:
                doc1_weak.append(row_weakSignals[j, 0])
                doc2_weak.append(row_weakSignals[j, 1])
            elif switch:
                doc1_weak.append(row_weakSignals[j, 1])
                doc2_weak.append(row_weakSignals[j, 0])

        docs_vector = np.zeros((batch_size, vocab_size))
        docs1_vector = np.zeros((batch_size, vocab_size))
        docs2_vector = np.zeros((batch_size, vocab_size))

        doc1_weak_vector = np.zeros(batch_size)
        doc2_weak_vector = np.zeros(batch_size)

        i0 = 0
        for i in range(upto-currentIndex):
            if i < N:
                docs_vector[i0] = docs[i].todense()
                docs1_vector[i0] = docs1[i].todense()
                docs2_vector[i0] = docs2[i].todense()

                doc1_weak_vector[i0] = doc1_weak[i]
                doc2_weak_vector[i0] = doc2_weak[i]

                i0 += 1

        currentIndex += batch_size
        upto += batch_size
        upto = np.min([upto, N])

        mask_filler_samples = np.ones(batch_size) # used when calculating loss
        mask_filler_samples[i0:] = 0

        yield docs_vector, docs1_vector, docs2_vector, doc1_weak_vector, doc2_weak_vector, mask_filler_samples

class DataGenerator_singleWeak(object):
    def __init__(self,
                 coord,
                 batch_size,
                 row_combos,
                 row_weakSignals,
                 text_matrix,
                 vocab_size,
                 maskval=0,
                 max_queue_size=32,
                 wait_time=0.001):
        # Change the shape of the input data here with the parameter shapes.
        self.wait_time = wait_time
        self.max_queue_size = max_queue_size
        self.vocab_size = vocab_size

        self.batch_size = batch_size
        self.maskval = maskval

        self.row_combos = row_combos
        self.row_weakSignals = row_weakSignals
        self.text_matrix = text_matrix

        shapes = [[None, vocab_size], [None, vocab_size], [None, vocab_size], [None,], [None,], [None,]]
        dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]

        self.queue = tf.PaddingFIFOQueue(max_queue_size, dtypes=dtypes, shapes=shapes)
        self.queue_size = self.queue.size()
        self.threads = []

        self.coord = coord
        self.doc_bow = tf.placeholder(tf.float32, [None, vocab_size], name="doc_bow")

        self.doc1_bow = tf.placeholder(tf.float32, [None, vocab_size], name="doc1_bow")
        self.doc2_bow = tf.placeholder(tf.float32, [None, vocab_size], name="doc2_bow")

        self.doc1_weak_signals = tf.placeholder(tf.float32, [None], name="doc1weak")
        self.doc2_weak_signals = tf.placeholder(tf.float32, [None], name="doc2weak")
        self.masking = tf.placeholder(tf.float32, [None])  # 1 if used, 0 if filler sample

        self.enqueue = self.queue.enqueue([self.doc_bow, self.doc1_bow, self.doc2_bow, self.doc1_weak_signals, self.doc2_weak_signals, self.masking])

    def dequeue(self):
        output = self.queue.dequeue()#(num_elements)
        return output

    def thread_main(self, sess, id, maxthreads):
        stop = False
        N = len(self.row_combos)
        frac = 1.0/maxthreads
        fromval = int(id*frac*N)
        toval = int((id+1)*frac*N)

        #rowcombos = rowcombos[:512]
        with tf.device("/cpu:0"):
            while not stop:

                iterator = generator_singleWeak_version(self.row_combos[fromval:toval], self.row_weakSignals[fromval:toval],
                                                        self.text_matrix, self.batch_size, self.vocab_size)
                #print("starting over (new epoch)")
                for data in iterator:
                    #print(data[0])


                    while self.queue_size.eval(session=sess) >= (self.max_queue_size-1):
                        if self.coord.should_stop():
                            print("should stop!!", id)
                            break
                        time.sleep(self.wait_time)
                        #print("sleep", id)
                    if self.coord.should_stop():
                        stop = True
                        print("Enqueue thread receives stop request.")
                        break
                    doc, doc1, doc2, doc1weak, doc2weak, mask = data

                    #print(self.queue_size.eval(session=sess), id, fromval, toval)
                    # print(doc1_weak_signals[:5]-doc2_weak_signals[:5])
                    # print("iterator")
                    sess.run(self.enqueue, feed_dict={self.doc_bow: doc, self.masking: mask, self.doc1_bow: doc1,
                                                      self.doc2_bow: doc2, self.doc1_weak_signals: doc1weak,
                                                      self.doc2_weak_signals: doc2weak})

    def start_threads(self, sess, n_threads=4):
        for kk in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,kk,n_threads))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
