import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops

import numpy as np



from tensorflow.losses import compute_weighted_loss, Reduction

def hinge_loss_eps(labels, logits, epsval, weights=1.0, scope=None,
               loss_collection=ops.GraphKeys.LOSSES,
               reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  if labels is None:
    raise ValueError("labels must not be None.")
  if logits is None:
    raise ValueError("logits must not be None.")
  with ops.name_scope(scope, "hinge_loss", (logits, labels, weights)) as scope:
    logits = math_ops.to_float(logits)
    labels = math_ops.to_float(labels)
    logits.get_shape().assert_is_compatible_with(labels.get_shape())
    # We first need to convert binary labels to -1/1 labels (as floats).
    all_eps = array_ops.ones_like(labels)*epsval
    all_ones = array_ops.ones_like(labels)

    labels = math_ops.subtract(2 * labels, all_ones)
    losses = nn_ops.relu(
        math_ops.subtract(all_eps, math_ops.multiply(labels, logits)))
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)


def make_graph_queue_SingleWeak(sigma_annealing, rank_weight, kl_weight, bits, dropout_keep, vocab_size,  emb_size,
                                embedding, importance_embedding, optimizer, batch_size, inputs, test_placeholder_set,
                                is_eval, maskvalue, output_activation_function=tf.nn.tanh, deterministic_eval=True,
                                noise_type=1, pretrained_emb=False, emb_input_embedding=None, use_sigma_directly=False,
                                use_ranking=True, hinge_val=1.0):

    print("network, use ranking", use_ranking)
    hidden_neurons_encode = 1000
    encoder_layers = 2

    used_input = tf.cond(is_eval, lambda: test_placeholder_set, lambda: inputs, name="train_or_test_cond")

    doc, doc1, doc2, doc1weak, doc2weak, masking = used_input

    if emb_input_embedding is not None:
        print("apply Importance embedding on docs")
        doc_enc  = doc * importance_embedding #tf.matmul(doc, tf.expand_dims(importance_embedding, -1) * emb_input_embedding)
        doc1_enc = doc1 * importance_embedding #tf.matmul(doc1, tf.expand_dims(importance_embedding, -1) * emb_input_embedding)
        doc2_enc = doc2 * importance_embedding #tf.matmul(doc2, tf.expand_dims(importance_embedding, -1) * emb_input_embedding)
    else:
        doc_enc = doc
        doc1_enc = doc1
        doc2_enc = doc2

    #################### Bernoulli Sample #####################
    ## ref code: https://r2rt.com/binary-stochastic-neurons-in-tensorflow.html
    def bernoulliSample(x):
        """
        Uses a tensor whose values are in [0,1] to sample a tensor with values in {0, 1},
        using the straight through estimator for the gradient.
        E.g.,:
        if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
        and the gradient will be pass-through (identity).
        """
        g = tf.get_default_graph()

        with ops.name_scope("BernoulliSample") as name:
            with g.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
                if deterministic_eval:
                    mus = tf.cond(is_eval, lambda: tf.ones(tf.shape(x))*0.5, lambda: tf.random_uniform(tf.shape(x)))
                else:
                    mus =  tf.random_uniform(tf.shape(x))
                return tf.ceil(x - mus, name=name)

    @ops.RegisterGradient("BernoulliSample_ST")
    def bernoulliSample_ST(op, grad):
        return [grad, tf.zeros(tf.shape(op.inputs[1]))]
    ###########################################################
    # encode
    def encoder(doc, hidden_neurons_encode, encoder_layers):
        doc_layer = tf.layers.dense(doc, hidden_neurons_encode, name="encode_layer0",
                        reuse=tf.AUTO_REUSE, activation=tf.nn.relu)
        #doc_layer = tf.nn.dropout(doc_layer, dropout_keep)

        for i in range(1,encoder_layers):
            doc_layer = tf.layers.dense(doc_layer, hidden_neurons_encode, name="encode_layer"+str(i),
                            reuse=tf.AUTO_REUSE, activation=tf.nn.relu)

        doc_layer = tf.nn.dropout(doc_layer, tf.cond(is_eval, lambda: 1.0, lambda: dropout_keep))

        doc_layer = tf.layers.dense(doc_layer, bits, name="last_encode", reuse=tf.AUTO_REUSE, activation=tf.nn.sigmoid)

        bit_vector = bernoulliSample(doc_layer)

        return bit_vector, doc_layer

    bit_vector, cont_vector = encoder(doc_enc, hidden_neurons_encode, encoder_layers)

    if use_ranking:
        bit_vector_doc1, cont1 = encoder(doc1_enc, hidden_neurons_encode, encoder_layers)
        bit_vector_doc2, cont2 = encoder(doc2_enc, hidden_neurons_encode, encoder_layers)

    # decode

    # transform s from [None, bits] into [None, emb_size]
    log_sigma2 = tf.layers.dense(cont_vector, bits, name="decode_logsigma2", activation=tf.nn.sigmoid)

    e = tf.random.normal([batch_size, bits])
    if noise_type == 2: #annealing
        print("use annealing")
        noisy_bit_vector = tf.math.multiply(e, sigma_annealing) + bit_vector
        #noisy_bit_vector = tf.maximum(noisy_bit_vector, 0)
    elif noise_type == 1: #learned
        if use_sigma_directly:
            print("use sigma directly")
            noisy_bit_vector = tf.math.multiply(e, log_sigma2) + bit_vector
        else:
            noisy_bit_vector = tf.math.multiply(e, tf.sqrt(tf.exp(log_sigma2))) + bit_vector
    elif noise_type == 0: #none
        noisy_bit_vector = bit_vector
    else:
        print("unknown noise_type", noise_type)
        exit()

    # s * Emb
    softmax_bias = tf.Variable(tf.zeros(vocab_size), name="softmax_bias")

    #print(importance_embedding, tf.transpose(embedding))
    #print( tf.multiply(tf.transpose(embedding), importance_embedding) )
    #exit()

    #if pretrained_emb:
    print("pretrained embedding downscaling layer")
    embedding = tf.layers.dense(embedding, bits, name="lower_dim_embedding_layer")

    dot_emb_vector = tf.linalg.matmul(noisy_bit_vector, tf.multiply(tf.transpose(embedding), importance_embedding) )  + softmax_bias

    softmaxed = tf.nn.softmax(dot_emb_vector)
    logaritmed = tf.math.log(tf.maximum(softmaxed, 1e-10))
    logaritmed = tf.multiply(logaritmed, tf.cast(doc > 0, tf.float32)) #tf.cast(doc>0, tf.float32)) # set words not occuring to 0

    # loss
    num_samples = tf.reduce_sum(masking)

    def my_dot_prod(a,b):
        return tf.reduce_sum(tf.multiply(a, b), 1)

    if use_ranking:
        use_dot = False
        if use_dot:
            bit_vector_sub = 2*bit_vector - 1
            bit_vector_doc1_sub = 2*bit_vector_doc1 - 1
            bit_vector_doc2_sub = 2*bit_vector_doc2 - 1
            dist1 = my_dot_prod(bit_vector_sub, bit_vector_doc1_sub) #tf.reduce_sum(tf.math.pow(bit_vector - bit_vector_doc1, 2), axis=1) #tf.norm(bit_vector - bit_vector_doc1, axis=1)
            dist2 = my_dot_prod(bit_vector_sub, bit_vector_doc2_sub) #tf.reduce_sum(tf.math.pow(bit_vector - bit_vector_doc2, 2), axis=1) #tf.norm(bit_vector - bit_vector_doc2, axis=1)
            signpart = tf.cast(doc1weak > doc2weak, tf.float32)
        else:
            dist1 = tf.reduce_sum(tf.math.pow(bit_vector - bit_vector_doc1, 2), axis=1) #tf.norm(bit_vector - bit_vector_doc1, axis=1)
            dist2 = tf.reduce_sum(tf.math.pow(bit_vector - bit_vector_doc2, 2), axis=1) #tf.norm(bit_vector - bit_vector_doc2, axis=1)
            signpart = tf.cast(doc1weak > doc2weak, tf.float32)

        if use_dot:
            rank_loss = hinge_loss_eps(labels=(signpart), logits=(dist1-dist2), epsval= hinge_val)#bits/4.0)
        else:
            equal_score = tf.cast( tf.abs(doc1weak - doc2weak) < 1e-10, tf.float32)
            unequal_score = tf.cast( tf.abs(doc1weak - doc2weak) >= 1e-10, tf.float32)

            rank_loss_uneq = hinge_loss_eps(labels=(signpart), logits=(dist2 - dist1), epsval=hinge_val, weights=unequal_score)#bits / 8.0)
            eq_dist = tf.abs(dist2 - dist1)
            rank_loss_eq = compute_weighted_loss( eq_dist, weights=equal_score, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)
            rank_loss = rank_loss_uneq + rank_loss_eq

    if use_ranking:
        e1 = tf.random.normal([batch_size, bits])
        e2 = tf.random.normal([batch_size, bits])

        noisy_bit_vector1 = tf.math.multiply(e1, sigma_annealing) + bit_vector_doc1
        noisy_bit_vector2 = tf.math.multiply(e2, sigma_annealing) + bit_vector_doc2

        dot_emb_vector1 = tf.linalg.matmul(noisy_bit_vector1,
                                          tf.multiply(tf.transpose(embedding), importance_embedding)) + softmax_bias
        dot_emb_vector2 = tf.linalg.matmul(noisy_bit_vector2,
                                      tf.multiply(tf.transpose(embedding), importance_embedding)) + softmax_bias

        softmaxed1 = tf.nn.softmax(dot_emb_vector1)
        logaritmed1 = tf.math.log(tf.maximum(softmaxed1, 1e-10))
        logaritmed1 = tf.multiply(logaritmed1, tf.cast(doc1 > 0, tf.float32))

        softmaxed2 = tf.nn.softmax(dot_emb_vector2)
        logaritmed2 = tf.math.log(tf.maximum(softmaxed2, 1e-10))
        logaritmed2 = tf.multiply(logaritmed2, tf.cast(doc2 > 0, tf.float32))

        loss_recon1 = tf.reduce_sum(tf.multiply(tf.reduce_sum(logaritmed1, 1), masking) / num_samples, axis=0)
        loss_recon2 = tf.reduce_sum(tf.multiply(tf.reduce_sum(logaritmed2, 1), masking) / num_samples, axis=0)

        doc_1_2_recon_loss = -(loss_recon1 + loss_recon2)

    # VAE loss part
    loss_recon = tf.reduce_sum( tf.multiply(tf.reduce_sum(logaritmed, 1), masking)/num_samples, axis=0)

    recon_per_word = logaritmed #print("--------", logaritmed)
    print("#################", importance_embedding)

    loss_kl = tf.multiply(cont_vector,     tf.math.log( tf.maximum(cont_vector/0.5, 1e-10) )) + \
              tf.multiply(1 - cont_vector, tf.math.log( tf.maximum((1 - cont_vector)/0.5, 1e-10) ))

    loss_kl = tf.reduce_sum( tf.multiply(tf.reduce_sum(loss_kl, 1), masking)/num_samples, axis=0)

    loss_vae = -(loss_recon - kl_weight*loss_kl)
    if use_ranking:
        loss_rank_weighted = rank_weight * rank_loss
        loss = loss_rank_weighted + loss_vae + doc_1_2_recon_loss # we want to maximize, but Adam only support minimize

        loss_rank_unweighted = rank_loss - (loss_recon - kl_weight*loss_kl) + doc_1_2_recon_loss
    else:
        loss = loss_vae
        rank_loss = loss*0
        loss_rank_weighted = -1
        loss_rank_unweighted = -1
        dist1 = -1
        dist2 = -1
        signpart = -1

        rank_loss_eq = rank_loss
        rank_loss_uneq = rank_loss

    tf.summary.scalar('loss_vae', loss_vae)
    tf.summary.scalar('loss_kl', loss_kl)
    tf.summary.scalar('loss_recon', loss_recon)
    tf.summary.scalar('loss_rank_raw', rank_loss)
    tf.summary.scalar('loss_rank_weighted', loss_rank_weighted)
    tf.summary.scalar('loss_total', loss)
    tf.summary.scalar("kl_weight", kl_weight)
    tf.summary.scalar("rank_weight", rank_weight)
    tf.summary.scalar("unweighted_loss", loss_rank_unweighted)

    tf.summary.scalar('loss_rank_raw_eq', rank_loss_eq)
    tf.summary.scalar('loss_rank_raw_uneq', rank_loss_uneq)

    #tf.summary.scalar("learned sigma value", tf.reduce_sum(log_sigma2)/(num_samples*bits))
    #tf.summary.scalar("learned sigma value (as used)", tf.reduce_sum(tf.sqrt(tf.exp(log_sigma2)))/(num_samples*bits))
    tf.summary.scalar("sigma annealing value", sigma_annealing)

    print("vae",loss_vae)
    print("recon",loss_recon)
    print("kl",loss_kl)
    print("rank weighted",loss_rank_weighted)
    print("rank loss", rank_loss)
    print("total loss", loss)
    print("kl weight", kl_weight)

    # optimize
    grad = optimizer.minimize(loss)
    init = tf.global_variables_initializer()

    # make input dictionary
    def input_dict(docval, maska):
        return {doc: docval, masking: maska}

    merged = tf.summary.merge_all()
    return init, grad, loss, input_dict, bit_vector, cont_vector, is_eval, dist1, dist2, signpart, \
           rank_loss, merged, loss_rank_unweighted, importance_embedding, recon_per_word


