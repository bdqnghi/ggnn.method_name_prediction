#!/usr/bin/env/python
"""
Usage:
    chem_tensorflow_dense.py [options]

Options:
    -h --help                Show this screen.
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components.
    --evaluate               example evaluation mode using a restored model
"""

from typing import Sequence, Any
from collections import defaultdict
import numpy as np
import tensorflow as tf
import sys, traceback
import pdb
import json

from utils.utils import glorot_init



'''
Comments provide the expected tensor shapes where helpful.

Key to symbols in comments:
---------------------------
[...]:  a tensor
; ; :   a list
b:      batch size
e:      number of edge types (4)
v:      number of vertices per graph in this batch
h:      GNN hidden size
'''

class DenseGGNNModel():
    def __init__(self, opt):
        self.batch_size = opt.batch_size
        self.node_token_dim = opt.node_token_dim
        self.node_type_dim = opt.node_type_dim
        self.node_dim = self.node_type_dim + self.node_token_dim
        self.target_dim = self.node_type_dim + self.node_token_dim

        self.node_type_lookup = opt.node_type_lookup
        self.node_token_lookup = opt.node_token_lookup
        self.label_lookup = opt.label_lookup
        self.target_token_lookup = opt.target_token_lookup

        self.rnn_size = self.node_type_dim + self.node_token_dim
        self.num_rnn_layers = 1
        self.beam_width = 10
        self.is_evaluating = opt.task

        self.hidden_layer_size = opt.hidden_layer_size
        self.num_hidden_layer = opt.num_hidden_layer
        self.aggregation_type = opt.aggregation
        self.distributed_function = opt.distributed_function
        self.num_labels = opt.num_labels
        self.num_edge_types = opt.n_edge_types
        self.num_timesteps= opt.n_steps
        self.placeholders = {}
        self.weights = {}

        self.prepare_specific_graph_model()
        self.nodes_representation = self.compute_nodes_representation()
        # self.graph_representation = self.pooling_layer(self.nodes_representation)
        

        self.graph_representation = self.aggregation_layer(self.nodes_representation)

        
        self.fake_encoder_state = tuple(tf.nn.rnn_cell.LSTMStateTuple(self.graph_representation, self.graph_representation) for _ in range(self.num_rnn_layers))
        
        if self.is_evaluating == 0:
            print("Evaluating, tile batch contexts...........")
            self.nodes_representation = tf.contrib.seq2seq.tile_batch(self.nodes_representation, multiplier=self.beam_width)

        self.training_logits, self.inference_logits, self.training_final_state, self.inference_final_state = self.decoding_layer(targets=self.placeholders["targets"],
                                                target_token_embeddings=self.target_token_embeddings,
                                                contexts=self.nodes_representation,
                                                batch_size=self.batch_size,
                                                start_of_sequence_id=self.target_token_lookup["<GO>"], 
                                                end_of_sequence_id=self.target_token_lookup["<EOS>"], 
                                                encoder_state=self.fake_encoder_state, 
                                                target_vocab_size=len(self.target_token_lookup.keys()), 
                                                rnn_size=self.rnn_size,
                                                num_rnn_layers=self.num_rnn_layers,
                                                target_sequence_length=self.target_sequence_length,
                                                max_target_sequence_length=self.max_target_sequence_length)        
        self.training_output = tf.identity(self.training_logits.rnn_output, name='training_output')
        self.training_sample_id = tf.identity(self.training_logits.sample_id, name='training_sample_id')
        # self.inference_output = tf.identity(self.inference_logits.rnn_output, name='inference_output')
        # self.inference_sample_id = tf.identity(self.inference_logits.sample_id, name='inference_sample_id')

        self.loss = self.loss_layer(self.training_output, self.placeholders["targets"], self.target_mask)



    def prepare_specific_graph_model(self) -> None:
        node_dim = self.node_dim
        
        # initializer = tf.contrib.layers.xavier_initializer()
        # inputs
        # self.placeholders['graph_state_keep_prob'] = tf.placeholder(tf.float32, None, name='graph_state_keep_prob')
        # self.placeholders['edge_weight_dropout_keep_prob'] = tf.placeholder(tf.float32, None, name='edge_weight_dropout_keep_prob')
        self.node_type_embeddings = tf.Variable(glorot_init([len(self.node_type_lookup.keys()), self.node_type_dim]), name='node_type_embeddings')
        self.node_token_embeddings = tf.Variable(glorot_init([len(self.node_token_lookup.keys()), self.node_token_dim]), name='node_token_embeddings')
        self.target_token_embeddings = tf.Variable(glorot_init([len(self.target_token_lookup.keys()), self.target_dim]), name='target_token_embeddings')

        self.placeholders["node_type_indices"] = tf.placeholder(tf.int32, shape=[None,None], name='node_type_indices')
        self.placeholders["node_token_indices"] = tf.placeholder(tf.int32, shape=[None,None,None], name='node_token_indices')
    
        self.node_type_representations = tf.nn.embedding_lookup(self.node_type_embeddings, self.placeholders["node_type_indices"])
        self.node_token_representations = tf.nn.embedding_lookup(self.node_token_embeddings, self.placeholders["node_token_indices"])
        self.node_token_representations = tf.reduce_mean(self.node_token_representations, axis=2)
        
        # self.placeholders['initial_node_representation'] = tf.placeholder(tf.float32, [None, None, self.node_dim], name='node_features')
        self.placeholders['num_vertices'] = tf.placeholder(tf.int32, (),  name='num_vertices')
        # self.placeholders['labels'] = tf.placeholder(tf.int32, shape=[None,30], name='labels')

        self.placeholders['adjacency_matrix'] = tf.placeholder(tf.float32,[None, self.num_edge_types, None, None], name='adjacency_matrix')     # [b, e, v, v]
        self.__adjacency_matrix = tf.transpose(self.placeholders['adjacency_matrix'], [1, 0, 2, 3])                    # [e, b, v, v]
        
        # batch normalization
        self.placeholders['is_training'] = tf.placeholder(tf.bool, name="is_train")
        self.node_type_representations = tf.layers.batch_normalization(self.node_type_representations, training=self.placeholders['is_training'])
        self.node_token_representations = tf.layers.batch_normalization(self.node_token_representations, training=self.placeholders['is_training'])
    
        # weights
        self.weights['edge_weights'] = tf.Variable(glorot_init([self.num_edge_types, node_dim, node_dim]),name='edge_weights')
        self.weights['edge_biases'] = tf.Variable(tf.zeros([self.num_edge_types, 1, node_dim]),name='edge_biases')
        
        # self.weights["hidden_layer_weights"] = tf.Variable(xavier_initializer([self.node_dim, self.num_labels]), name='hidden_layer_weights')
        # self.weights["hidden_layer_biases"] = tf.Variable(xavier_initializer([self.num_labels,]), name='hidden_layer_biases')

        self.placeholders["targets"] =  tf.placeholder(tf.int32, shape=(None, None), name='targets') # batch_size x max_sequence
        self.placeholders["node_indicators"] = tf.placeholder(tf.float32, shape=(None, None), name='node_indicators')
        self.placeholders["length_targets"] = tf.placeholder(tf.int32, shape=(None,), name='length_targets')
        self.mask = tf.reshape(self.placeholders["node_indicators"], shape=[self.batch_size, tf.shape(self.placeholders["node_indicators"])[1], -1])
        self.target_sequence_length = tf.ones([self.batch_size], dtype=tf.int32)
        self.target_sequence_length = self.target_sequence_length * tf.shape(self.placeholders["targets"])[1]
        self.max_target_sequence_length = tf.reduce_max(self.target_sequence_length)
        self.target_mask = tf.sequence_mask(self.placeholders["length_targets"], self.max_target_sequence_length, dtype=tf.float32, name='target_mask')  

        with tf.variable_scope("gru_scope"):
            cell = tf.contrib.rnn.GRUCell(node_dim)
            # cell = tf.python.ops.rnn_cell.GRUCell(node_dim)
            # cell = tf.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            # cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, state_keep_prob=self.placeholders['graph_state_keep_prob'])
            self.weights['node_gru'] = cell


    def decoding_layer(self, targets, target_token_embeddings, contexts, batch_size, start_of_sequence_id, end_of_sequence_id, 
                    encoder_state, target_vocab_size, rnn_size, num_rnn_layers, target_sequence_length, max_target_sequence_length):
        
        decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_rnn_layers)])
       
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=rnn_size,
            memory=contexts
        )
       
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                           attention_layer_size=rnn_size)

        decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, output_keep_prob=0.5)
        
        output_layer = tf.layers.Dense(target_vocab_size)
        with tf.variable_scope("decode"):
            initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
            target_embeddings = tf.nn.embedding_lookup(target_token_embeddings, targets)
            helper = tf.contrib.seq2seq.TrainingHelper(target_embeddings, target_sequence_length)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer)
            training_logits, training_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                    impute_finished=True, 
                                                    maximum_iterations=max_target_sequence_length)
        with tf.variable_scope("decode", reuse=True):
            # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(target_token_embeddings, tf.fill([batch_size], start_of_sequence_id), end_of_sequence_id)
            # decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state, output_layer)
            # inference_logits, inference_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
            #                                         impute_finished=True, 
            #                                         maximum_iterations=4)


            initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                                batch_size=self.batch_size * self.beam_width)
            initial_state = initial_state.clone(cell_state=tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=self.beam_width))
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=target_token_embeddings,
                start_tokens=tf.fill([batch_size], start_of_sequence_id),
                end_token=end_of_sequence_id,
                initial_state=initial_state,
                beam_width=self.beam_width,
                output_layer=output_layer,
                length_penalty_weight=0.0)

            inference_logits, inference_final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=False, maximum_iterations=4)
      
    
        return training_logits, inference_logits, training_final_state, inference_final_state


    def compute_nodes_representation(self):
        node_dim = self.node_dim
        v = self.placeholders['num_vertices']
        # h = self.placeholders['initial_node_representation']                                                # [b, v, h]
        
        

        h = tf.concat([self.node_type_representations, self.node_token_representations], -1)
        h = tf.reshape(h, [-1, self.node_token_dim + self.node_type_dim])

        with tf.compat.v1.variable_scope("gru_scope") as scope:
            for i in range(self.num_timesteps):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                for edge_type in range(self.num_edge_types):
                    # print("edge type : " + str(edge_type))
                    # m = tf.matmul(h, tf.nn.dropout(self.weights['edge_weights'][edge_type], rate=1-self.placeholders['edge_weight_dropout_keep_prob'])) # [b*v, h]
                    m = tf.matmul(h, self.weights['edge_weights'][edge_type])                               # [b*v, h]

                    m = tf.reshape(m, [-1, v, node_dim])                                                       # [b, v, h]
                    m += self.weights['edge_biases'][edge_type]                                             # [b, v, h]

                    if edge_type == 0:
                        acts = tf.matmul(self.__adjacency_matrix[edge_type], m)
                    else:
                        acts += tf.matmul(self.__adjacency_matrix[edge_type], m)
                acts = tf.reshape(acts, [-1, node_dim])                                                        # [b*v, h]

                h = self.weights['node_gru'](acts, h)[1]                                                    # [b*v, h]
            last_h = tf.reshape(h, [-1, v, node_dim])
        return last_h

    def pooling_layer(self, nodes_representation):
        """Creates a max dynamic pooling layer from the nodes."""
        with tf.name_scope("pooling"):
            pooled = tf.reduce_max(nodes_representation, axis=1)
            return pooled

    def hidden_layer(self, pooled, input_size, output_size):
        """Create a hidden feedforward layer."""
        # self.weights["hidden_layer_weights"] = tf.Variable(xavier_initializer([self.node_dim, self.num_labels]), name='hidden_layer_weights')
        # self.weights["hidden_layer_biases"] = tf.Variable(xavier_initializer([self.num_labels,]), name='hidden_layer_biases')

        with tf.name_scope("hidden"):
            weights = tf.Variable(self.xavier_initializer([input_size, output_size]))
            biases = tf.Variable(self.xavier_initializer([output_size,]))
            return tf.nn.leaky_relu(tf.matmul(pooled, weights) + biases)

    def loss_layer(self, training_logits, targets, target_mask):
        """Create a loss layer for training."""
       

        with tf.name_scope('loss_layer'):
            loss = tf.contrib.seq2seq.sequence_loss(training_logits,targets,target_mask)

            return loss
    
    def aggregation_layer(self, nodes_representation):
        # conv is (batch_size, max_tree_size, output_size)
        with tf.name_scope("global_attention"):
            batch_size = tf.shape(nodes_representation)[0]
            max_tree_size = tf.shape(nodes_representation)[1]

            contexts_sum = tf.reduce_sum(nodes_representation, axis=1)
            contexts_sum_average = tf.divide(contexts_sum, tf.to_float(tf.expand_dims(max_tree_size, -1)))
          
            return contexts_sum_average

    # def aggregation_layer(self, nodes_representation, aggregation_type, distributed_function):
    #     # nodes_representation is (batch_size, max_graph_size, self.node_dim)
    #     w_attention = self.weights['attention_weights']
    #     with tf.name_scope("global_attention"):
    #         batch_size = tf.shape(nodes_representation)[0]
    #         max_tree_size = tf.shape(nodes_representation)[1]

    #         # (batch_size * max_graph_size, self.node_dim)
    #         flat_nodes_representation = tf.reshape(nodes_representation, [-1, self.node_dim])
    #         aggregated_vector = tf.matmul(flat_nodes_representation, w_attention)

    #         attention_score = tf.reshape(aggregated_vector, [-1, max_tree_size, 1])

    #         """A note here: softmax will distributed the weights to all of the nodes (sum of node weghts = 1),
    #         an interesting finding is that for some nodes, the attention score will be very very small, i.e e-12, 
    #         thus making parts of aggregated vector becomes near zero and affect on the learning (very slow nodes_representationergence
    #         - Better to use sigmoid"""

    #         if distributed_function == 0:
    #             attention_weights = tf.nn.softmax(attention_score, dim=1)
    #         if distributed_function == 1:
    #             attention_weights = tf.nn.sigmoid(attention_score)

    #         # TODO: reduce_max vs reduce_sum vs reduce_mean
    #         if aggregation_type == 1:
    #             print("Using tf.reduce_sum...........")
    #             weighted_average_nodes = tf.reduce_sum(tf.multiply(nodes_representation, attention_weights), axis=1)
    #         if aggregation_type == 2:
    #             print("Using tf.reduce_max...........")
    #             weighted_average_nodes = tf.reduce_max(tf.multiply(nodes_representation, attention_weights), axis=1)
    #         if aggregation_type == 3:
    #             print("Using tf.reduce_mean...........")
    #             weighted_average_nodes = tf.reduce_mean(tf.multiply(nodes_representation, attention_weights), axis=1)

    #         return weighted_average_nodes, attention_weights


    def softmax(self, logits_node):
        """Apply softmax to the output layer."""
        with tf.name_scope('output'):
            return tf.nn.softmax(logits_node)
