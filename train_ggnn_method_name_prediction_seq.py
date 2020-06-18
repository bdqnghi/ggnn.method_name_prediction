import argparse
import random

import pickle

import tensorflow as tf
from utils.data.method_name_prediction_dataset import MethodNamePredictionData
from utils.utils import ThreadedIterator
from utils.network.dense_ggnn2seq import DenseGGNNModel
import os
import sys
import re
import time

from bidict import bidict
import copy
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import evaluation
from scipy.spatial import distance
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int,
                    default=5, help='input batch size')
parser.add_argument('--train_batch_size', type=int,
                    default=5, help='train input batch size')
parser.add_argument('--test_batch_size', type=int,
                    default=5, help='test input batch size')
parser.add_argument('--val_batch_size', type=int,
                    default=5, help='val input batch size')
parser.add_argument('--state_dim', type=int, default=30,
                    help='GGNN hidden state dimension size')
parser.add_argument('--node_type_dim', type=int, default=50,
                    help='node type dimension size')
parser.add_argument('--node_token_dim', type=int,
                    default=100, help='node token dimension size')
parser.add_argument('--hidden_layer_size', type=int,
                    default=100, help='size of hidden layer')
parser.add_argument('--num_hidden_layer', type=int,
                    default=1, help='number of hidden layer')
parser.add_argument('--n_steps', type=int, default=10,
                    help='propagation steps number of GGNN')
parser.add_argument('--n_edge_types', type=int, default=7,
                    help='number of edge types')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--cuda', default="0", type=str, help='enables cuda')
parser.add_argument('--verbal', type=bool, default=True,
                    help='print training info or not')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--model_path', default="model",
                    help='path to save the model')
# parser.add_argument('--model_accuracy_path', default="model_accuracy/method_name.txt",
#                     help='path to save the the best accuracy of the model')
parser.add_argument('--n_hidden', type=int, default=50,
                    help='number of hidden layers')
parser.add_argument('--log_path', default="logs/",
                    help='log path for tensorboard')
parser.add_argument('--checkpoint_every', type=int,
                    default=500, help='check point to save model')
parser.add_argument('--validating', type=int,
                    default=1, help='validating or not')
parser.add_argument('--graph_size_threshold', type=int,
                    default=1000, help='graph size threshold')
parser.add_argument('--sampling_size', type=int,
                    default=60, help='sampling size for each epoch')
parser.add_argument('--best_f1', type=float,
                    default=0.0, help='best f1 to save model')
parser.add_argument('--aggregation', type=int, default=1, choices=range(0, 4),
                    help='0 for max pooling, 1 for attention with sum pooling, 2 for attention with max pooling, 3 for attention with average pooling')
parser.add_argument('--distributed_function', type=int, default=0,
                    choices=range(0, 2), help='0 for softmax, 1 for sigmoid')
parser.add_argument('--train_path', default="sample_data/java-small-graph-transformed/training",
                    help='path of training data')
parser.add_argument('--val_path', default="sample_data/java-small-graph-transformed/testing",
                    help='path of validation data')
parser.add_argument('--dataset', default="java-small",
                    help='name of dataset')
parser.add_argument('--node_type_vocabulary_path', default="preprocessed_data/node_type_vocab.txt",
                    help='name of dataset')
parser.add_argument('--token_vocabulary_path', default="preprocessed_data/ggnn/java-small/token_vocab.txt",
                    help='name of dataset')
parser.add_argument('--train_label_vocabulary_path', default="preprocessed_data/ggnn/java-small/train_label_vocab.txt",
                    help='name of dataset')
parser.add_argument('--val_label_vocabulary_path', default="preprocessed_data/ggnn/java-small/val_label_vocab.txt",
                    help='name of dataset')
parser.add_argument('--target_vocabulary_path', default="preprocessed_data/ggnn/java-small/label_sub_tokens.txt",
                    help='name of dataset')
parser.add_argument('--task', type=int, default=0,
                    choices=range(0, 3), help='1 for training, 0 for testing')
parser.add_argument('--predictions_ouput', default="predictions/original_predictions.txt",
                    help='name of dataset')          

# parser.add_argument('--pretrained_embeddings_url', default="embedding/fast_pretrained_vectors.pkl", help='pretrained embeddings url, there are 2 objects in this file, the first object is the embedding matrix, the other is the lookup dictionary')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.cuda

print(opt)

# opt.model_path = os.path.join(opt.model_path,)
# opt.model_path = os.path.join(opt.model_path,"method_name_prediction" + "_aggregation_" + str(opt.aggregation) + "_distributed_function_" + str(opt.distributed_function) + "_hidden_layer_size_" + str(opt.hidden_layer_size) + "_num_hidden_layer_"  + str(opt.num_hidden_layer) + "_node_type_dim_" + str(opt.node_type_dim) + "_node_token_dim_" + str(opt.node_token_dim))

def form_model_path(opt):
    model_traits = {}
    model_traits["dataset"] = str(opt.dataset)
    model_traits["aggregation"] = str(opt.aggregation)
    model_traits["distributed_function"] = str(opt.distributed_function)
    model_traits["node_type_dim"] = str(opt.node_type_dim)
    model_traits["node_token_dim"] = str(opt.node_token_dim)
    
    model_path = []
    for k, v in model_traits.items():
        model_path.append(k + "_" + v)
    
    return "ggnn2seq" + "_" + "-".join(model_path)

def load_vocabs(opt):

    train_label_lookup = {}
    node_type_lookup = {}
    node_token_lookup = {}
    val_label_lookup = {}
    target_token_lookup = {}

    node_type_vocabulary_path = opt.node_type_vocabulary_path
    train_label_vocabulary_path = opt.train_label_vocabulary_path
    token_vocabulary_path = opt.token_vocabulary_path
    val_label_vocabulary_path = opt.val_label_vocabulary_path
    target_vocabulary_path = opt.target_vocabulary_path

    with open(train_label_vocabulary_path, "r") as f1:
        data = f1.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            train_label_lookup[splits[1]] = int(splits[0])

    with open(node_type_vocabulary_path, "r") as f2:
        data = f2.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_type_lookup[splits[1]] = int(splits[0])

    with open(token_vocabulary_path, "r") as f3:
        data = f3.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_token_lookup[splits[1]] = int(splits[0])

    with open(val_label_vocabulary_path, "r") as f4:
        data = f4.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            val_label_lookup[splits[1]] = int(splits[0])

    with open(target_vocabulary_path, "r") as f3:
        data = f3.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            target_token_lookup[splits[1]] = int(splits[0])

    train_label_lookup = bidict(train_label_lookup)
    node_type_lookup = bidict(node_type_lookup)
    node_token_lookup = bidict(node_token_lookup)
    val_label_lookup = bidict(val_label_lookup)
    target_token_lookup = bidict(target_token_lookup)

    return train_label_lookup, node_type_lookup, node_token_lookup, val_label_lookup, target_token_lookup

def get_best_f1_score(opt):
    best_f1_score = 0.0
    
    try:
        os.mkdir("model_accuracy")
    except Exception as e:
        print(e)
    
    opt.model_accuracy_path = os.path.join("model_accuracy",form_model_path(opt) + ".txt")

    if os.path.exists(opt.model_accuracy_path):
        print("Model accuracy path exists : " + str(opt.model_accuracy_path))
        with open(opt.model_accuracy_path,"r") as f4:
            data = f4.readlines()
            for line in data:
                best_f1_score = float(line.replace("\n",""))
    else:
        print("Creating model accuracy path : " + str(opt.model_accuracy_path))
        with open(opt.model_accuracy_path,"w") as f5:
            f5.write("0.0")
    
    return best_f1_score
    
def main(opt):
    opt.model_path = os.path.join(opt.model_path, form_model_path(opt))
    checkfile = os.path.join(opt.model_path, 'cnn_tree.ckpt')
    ckpt = tf.train.get_checkpoint_state(opt.model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Continue training with old model : " + str(checkfile))

    train_label_lookup, node_type_lookup, node_token_lookup, val_label_lookup, target_token_lookup = load_vocabs(opt)

    opt.label_lookup = train_label_lookup
    opt.num_labels = len(train_label_lookup.keys())
    opt.node_type_lookup = node_type_lookup
    opt.node_token_lookup = node_token_lookup
    opt.target_token_lookup = target_token_lookup

    if opt.task == 1:
        train_dataset = MethodNamePredictionData(opt, opt.train_path, True, False, False)
    
    val_opt = copy.deepcopy(opt)
    val_opt.label_lookup = val_label_lookup
    val_opt.num_labels = len(val_label_lookup.keys())
    val_opt.node_token_lookup = node_token_lookup
    validation_dataset = MethodNamePredictionData(val_opt, opt.val_path, False, False, True)

    ggnn = DenseGGNNModel(opt)

 
    loss_node = ggnn.loss
    optimizer = tf.compat.v1.train.AdamOptimizer(opt.lr)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    params = tf.trainable_variables()
    gradients = tf.gradients(loss_node, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
    training_point = optimizer.apply_gradients(zip(clipped_gradients, params))

        # training_point = optimizer.minimize(loss_node)

    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)
    init = tf.global_variables_initializer()



    best_f1_score = get_best_f1_score(opt)
    print("Best f1 score : " + str(best_f1_score))
    with tf.Session() as sess:
        sess.run(init)

        if ckpt and ckpt.model_checkpoint_path:
            print("Continue training with old model")
            print("Checkpoint path : " + str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i, var in enumerate(saver._var_list):
                print('Var {}: {}'.format(i, var))

        if opt.task == 1:
            print("Training model.............")
            average_f1 = 0.0
        
            for epoch in range(1,  opt.epochs + 1):

                t0_train = time.time()
                train_batch_iterator = ThreadedIterator(
                    train_dataset.make_minibatch_iterator(), max_queue_size=1)
                for train_step, train_batch_data in enumerate(train_batch_iterator):
                    print("-------------------------------------")
                    # print(train_batch_data['labels_index'])
                    # print(train_batch_data["labels_sub_tokens"])
                    _, err = sess.run(
                        [training_point, loss_node],
                        feed_dict={
                            ggnn.placeholders["num_vertices"]: train_batch_data["num_vertices"],
                            ggnn.placeholders["adjacency_matrix"]:  train_batch_data['adjacency_matrix'],
                            ggnn.placeholders["node_type_indices"]: train_batch_data["node_type_indices"],
                            ggnn.placeholders["node_token_indices"]: train_batch_data["node_token_indices"],
                            ggnn.placeholders["targets"]: train_batch_data["labels_sub_tokens"],
                            ggnn.placeholders["length_targets"]: train_batch_data["length_targets"],
                            ggnn.placeholders["node_indicators"]: train_batch_data["node_indicators"],
                            ggnn.placeholders["is_training"]: True
                        }
                    )

                    # batch_ground_truth_sub_tokens = train_batch_data["labels_sub_tokens"]
                    # batch_ground_truth_sub_tokens_labels = []
                    # for ground_truth_sub_tokens in batch_ground_truth_sub_tokens:
                    #     ground_truth_sub_tokens_labels = []
                    #     for token_id in ground_truth_sub_tokens:
                    #         token = target_token_lookup.inverse[token_id]
                    #         ground_truth_sub_tokens_labels.append(token)
                    #     batch_ground_truth_sub_tokens_labels.append(ground_truth_sub_tokens_labels)

                    # batch_original_labels = train_batch_data["labels_index"]
                    # original_labels_token = []
                    # for original_label in batch_original_labels:
                    #     l = train_label_lookup.inverse[original_label]
                    #     original_labels_token.append(l)
                    # print(batch_ground_truth_sub_tokens_labels)
                    # print(original_labels_token)

                    # _, err = sess.run(
                    #     [training_point, loss_node],
                    #     feed_dict={
                    #         ggnn.placeholders["num_vertices"]: train_batch_data["num_vertices"],
                    #         ggnn.placeholders["adjacency_matrix"]:  train_batch_data['adjacency_matrix'],
                    #         ggnn.placeholders["node_type_indices"]: train_batch_data["node_type_indices"],
                    #         ggnn.placeholders["node_token_indices"]: train_batch_data["node_token_indices"],
                    #         ggnn.placeholders["targets"]: train_batch_data["labels_sub_tokens_onehot"],
                    #         ggnn.placeholders["length_targets"]: train_batch_data["length_targets"],
                    #         ggnn.placeholders["node_indicators"]: train_batch_data["node_indicators"],
                    #         ggnn.placeholders["is_training"]: True
                    #     }
                    # )
                    # scores = sess.run(
                    #     [ggnn.nodes_representation],
                    #     feed_dict={
                    #         ggnn.placeholders["num_vertices"]: train_batch_data["num_vertices"],
                    #         ggnn.placeholders["adjacency_matrix"]:  train_batch_data['adjacency_matrix'],
                    #         ggnn.placeholders["node_type_indices"]: train_batch_data["node_type_indices"],
                    #         ggnn.placeholders["node_token_indices"]: train_batch_data["node_token_indices"],
                    #         ggnn.placeholders["targets"]: train_batch_data["labels_sub_tokens"],
                    #         ggnn.placeholders["length_targets"]: train_batch_data["length_targets"],
                    #         ggnn.placeholders["node_indicators"]: train_batch_data["node_indicators"],
                    #         ggnn.placeholders["is_training"]: True
                    #     }
                    # )

                    # print(scores[0].shape)
                    print("Epoch:", epoch, "Step:", train_step, "Loss:", err, "Current F1:", average_f1, "Best F1:", best_f1_score)

                    # print(label_embeddings_matrix.shape)
                    

                    # if train_step % opt.checkpoint_every == 0 and train_step > 0:
                    #     saver.save(sess, checkfile)                  
                    #     print('Checkpoint saved, epoch:' + str(epoch) + ', step: ' + str(train_step) + ', loss: ' + str(err) + '.')

                # --------------------------------------
                    if opt.validating == 0:
                        if train_step % opt.checkpoint_every == 0 and train_step > 0:
                            saver.save(sess, checkfile)                  
                            print('Checkpoint saved, epoch:' + str(epoch) + ', step: ' + str(train_step) + ', loss: ' + str(err) + '.')
                
                    if opt.validating == 1:
                        if train_step % opt.checkpoint_every == 0 and train_step > 0:
                            print("Validating at epoch:", epoch)
                            # predictions = []
                            validation_batch_iterator = ThreadedIterator(
                                validation_dataset.make_minibatch_iterator(), max_queue_size=5)
                            
                            # f1_scores_of_val_data = []
                            all_predicted_labels = []
                            all_ground_truth_labels = []

                            for val_step, val_batch_data in enumerate(validation_batch_iterator):
                        
                                label_embeddings_matrix, scores = sess.run(
                                    [label_embeddings, logits],
                                    feed_dict={
                                        ggnn.placeholders["num_vertices"]: val_batch_data["num_vertices"],
                                        ggnn.placeholders["adjacency_matrix"]:  val_batch_data['adjacency_matrix'],
                                        ggnn.placeholders["node_type_indices"]: val_batch_data["node_type_indices"],
                                        ggnn.placeholders["node_token_indices"]: val_batch_data["node_token_indices"],
                        
                                        ggnn.placeholders["is_training"]: False
                                    }
                                )

                                
                                predictions = np.argmax(scores, axis=1)
                            
                                ground_truths = np.argmax(val_batch_data['labels'], axis=1)
                            
                                predicted_labels = []
                                for prediction in predictions:
                                    predicted_labels.append(train_label_lookup.inverse[prediction])

                                ground_truth_labels = []
                                for ground_truth in ground_truths:
                                    ground_truth_labels.append(
                                        val_label_lookup.inverse[ground_truth])
                                
                                # print("Predicted : " + str(predicted_labels))
                                # print("Ground truth : " + str(ground_truth_labels))
                                f1_score = evaluation.calculate_f1_scores(predicted_labels, ground_truth_labels)
                                print(ground_truth_labels)
                                print(predicted_labels)
                                print("F1:", f1_score, "Step:", val_step)
                                all_predicted_labels.extend(predicted_labels)
                                all_ground_truth_labels.extend(ground_truth_labels)

                            average_f1 = evaluation.calculate_f1_scores(all_predicted_labels, all_ground_truth_labels)
                            # print("F1 score : " + str(f1_score))
                            print("Validation with F1 score ", average_f1)
                            if average_f1 > best_f1_score:
                                best_f1_score = average_f1

                                checkfile = os.path.join(opt.model_path, 'cnn_tree.ckpt')
                                saver.save(sess, checkfile)

                                checkfile = os.path.join(opt.model_path + "_" + str(datetime.utcnow().timestamp()), 'cnn_tree.ckpt')
                                saver.save(sess, checkfile)

                                print('Checkpoint saved, epoch:' + str(epoch) + ', loss: ' + str(err) + '.')
                                with open(opt.model_accuracy_path,"w") as f1:
                                    f1.write(str(best_f1_score))
                t1_train = time.time()
                total_train = t1_train-t0_train
                # print("Epoch:", epoch, "Execution time:", str(total_train))

        if opt.task == 0:
            average_f1 = 0.0
            validation_batch_iterator = ThreadedIterator(
                validation_dataset.make_minibatch_iterator(), max_queue_size=5)
            
            all_predicted_labels = []
            all_ground_truth_labels = []
            for val_step, val_batch_data in enumerate(validation_batch_iterator):
                print("----------------------------------------")
                # print(val_batch_data['labels_index'])
                prediction_scores = sess.run(
                    [ggnn.inference_logits],
                    feed_dict={
                        ggnn.placeholders["num_vertices"]: val_batch_data["num_vertices"],
                        ggnn.placeholders["adjacency_matrix"]:  val_batch_data['adjacency_matrix'],
                        ggnn.placeholders["node_type_indices"]: val_batch_data["node_type_indices"],
                        ggnn.placeholders["node_token_indices"]: val_batch_data["node_token_indices"],
                        ggnn.placeholders["length_targets"]: val_batch_data["length_targets"],
                        ggnn.placeholders["node_indicators"]: val_batch_data["node_indicators"],
                        ggnn.placeholders["is_training"]: False
                    }
                )
        
                batch_predicted_indices = prediction_scores[0].predicted_ids
                batch_top_scores = prediction_scores[0].beam_search_decoder_output.scores
                # print(prediction_scores[0].predicted_ids)
                # print(prediction_scores[0].beam_search_decoder_output.scores)
                # print(prediction_scores[0].predicted_ids.shape)
                # print(prediction_scores[0].beam_search_decoder_output.scores.shape)
                # print(val_batch_data["batch_targets"])
                

                predicted_labels = []
                for i, predicted_indices in enumerate(batch_predicted_indices):
                    # print("++++++")
                    predicted_strings = [[target_token_lookup.inverse[sugg] for sugg in timestep]for timestep in predicted_indices]  # (target_length, top-k)  
                    predicted_strings = list(map(list, zip(*predicted_strings)))  # (top-k, target_length)
                    top_scores = [np.exp(np.sum(s)) for s in zip(*batch_top_scores[i])]
                    top_scores_max = np.argmax(top_scores)
                    
                    highest_score_prediction_string = predicted_strings[top_scores_max]
                    highest_score_prediction_string_temp = []
                    for token in highest_score_prediction_string:
                        if token != "<GO>" and token != "<PAD>" and token != "<EOS>":
                            highest_score_prediction_string_temp.append(token)
                    predicted_labels.append("_".join(highest_score_prediction_string_temp))

                # print(predicted_labels)

                # predicted_labels = []
                # for prediction_score in prediction_scores[0]:
                #     prediction = []
                #     for token_id in prediction_score:
                #         if token_id != 0 and token_id != 1 and token_id != 2:
                #             token = target_token_lookup.inverse[token_id]
                #             prediction.append(token)
                            
                #     predicted_labels.append("_".join(prediction))
                

                # ground_truths = np.argmax(val_batch_data['labels_index'], axis=1)
                ground_truth_labels = []
                for ground_truth in val_batch_data['labels_index']:
                    ground_truth_labels.append(
                        val_label_lookup.inverse[ground_truth])
                
                print("Predicted : " + str(predicted_labels))
                print("Ground truth : " + str(ground_truth_labels))
                f1_score = evaluation.calculate_f1_scores(predicted_labels, ground_truth_labels)
                print("F1:", f1_score, "Step:", val_step)
                all_predicted_labels.extend(predicted_labels)
                all_ground_truth_labels.extend(ground_truth_labels)

            average_f1 = evaluation.calculate_f1_scores(all_predicted_labels, all_ground_truth_labels)
            # print("F1 score : " + str(f1_score))
            print("Validation with F1 score ", average_f1)
         
          

if __name__ == "__main__":
    main(opt)
