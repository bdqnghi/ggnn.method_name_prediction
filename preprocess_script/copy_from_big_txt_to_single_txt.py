import os
from concurrent.futures import ProcessPoolExecutor
import copy
import sys
import argparse
import re
sys.path.append("../utils")
import identifier_splitting
from bidict import bidict
from pathlib import Path

regex = '\\"\s+([^"]+)\s+\\"'
excluded_tokens = [",","{",";","}",")","(",'"',"'","`",""," ","[]","[","]","/",":","."," "]
parser = argparse.ArgumentParser()
parser.add_argument("--node_type_path",
                    default="../preprocessed_data/node_type_vocab.txt", type=str, help="Type vocab")
parser.add_argument("--node_token_path",
                    default="../preprocessed_data/token_vocab.txt", type=str, help="Token vocab")
parser.add_argument("--label_path",
                    default="../preprocessed_data/train_label_vocab.txt", type=str, help="Label vocab")
parser.add_argument(
    "--input", default="../sample_data/java-small/training", type=str, help="Input path")
parser.add_argument(
    "--output", default="../sample_data/java-small-graph-transformed/training", type=str, help="Output path")

args = parser.parse_args()

if not os.path.exists(args.output):
    Path(args.output).mkdir(parents=True, exist_ok=True)


def exclude_tokens(all_vocabularies):
    temp_vocabs = []
    for vocab in all_vocabularies:
        if vocab not in excluded_tokens:
            temp_vocabs.append(vocab)
    return temp_vocabs


def process_token(token):
    for t in excluded_tokens:
        token = token.replace(t, "")
    return token

def main():

    input_path = args.input
    output_path = args.output

    node_type_lookup = {}
    node_token_lookup = {}
    label_lookup = {}

    token_vocabulary_path = args.node_token_path
    node_type_vocabulary_path = args.node_type_path
    label_vocabulary_path = args.label_path

    with open(token_vocabulary_path, "r") as f1:
        data = f1.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_token_lookup[splits[1]] = int(splits[0])

    with open(node_type_vocabulary_path, "r") as f2:
        data = f2.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            node_type_lookup[splits[1]] = int(splits[0])
    
    with open(label_vocabulary_path, "r") as f3:
        data = f3.readlines()
        for line in data:
            splits = line.replace("\n", "").split(",")
            label_lookup[splits[1]] = int(splits[0])
    
    node_type_lookup = bidict(node_type_lookup)
    node_token_lookup = bidict(node_token_lookup)
    label_lookup = bidict(label_lookup)
    print(label_lookup)

    for subdir, dirs, files in os.walk(input_path):  
        for file in files:
            if file.endswith(".txt"):
                graphs_path = os.path.join(subdir,file)

                single_graph_file = []
                with open(graphs_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                       
                        # print(line)
                        line = line.replace("\n", "")
                        line = line.replace("'", "")
                        line = " ".join(line.split())
                        # line = strip(line)
                        # line
                        # print(line)
                        
                        new_line_arr = []
                        splits = line.split(" ")
                        if splits[0] != "?":
                    
                            single_graph_file.append(line)
                        else:
                        
                            single_graph_file.append(line)

                            # Reset the graph file object
                            single_graph_file = []                
                                    



if __name__ == "__main__":
    main()
