# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# TODO I have export PYTHONPATH="${PYTHONPATH}:/workspace/megatron/codeformer_utils/vendor/codeformer" to import jetnn
import time
import gzip
import glob
import torch
import multiprocessing
from codeformer_utils.vendor.codeformer import (
    # AstCodeSplitter,
    MyTextTree,
    MyCodeTree,
    transform_sequence_according_to_split_with_begin_end_tokens
)

import pydevd_pycharm

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


PORT_DEBUG = 2000
# pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
# class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
#
#     _period_context_fmt = r"""
#         \S*                          # some word material
#         %(SentEndChars)s             # a potential sentence ending
#         \s*                       #  <-- THIS is what I changed
#         (?=(?P<after_tok>
#             %(NonWord)s              # either other punctuation
#             |
#             (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
#         ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args
        self.task = self.args.task

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        path_to_tree_sitter = f"{self.args.tree_sitter_path}/tree-sitter-{self.args.language}"
        if self.task == "method_naming":
            Encoder.splitter = MyCodeTree(self.args.language, path_to_tree_sitter)
            Encoder.process_input = Encoder.splitter.process_code
            # config = {
            #     "programming_language": self.args.language,
            #     "path_to_tree_sitter": f"{self.args.tree_sitter_path}/tree-sitter-{self.args.language}",
            #     "max_code_parts": self.args.max_context_length,
            #     "max_subsequence_size": self.args.max_sent_length,
            #     "max_subsequences_number": self.args.max_sent_num,
            # }
            # Encoder.splitter = AstCodeSplitter(config, Encoder.tokenizer)
        else:
            Encoder.splitter = MyTextTree()
            Encoder.process_input = Encoder.splitter.process_text

    def split_and_tokenize(self, json_line):
        data_key = None
        # TODO may be pass from args
        if "code" in self.args.json_keys:
            data_key = "code"
            label_key = "label"
        if "text" in self.args.json_keys:
            data_key = "text"
            label_key = None
        try:
            data = json.loads(json_line)
            input_data = data[data_key]
            if label_key is not None:
                label = data[label_key]
        except:
            # TODO this is a brute-force solution. Refactor it!
            if len(self.args.json_keys)==1 and data_key=='text':
                input_data = json_line[10:-2]
            else:
                return [], [0]

        pad_id = Encoder.tokenizer.pad
        bos_id = Encoder.tokenizer.bos
        eos_id = Encoder.tokenizer.eos

        # TODO unify then, when ASTSplitter updated
        if self.task == "method_naming":
            try:
                input_data, _, method_location = Encoder.splitter.remove_comments(input_data, data["method_location"])
            except:
                return [], []
            label_tokenized = Encoder.tokenizer.tokenize(
                label.replace("|", " "),
                padding="max_length",
                truncation=True,
                max_length=self.args.max_label_length + 2,  # Accounts for BOS and EOS tokens
                )
        else:
            label_tokenized = None
        try:
            # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)

            tokenized_input = Encoder.tokenizer.tokenize(
                input_data,
                add_special_tokens=False,
                padding="do_not_pad",
                max_length=self.args.max_context_length,
                truncation=True,
            )

            tokens = Encoder.tokenizer.batch_decode(tokenized_input, skip_special_tokens=True)
            splits = Encoder.process_input(input_data, tokens, self.args.max_sent_length)
            
            num_splits = len(splits)
            # returns tensor(n_sentence, sent_len) - splitted seq. Each sent wrapped by BOS and EOS token
            split_res = transform_sequence_according_to_split_with_begin_end_tokens(
                torch.tensor(tokenized_input),  # list(token_ids)
                splits,  # list(sentence_lengths)
                num_splits,  # num_sentences
                self.args.max_sent_length,
                bos_id,
                eos_id,
            )
            if sum(splits) != len(tokens):
                with open('len_diff.txt', 'a') as f:
                    f.write(str(sum(splits)-len(tokens)))
                    f.write('\n')
                # with open('len_diff_rel.txt', 'a') as f:
                #     f.write(str(sum(splits)/len(tokens) -1))
                #     f.write('\n')
                #return [], []
            if self.args.max_sent_num < len(splits):
                raise ValueError("Example is too long for such context length")
        except:
            return [], []

        return split_res, label_tokenized

    def encode(self, json_line):
        ids = {}
        lens = {}
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        ids["code"], ids["label"] = self.split_and_tokenize(json_line)
        if len(ids["label"]) > 0:
            lens["label"] = [len(ids["label"])]
            lens["code"] = [len(sentence) for sentence in ids["code"]]
            len_line = len(json_line)
        else:
            lens["label"] = [0]
            lens["code"] = [0]
            len_line = 0

        return ids, lens, len_line

    def encode_text(self, json_line):
        ids, lens = {}, {}
        len_line = 0
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        ids["text"], _ = self.split_and_tokenize(json_line)
        if len(ids["text"]) > 0:
            lens["text"] = [len(sentence) for sentence in ids["text"]]
            len_line = len(json_line)

        return ids, lens, len_line

class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.task = self.args.task
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {count} documents", f"({count/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, "r", encoding="utf-8")
        fout = open(output_file_name, "w")

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)
        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def save_data_size(self, output_json_path, prefix, total_docs_processed):
        line_count_dict = {prefix: total_docs_processed}
        with open(output_json_path, "a") as json_file:
            json.dump(line_count_dict, json_file)
            json_file.write("\n")

    def process_json_file(self, file_name, output_prefix, processed_folder, output_json_path, separate_split_files):
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        input_file_name, output_prefix = file_name, output_prefix[:-6]
        _, output_name = os.path.split(output_prefix)
        output_prefix = os.path.join(processed_folder, output_name)
        print("Opening", input_file_name)
        fin = open(input_file_name, "r", encoding="utf-8")

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        if self.task == "method_naming":
            encoded_docs = pool.imap(encoder.encode, fin, 32)
        elif self.task == "language_modeling":
            encoded_docs = pool.imap(encoder.encode_text, fin, 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            # if "train" in output_prefix:
            #     output_prefix = "train"
            # elif "val" in output_prefix:
            #     output_prefix = "val"
            # elif "test" in output_prefix:
            #     output_prefix = "test"

            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix, key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix, key, level)
            builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        total_docs_processed = 0
        total_errors = 0
        print("Time to startup:", startup_end - startup_start)
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            if bytes_processed == 0:
                total_errors += 1
                continue
            total_docs_processed += 1
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_doc(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
        print(f'Number of tree errors = {total_errors}')
        if not separate_split_files:
            output_name = 'train'
        print(f'Final number of docs in {output_name} = {total_docs_processed}')
        self.save_data_size(output_json_path, output_name, total_docs_processed)
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--input", nargs="+", type=str, required=True, help="Path to input JSON")
    group.add_argument("--separate-split-files", action="store_true", help="If train/val/test splits are in separate files")
    group.add_argument("--processed-folder", type=str, default="", help="Path to tokenized data")
    group.add_argument("--task", type=str, choices=["method_naming", "language_modeling"])
    group.add_argument("--dataset-size-file", type=str, default="dataset_size.json", help="Path to tokenized data")
    group.add_argument(
        "--json-keys", nargs="+", help="space separate listed of keys to extract from json"
    )
    group.add_argument("--split-sentences", action="store_true", help="Split documents into sentences.")
    group.add_argument("--keep-newlines", action="store_true", help="Keep newlines between sentences when splitting.")

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "BertWordPieceLowerCase",
            "BertWordPieceCase",
            "GPT2BPETokenizer",
            "SentencePieceTokenizer",
            "GPTSentencePieceTokenizer",
            "NullTokenizer",
            "HFTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument("--tokenizer-model", type=str, default=None, help="YTTM tokenizer model.")
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file")
    group.add_argument("--vocab-size", default=786, help="size of vocab for use with NullTokenizer")
    group.add_argument("--merge-file", type=str, default=None, help="Path to the BPE merge file (if necessary).")
    group.add_argument("--append-eod", action="store_true", help="Append an <eod> token to the end of a document.")
    group.add_argument(
        "--lang", type=str, default="english", help="Language to use for NLTK-powered sentence splitting."
    )
    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to binary output file without suffix")

    group.add_argument("--language", type=str, default=None, help="Code dataset language.")
    group.add_argument("--tree-sitter-path", type=str, default=None, help="Path to the tree sitter vocab.")
    group.add_argument("--max-sent-num", type=int, default=None, help="Max number of trees in doc (method).")
    group.add_argument("--max-sent-length", type=int, default=None, help="Max len of subsequence tree.")
    group.add_argument("--max-label-length", type=int, default=None, help="Max len of label (method name).")
    group.add_argument("--max-context-length", type=int, default=None, help="Max size of the method.")

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers",
        type=int,
        required=True,
        help=(
            "Number of worker processes to launch."
            "A good default for fast pre-processing "
            "is: (workers * partitions) = available CPU cores."
        ),
    )
    group.add_argument("--partitions", type=int, default=1, help="Number of file partitions")
    group.add_argument("--log-interval", type=int, default=1000, help="Interval between progress updates")
    group.add_argument(
        "--keep-sequential-samples",
        action="store_true",
        help="Ensure ordering of samples in .jsonl files is " "preserved when using partitions>1.",
    )
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith("bert") and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)

    if args.json_keys is None:
        if args.task == "method_naming":
            args.json_keys = ["code", "label"]
        elif args.task == "language_modeling":
            args.json_keys = ["text"]

    os.makedirs(args.processed_folder, exist_ok=True)

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {"partition": input_file_name, "sentence_split": sentence_split_file, "output_prefix": output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True

def combine_jsonl_dicts(output_json_path):

    combined_data = {}
    with open(output_json_path, "r") as input_file:
        for line in input_file:
            data = json.loads(line)
            combined_data.update(data)
    with open(output_json_path, "w") as output_file:
        output_file.write(json.dumps(combined_data))

def main():
    args = get_args()
    processed_folder = args.processed_folder
    dataset_size_file = args.dataset_size_file
    in_ss_out_names = []
    if args.separate_split_files:
        for file in args.input:
            file_names = {
                "partition": file,
                "output_prefix": args.output_prefix,
            }
            in_ss_out_names.append(file_names)
    elif args.partitions == 1:  # Number of files in partition
        args.input = args.input[0]
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            "partition": args.input,
            "sentence_split": sentence_split_file,
            "output_prefix": args.output_prefix,
        }
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)

        # Count total number of lines across .jsonl files
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += fc + 1
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(in_ss_out_names, "partition", args.partitions)
        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, "sentence_split", args.partitions)

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]["partition"], "w")
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples:
                line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, "rt")
                else:
                    fin = open(in_file_name, "r", encoding="utf-8")

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1) % args.partitions

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers // args.partitions)

    # pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
    # encode partition files in parallel
    processes = []
    # input_key = "sentence_split" if args.split_sentences else "partition"
    output_json_path = os.path.join(processed_folder, dataset_size_file)
    if os.path.exists(output_json_path):
        os.remove(output_json_path)

    # pydevd_pycharm.settrace("localhost", port=PORT_DEBUG, stdoutToServer=True, stderrToServer=True)
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file, args=((name["partition"], name["partition"], processed_folder, output_json_path, args.separate_split_files)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    combine_jsonl_dicts(output_json_path)
    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = build_tokenizer(args)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix, key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix, key, level)
        builders[key] = indexed_dataset.MMapIndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name["output_prefix"]
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix, key, level)
            builders[key].merge_file_(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])

if __name__ == "__main__":

    main()
