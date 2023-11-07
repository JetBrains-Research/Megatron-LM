from checkpoint_reshaping_and_interoperability import main
import sys
import argparse

# load_path = '/home/galimzyanov/data/Megatron/checkpoints_dev/iter_0013281'
# save_path = '/home/galimzyanov/data/Megatron/checkpoints_HF'

load_path = '../checkpoints/iter_0013281'
save_path = '../checkpoints/HF_check'


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument("--convert_checkpoint_from_megatron_to_transformers", action="store_true")
#     parser.add_argument("--load_path", type=str, default=load_path)
#     parser.add_argument("--save_path", type=str, default=save_path)
#     # parser.add_argument("--max_shard_size", type=str, default="200MB")
#     parser.add_argument("--tokenizer_name", type=str, default="Salesforce/codet5p-220m")
#     parser.add_argument("--print-checkpoint-structure", action="store_true")
#
#     args = parser.parse_args()
#     checkpoint_reshaping_and_interoperability.main(args)


sys.argv = [
    "checkpoint_reshaping_and_interoperability.py",
    "--convert_checkpoint_from_megatron_to_transformers",
    "--load_path", load_path,
    "--save_path", save_path,
    "--tokenizer_name", "Salesforce/codet5p-220m",
    "--print-checkpoint-structure",
]

# import pydevd_pycharm
# pydevd_pycharm.settrace("localhost", port=2000, stdoutToServer=True, stderrToServer=True)
# Call the main function with the arguments
main()

# args = {
#     "convert_checkpoint_from_megatron_to_transformers": True,
#     "load_path": load_path,
#     "save_path": save_path,
# #    "max_shard_size": "200MB",
#     "tokenizer_name": "Salesforce/codet5p-220m",
#     "print-checkpoint-structure": True,  # Note: The argument name contains a hyphen, which may need to be adjusted.
# }


