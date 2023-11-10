from checkpoint_reshaping_and_interoperability import main
import sys
import os

# load_path = '/home/galimzyanov/data/Megatron/checkpoints_dev/iter_0013281'
# save_path = '/home/galimzyanov/data/Megatron/checkpoints_HF'

load_path = '../checkpoints/iter_0031924'
save_path = '../checkpoints/HF_checkpoint'
os.makedirs(save_path, exist_ok=True)

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
main()
