from utils import files_to_list
from dataclasses import dataclass

@dataclass
class FileInfo:
    file_source: str
    split_dir: str
    parsed: str
    json_file_path: str
    w2v_path: str
    shard_directory: str
    files: list

    def __init__(self, file_source, split_dir, parsed,
                json_file_path, w2v_path, shard_directory):
        self.file_source = file_source
        self.split_dir = split_dir
        self.parsed = parsed
        self.json_file_path = json_file_path
        self.w2v_path = w2v_path
        self.shard_directory = shard_directory

        self.files = files_to_list(self.file_source)
    

sysevr_info = FileInfo(file_source="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/renamed_code",
                       split_dir="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/renamed_code/",
                       parsed="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/parsed_sysevr/renamed_code/",
                       json_file_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/sysevr/process_slices.json",
                       w2v_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/saved_models/li_et_al_wv",
                       shard_directory="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/sysevr/shards/")

fq_info = FileInfo(file_source="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq",
                   split_dir="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq/",
                   parsed="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/parsed_fq/",
                   json_file_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/fq_json/process_slices.json",
                   w2v_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/saved_models/fq_wv",
                   shard_directory="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/fq_json/shards/")

new_fq_info = FileInfo(file_source="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq_new",
                   split_dir="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq_new/",
                   parsed="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/parsed_fq_new/",
                   json_file_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/new_fq_json/process_slices.json",
                   w2v_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/saved_models/fq_wv",
                   shard_directory="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/new_fq_json/shards/")


old_fq_test = FileInfo(file_source="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq_test",
                   split_dir="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq_test/",
                   parsed="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/parsed_fq_test/",
                   json_file_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/fq_json/process_slices.json",
                   w2v_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/saved_models/fq_wv",
                   shard_directory="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/fq_json/shards/")

new_fq_test = FileInfo(file_source="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq_new_test",
                   split_dir="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code_fq_new_test/",
                   parsed="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/parsed_fq_new_test/",
                   json_file_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/new_fq_json/process_slices.json",
                   w2v_path="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/saved_models/fq_wv",
                   shard_directory="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/new_fq_json/shards/")