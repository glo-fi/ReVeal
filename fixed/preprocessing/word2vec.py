import os
import json
from utils import files_to_list
from gensim.models import Word2Vec
from tokenizer import symbolic_tokenize

# Constants for file processing
UNFORMATTED_SUBDIR = 'raw_code'
FORMATTED_SUBDIR = 'json_code_fq'
MARKER = '-------------------------' 
MIN_EXAMPLE_LINES = 3


def w2v(data_paths, 
        save_model_dir, 
        model_name='fq_wv', 
        min_occ=1, 
        embedding_size=64, 
        epochs=5):
    """
    Train a Word2Vec model on tokenized code from JSON files.

    Parameters:
        data_paths (list): List of file paths to JSON data.
        save_model_dir (str): Directory to save the Word2Vec model.
        model_name (str): Name of the saved model file.
        min_occ (int): Minimum number of occurrences for a token to be included.
        embedding_size (int): Dimensionality of the word vectors.
        epochs (int): How many times to iterate over the sentences during training.
    """

    # List to collect tokenized sentences from all files.
    sentences = []
    for file_path in os.listdir(data_paths):
        try:
            with open(os.path.join(data_paths, file_path), 'r') as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # Process each example in the JSON data.
        for example in data:
            # Get the tokenized code. Expecting a string of tokens separated by whitespace.
            tokenized_code = example.get('tokenized', '')
            # Strip and split tokens, filtering out any empty tokens.
            tokens = [token.strip() for token in tokenized_code.split() if token.strip()]
            if tokens:
                sentences.append(tokens)

    print(f"Total number of sentences: {len(sentences)}")

    # Initialize the Word2Vec model with the given parameters.
    word2vec_model = Word2Vec(
        vector_size=embedding_size,
        min_count=min_occ,
        workers=8,
        epochs=epochs  # Setting the number of epochs within the model
    )

    word2vec_model.build_vocab(sentences)
    print("Vocabulary built.")

    word2vec_model.train(sentences, total_examples=len(sentences), epochs=epochs)

    print(f"Embedding Size: {word2vec_model.vector_size}")

    # Alternatively, if you want to iterate epochs manually:
    # for _ in range(epochs):
    #     word2vec_model.train(sentences, total_examples=len(sentences), epochs=1)

    # Ensure the saving directory exists
    os.makedirs(save_model_dir, exist_ok=True)
    save_file_path = os.path.join(save_model_dir, model_name)
    
    # Save the model
    word2vec_model.save(save_file_path)
    print(f"Model saved to {save_file_path}")

def w2v_process_file(file_identifier, directory):
    """
    Processes a single file containing code examples.
    
    Arguments:
        file_identifier (str): Unique identifier for the file.
        directory (str): The base directory where files are stored.
    
    Returns:
        tuple: (all_examples, vulnerable_count, non_vulnerable_count)
    """
    # Build the input file path
    file_path = os.path.join(directory, UNFORMATTED_SUBDIR, f"{file_identifier}.c")
    
    all_examples = []
    vulnerable_count = 0
    non_vulnerable_count = 0

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # read and strip non-empty lines
    current_example = []
    
    # Function to process the current example
    def process_current_example(example_lines):
        nonlocal vulnerable_count, non_vulnerable_count, all_examples
        if len(example_lines) < MIN_EXAMPLE_LINES:
            return  # Not enough lines to process
        
        # Code is assumed to be between the first and last lines of example_lines
        code_snippet = "\n".join(example_lines[1:-1])
        # The last line should be the label
        try:
            label = int(file_identifier.split("_")[-1])
        except ValueError:
            # If label conversion fails, skip example and could log or raise a warning if needed.
            return
        
        # Update counters based on label
        if label == 0:
            non_vulnerable_count += 1
        else:
            vulnerable_count += 1
        
        # Process the code snippet (symbolic_tokenize should be defined elsewhere)
        tokenized = symbolic_tokenize(code_snippet)
        
        example_data = {
            'code': code_snippet,
            'label': label,
            'tokenized': tokenized
        }
        all_examples.append(example_data)
    
    # Process each line and build code examples
    for line in lines:
        current_example.append(line)
    
    # Optionally, process any remaining example that isn't terminated by a marker.
    if current_example:
        process_current_example(current_example)
        current_example = []  # Reset for the next example
    return all_examples, vulnerable_count, non_vulnerable_count

def w2v_process_all_files(vul_files, directory):
    """
    Processes multiple files and writes the processed examples to JSON files.
    
    Arguments:
        vul_files (iterable): List of file identifiers.
        directory (str): Base directory where files are stored.
    """
    for file_identifier in vul_files:
        examples, vulnerable, non_vulnerable = w2v_process_file(file_identifier, directory)
        
        # Build the output file path
        output_file_path = os.path.join(os.path.join(directory, FORMATTED_SUBDIR), f"{file_identifier}-processed.json")
        
        # Write the processed examples to the JSON file using a context manager
        with open(output_file_path, 'w') as out_file:
            json.dump(examples, out_file, indent=4)
    
        print(f"File: {file_identifier} - Vulnerable: {vulnerable}, Non-vulnerable: {non_vulnerable}")
    return output_file_path

# Example usage

if __name__ == "__main__":
    directory = '/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/'
    vul_files = files_to_list(os.path.join(directory, UNFORMATTED_SUBDIR))  # List of file identifiers.
    w2v_process_all_files(vul_files, directory)
    w2v(os.path.join(directory, FORMATTED_SUBDIR), "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/saved_models/")
