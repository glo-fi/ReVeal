import os
import csv
import tqdm
import json
import shutil

def read_csv_orig(csv_file_path, split='\t'):
    """
    Reads a tab-delimited CSV file from the specified file path and returns a list of dictionaries.
    Each dictionary in the returned list corresponds to one row of data. The first line of the file
    is treated as the header.
    
    Args:
        csv_file_path (str): The path to the CSV (tab-delimited) file.
        
    Returns:
        list: A list of dictionaries where each dictionary represents a row of data.
    """
    # Initialise an empty list to store the results
    data = []
    # Open the file in read mode
    with open(csv_file_path, 'r') as file_pointer:
        # Read the header line, strip whitespace, split by tabs, and strip each header name
        headers = [column.strip() for column in file_pointer.readline().strip().split(split)]
        # Iterate over each subsequent line in the file
        for line in file_pointer:
            # Strip whitespace and split the line by tabs
            row_values = [value.strip() for value in line.strip().split(split)]
            # Create a dictionary mapping each header to the corresponding value
            # If a row lacks a value for a header, use an empty string
            instance = {}
            for i, header in enumerate(headers):
                # If the row provides enough columns, use it; otherwise, default to an empty string
                instance[header] = row_values[i] if i < len(row_values) else ''
            # Add the dictionary for this line to our list of results
            data.append(instance)
    
    # Return the final list containing data for every row
    return data

def read_csv(csv_file_path, delimiter=','):
    """
    Reads a CSV file using Python's standard csv library, ensuring that column headers are stripped
    of leading/trailing whitespace. Each row is returned as a dictionary keyed by those stripped headers.
    
    By default, assumes a tab-delimited file (delimiter='\\t').

    Args:
        csv_file_path (str): The path to the CSV file.
        delimiter (str): The field delimiter (defaults to '\\t').

    Returns:
        list of dict: A list of dictionaries, each representing one row, keyed by stripped column header.
    """
    data = []
    with open(csv_file_path, mode='r', newline='') as file_pointer:
        # Read the *first line only* to extract/strip the headers manually
        first_line = file_pointer.readline()
        # Use csv.reader to parse that single line
        raw_headers = next(csv.reader([first_line], delimiter=delimiter))
        # Strip whitespace from each header
        stripped_headers = [col.strip() for col in raw_headers]

        # Now, create a DictReader for the *rest of the file*,
        # supplying our own fieldnames (the stripped headers)
        reader = csv.DictReader(file_pointer, fieldnames=stripped_headers, delimiter=delimiter)
        
        # DictReader will treat each subsequent line as data (not headers),
        # mapping them to the stripped fieldnames.
        for row in reader:
            data.append(row)

    return data


def read_code_file(file_path):
    """
    Reads a file line by line, removes any content following the '//' comment delimiter, 
    and stores the resulting lines in a dictionary keyed by line number.

    Args:
        file_path (str): The path to the code file to read.

    Returns:
        dict: A dictionary where keys are line numbers (starting from 1) and 
        values are the corresponding lines of code with comment sections stripped out.
    """
    # Initialise a dictionary to store code lines, keyed by line number
    code_lines = {}
    # Open the file in read mode, giving the file handler a descriptive name
    with open(file_path, 'r') as file_pointer:
        # Enumerate each line, starting from 1 to match line numbering in the dictionary
        for ln, line in enumerate(file_pointer, start=1):
            # Strip any leading or trailing whitespace, then remove text following '//'
            processed_line = line.strip().split('//', 1)[0]
            # Store the processed line keyed by its line number
            code_lines[ln] = processed_line
    
    # Return the dictionary containing all code lines without inline comments
    return code_lines

def read_file(path):
    """
    Reads the entire contents of a file from the given path and returns it as a single string,
    with lines separated by spaces instead of newline characters.

    Args:
        path (str): The path to the file to be read.
    
    Returns:
        str: The contents of the file, with each line joined by a space.
    """
    # Open the file in read mode
    with open(path) as f:
        # Read all lines into a list
        lines = f.readlines()
        # Return the contents by joining each line with a space
        return ' '.join(lines) # ??? Is this right?


def extract_line_number(idx, nodes, loc_key="location"):
    """
    Extracts a line number from a given list of node objects by searching backward from a specified index.
    
    The function looks for a 'location' key in nodes[idx] and attempts to parse out the line number 
    (the portion before the first colon). If no valid line number is found while searching backwards
    through the nodes, the function returns -1.

    Args:
        idx (int): The starting index from which to search backwards.
        nodes (list): A list of node dictionaries, where each node may contain a 'location' key.

    Returns:
        int: The extracted line number if found, otherwise -1.
    """
    # Search backwards from 'idx' to 0
    while idx >= 0:
        # Retrieve the current node
        c_node = nodes[idx]
        # Check if 'location' exists in the node
        if loc_key in c_node.keys():
            location = c_node[loc_key]
            # If location is not empty, try to parse it
            if location.strip() != '':
                try:
                    # Extract the line number, which is the part before the first colon
                    ln = int(location.split(':')[0])
                    return ln
                except:
                    # If parsing fails, ignore the error and continue searching
                    pass
        # Move one step backwards
        idx -= 1
    
    # If no valid location was found, return -1
    return -1

def checkVul(cFile):
    with open(cFile, 'r') as f:
        fileString = f.read()
        return (1 if "BUFWRITE_COND_UNSAFE" in fileString or "BUFWRITE_TAUT_UNSAFE" in fileString else 0)

def rename_sysevr_nvd_files(input_directory, output_directory):
    for idx, filename in enumerate(os.listdir(input_directory)):
        full_path = os.path.join(input_directory, filename)
        if os.path.isfile(full_path):
            if "VULN" in filename:
                output_file = os.path.join(output_directory, f"{idx}_1.c")
            elif "PATCHED" in filename:
                output_file = os.path.join(output_directory, f"{idx}_0.c")
            shutil.copy(full_path, output_file)

def files_to_list(directory):
    output_list = []
    for filename in os.listdir(directory):
        output_list.append(os.path.splitext(filename)[0])
    return output_list

def process_fq(filepath, output_directory):
    with open(filepath, "r") as f:
        func_list = json.load(f)
        for idx, func in enumerate(tqdm.tqdm(func_list)):
            func_body = func["func"]
            func_label = func["target"]
            output_file = os.path.join(output_directory, f"{idx}_{func_label}.c")
            with open(output_file, "w+") as of:
                of.write(func_body) 



if __name__ == "__main__":
    process_fq("/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/raw/ffmpeg_qemu/function.json",
               "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/fq")
    #rename_sysevr_nvd_files("/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/raw_code",
    #                        "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/renamed_code")