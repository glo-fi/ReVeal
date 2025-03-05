import os
import csv
import nltk
import json
import tqdm
import utils
import clang
import l_funcs
import numpy as np
from graphviz import Digraph
from gensim.models import Word2Vec
from tokenizer import tokenize, symbolic_tokenize


##################################
######### Helper Methods #########
##################################

def set_clang_config():
    try:
        clang.cindex.Config.set_library_path("/usr/lib/x86_64-linux-gnu")
    except Exception as e:
        print(f"[!] Clang library path not found: {e}")
    try:
        clang.cindex.Config.set_library_file('''/usr/lib/
                                             x86_64-linux-gnu/libclang-6.0.so.1''')
    except Exception as e:
        print(f"[!] Clang library file not found: {e}")

def extract_nodes_with_location_info(nodes):
    """
    Extracts location information for each node in the provided list of nodes,
    returning indices, node IDs, line numbers, and a mapping from node ID to 
    its associated line number.

    Args:
        nodes (list): A list of node objects (dictionaries) that may contain 'location' 
                      and 'key' attributes.

    Returns:
        tuple:
            - node_indices (list): Indices of the nodes within the 'nodes' list 
              that have valid location information.
            - node_ids (list): The corresponding node IDs (from the 'key' field).
            - line_numbers (list): The line number (extracted from each node's 
              'location' field).
            - node_id_to_line_number (dict): A dictionary mapping node ID to its 
              line number.
    """
    # Will return an array identifying the indices of those nodes in the nodes array,
    # another array identifying the node_id of those nodes,
    # and another array indicating the line numbers.
    # All 3 return arrays should have the same length, indicating 1-to-1 matching.
    
    node_indices = []
    node_ids = []
    line_numbers = []
    node_id_to_line_number = {}

    # Loop through each node, enumerating so we can keep track of the index
    for node_index, node in enumerate(nodes):
        # Validate that the node is a dictionary
        assert isinstance(node, dict)
        # Check if node contains the 'location' key
        if 'location' in node.keys():
            location = node['location']
            # If the location field is an empty string, skip this node
            if location == '':
                continue
            # Extract the line number from the location, 
            # which is the part before ':'
            line_num = int(location.split(':')[0])
            # Retrieve and strip the node ID from the 'key' field
            node_id = node['key'].strip()
            # Append the data we extracted to the respective lists
            node_indices.append(node_index)
            node_ids.append(node_id)
            line_numbers.append(line_num)
            # Record the mapping from node ID to line number
            node_id_to_line_number[node_id] = line_num
    
    # Return the four items in a tuple
    return node_indices, node_ids, line_numbers, node_id_to_line_number


def create_adjacency_list(line_numbers, 
                          node_id_to_line_numbers, 
                          edges, 
                          data_dependency_only=False):
    """
    Creates an adjacency list for control flow and data flow based on the provided edges.
    
    The adjacency list maps each line number to a list of two sets:
    - The first set (index 0) contains line numbers reachable via control flow edges (labelled 'CONTROLS'). 
    - The second set (index 1) contains line numbers reachable via data flow edges (labelled 'REACHES').
    
    Args:
        line_numbers (list): A list of line numbers to be included in the adjacency list.
        node_id_to_line_numbers (dict): A dictionary mapping node IDs to their corresponding line number.
        edges (list): A list of edges, where each edge is a dictionary with 'type', 'start', and 'end' keys.
        data_dependency_only (bool, optional): If True, ignore control flow edges ('CONTROLS') 
            and only consider data flow edges ('REACHES'). Defaults to False.
    
    Returns:
        dict: A dictionary mapping each line number to a list of two sets. 
              The first set holds the control flow adjacency, the second set holds the data flow adjacency.
    """
    # Initialise a dictionary that will map each line number 
    # to a list of two sets: [control_flow_set, data_flow_set]
    adjacency_list = {}
    # Populate the adjacency list with every unique line number from the input,
    # each line number starting with two empty sets
    for ln in set(line_numbers):
        adjacency_list[ln] = [set(), set()]
    # Iterate through each edge in the provided list of edges
    for edge in edges:
        # Extract the edge type and trim any whitespace
        edge_type = edge['type'].strip()
        # This check currently always evaluates to True, 
        # but was originally intended to filter specific edge types
        if True:  # edge_type in ['IS_AST_PARENT', 'FLOWS_TO']:
            # Retrieve the start and end node IDs, stripping whitespace
            start_node_id = edge['start'].strip()
            end_node_id = edge['end'].strip()
            # Check if both node IDs exist in the node_id_to_line_numbers dictionary
            if start_node_id not in node_id_to_line_numbers.keys() or end_node_id not in node_id_to_line_numbers.keys():
                continue
            # Retrieve the line numbers corresponding to the start and end node IDs
            start_ln = node_id_to_line_numbers[start_node_id]
            end_ln = node_id_to_line_numbers[end_node_id]
            # If we are not restricting ourselves to data_dependency_only, 
            # check whether this is a control flow edge
            if not data_dependency_only:
                if edge_type == 'CONTROLS':  # Control Flow edges
                    adjacency_list[start_ln][0].add(end_ln)
            # If the edge is a data flow edge ('REACHES'), add adjacency accordingly
            if edge_type == 'REACHES':  # Data Flow edges
                adjacency_list[start_ln][1].add(end_ln)
    
    # Return the constructed adjacency list
    return adjacency_list


def create_visual_graph(code, adjacency_list, file_name='test_graph', verbose=False):
    graph = Digraph('Code Property Graph')
    for ln in adjacency_list:
        graph.node(str(ln), str(ln) + '\t' + code[ln], shape='box')
        control_dependency, data_dependency = adjacency_list[ln]
        for anode in control_dependency:
            graph.edge(str(ln), str(anode), color='red')
        for anode in data_dependency:
            graph.edge(str(ln), str(anode), color='blue')
    graph.render(file_name, view=verbose)


def create_forward_slice(adjacency_list, line_no):
    """
    Generates a forward slice starting from a given line number in the adjacency list.
    This forward slice includes the specified line and all lines reachable from it.
    
    Args:
        adjacency_list (dict): A dictionary mapping each line to a collection of lines 
            that can be visited directly from it. 
        line_no (int): The starting line number for the forward slice.
        
    Returns:
        list: A sorted list of all line numbers that are reachable from the starting line,
              inclusive of the starting line itself.
    """
    # Use a set to keep track of sliced lines to avoid duplicates
    sliced_lines = set()
    # Add the initial line number to the set
    sliced_lines.add(line_no)
    # Initialise a stack (list) with our starting line
    stack = list()
    stack.append(line_no)
    # While the stack is not empty, keep processing
    while len(stack) != 0:
        # Pop the top element from the stack
        cur = stack.pop()
        # If this line has not yet been recorded, add it
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        # Retrieve the adjacent lines from the adjacency list
        adjacents = adjacency_list[cur]
        # For each adjacent line, if it's not already in the sliced set, push it onto the stack
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    
    # Sort the sliced lines before returning to produce a stable order
    sliced_lines = sorted(sliced_lines)
    return sliced_lines



def combine_control_and_data_adjacents(adjacency_list):
    """
    Combines the control-flow adjacents and data-flow adjacents for each line number in the adjacency list.
    
    The input 'adjacency_list' is assumed to map each line number (key) to a list of two sets:
    [control_flow_set, data_flow_set]. This function merges both sets for each line number into a single set.

    Args:
        adjacency_list (dict): A dictionary mapping line numbers to a list of two sets:
                               index 0 is the control-flow adjacency set,
                               index 1 is the data-flow adjacency set.

    Returns:
        dict: A new dictionary where each line number is mapped to a single combined set of adjacent lines.
    """
    # Create a new dictionary to store the combined adjacency
    cgraph = {}
    # Iterate over each line number key in the adjacency list
    for ln in adjacency_list:
        # Initialise an empty set for this line number
        cgraph[ln] = set()
        # Union the control-flow set (index 0) into the combined set
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][0])
        # Union the data-flow set (index 1) into the combined set
        cgraph[ln] = cgraph[ln].union(adjacency_list[ln][1])
    
    # Return the dictionary with merged adjacents
    return cgraph


def invert_graph(adjacency_list):
    """
    Inverts the direction of edges in the given adjacency list.

    The function takes a dictionary (adjacency_list) where each key (line number) 
    maps to a set of line numbers reachable from it. It returns a new dictionary (igraph) 
    representing the inverse graph: every edge is reversed. Specifically, if A -> B in the 
    original graph, then B -> A in the inverted graph.

    Args:
        adjacency_list (dict): A dictionary mapping each line (int) to a set of lines (int)
            that can be reached from it.

    Returns:
        dict: A dictionary (inverted graph) mapping each line to a set of lines that 
        can now reach it in the reversed direction.
    """
    # Initialise an empty dictionary for the inverted graph
    igraph = {}
    # Ensure that every line number is present as a key in the new dictionary,
    # each mapped to an empty set initially
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    # For each line number in the original adjacency list,
    # add a reverse edge to the new graph
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        # For every node that 'ln' can reach, add 'ln' as reachable from 'node' in the inverted graph
        for node in adj:
            igraph[node].add(ln)
    # Return the inverted graph
    return igraph


def create_backward_slice(adjacency_list, line_no):
    """
    Constructs a backward slice starting from a specified line number by 
    inverting the adjacency list and then performing a forward slice on it.

    A backward slice finds all lines that can reach the specified line number.

    Args:
        adjacency_list (dict): A dictionary mapping each line (int) to a set of lines (int)
            that can be reached from it.
        line_no (int): The line number from which the backward slice is computed.

    Returns:
        list: A list of line numbers forming the backward slice, extracted by 
        creating an inverted graph and then using create_forward_slice on that graph.
    """
    # Invert the adjacency list to reverse the direction of edges
    inverted_adjacency_list = invert_graph(adjacency_list)
    # Perform a forward slice on the inverted graph to get all lines that can
    # eventually lead to the specified line_no in the original graph
    return create_forward_slice(inverted_adjacency_list, line_no)

def extract_slices(code_with_lines, list_of_slices):
    sliced_codes = []
    for slice in list_of_slices:
        tokenized = []
        for ln in slice:
            code = code_with_lines[ln]
            tokenized.append(symbolic_tokenize(code))
        sliced_codes.append(' '.join(tokenized))
    return sliced_codes

def unify_slices(list_of_list_of_slices):
    taken_slice = set()
    unique_slice_lines = []
    for list_of_slices in list_of_list_of_slices:
        for slice in list_of_slices:
            slice_id = str(slice)
            if slice_id not in taken_slice:
                unique_slice_lines.append(slice)
                taken_slice.add(slice_id)
    return unique_slice_lines

def reformat_code_line_graph(code_lines, adjacency_lists, wv_model_li, label):
    actual_lines = []
    for ln in adjacency_lists.keys():
        cd, dd = adjacency_lists[ln]
        new_cd = [l for l in cd]
        new_dd = [l for l in dd] 
        actual_lines.extend(new_cd)
        actual_lines.extend(new_dd)
        actual_lines.append(ln)
    actual_lines = sorted(list(set(actual_lines)))
    line_no_to_idx = {}
    idx_to_line_no = {}
    for idx, ln in enumerate(actual_lines):
        line_no_to_idx[ln] = idx
        idx_to_line_no[idx] = ln
    data_point = {}
    graph = []
    for src in adjacency_lists.keys():
        cd, dd = adjacency_lists[src]
        for dest in cd:
            graph.append([line_no_to_idx[src], 0, line_no_to_idx[dest]])
            graph.append([line_no_to_idx[dest], 1, line_no_to_idx[src]])
        for dest in dd:
            graph.append([line_no_to_idx[src], 2, line_no_to_idx[dest]])
            graph.append([line_no_to_idx[dest], 3, line_no_to_idx[src]])
    original_tokens = []
    symbolic_tokens = []
    line_features_wv = []
    sym_line_features_wv = []
    
    for lidx in range(len(idx_to_line_no.keys())):
        actual_code_line = code_lines[idx_to_line_no[lidx]]
        actual_line_tokens = nltk.wordpunct_tokenize(actual_code_line)
        symbolic_line_tokens = symbolic_tokenize(actual_code_line).split()
        original_tokens.append(actual_line_tokens)
        symbolic_tokens.append(symbolic_line_tokens)
        
        #############################################
        # The commented-out code below is part of the 
        # original code. However, it is unclear how
        # it is being used. It produces an output
        # more or less identical to the other loop,
        # but with a slightly different tokeniser and
        # size.
        #############################################

        #nrp = np.zeros(100)
        #for token in actual_line_tokens:
        #    try:
        #        embedding = wv_model_original.wv[token]
        #    except:
        #        embedding = np.zeros(100)
        #    nrp = np.add(nrp, embedding)
        #if len(actual_line_tokens) > 0:
        #    fNrp = np.divide(nrp, len(actual_line_tokens))
        #else:
        #    fNrp = nrp
        #line_features_wv.append(fNrp.tolist())
        
        nrp = np.zeros(64)
        for token in symbolic_line_tokens:
            try:
                embedding = wv_model_li.wv[token]
            except:
                embedding = np.zeros(64)
            nrp = np.add(nrp, embedding)
        if len(actual_line_tokens) > 0:
            fNrp = np.divide(nrp, len(symbolic_line_tokens))
        else:
            fNrp = nrp
        sym_line_features_wv.append(fNrp.tolist())
    data_point = {
        #'node_features': line_features_wv,
        'node_features_sym': sym_line_features_wv,
        'graph': graph,
        'original_tokens': original_tokens,
        'symbolic_tokens': symbolic_tokens,
        'targets': [[label]]
    }
    return data_point

############################################
############### Main Methods ###############
############################################


def process_slices(files, split_dir, parsed):
    # Note: This code snippet iterates over a list of file names. For each file, 
# it reads the corresponding code text, parses node and edge information, 
# extracts various line sets (call, array, pointer, arithmetic), creates 
# forward/backward slices, tokenises the code, and then stores the results 
# in a data structure for further analysis.
    all_data = []

    for i, file_name in enumerate(files):
        """
        Main loop processing each file in the 'files' list. 
        'file_name' refers to a single source file or code split to be analysed.

        Args:
            i (int): The index of the current file in the enumeration.
            file_name (str): The name (or partial path) of a file to process.

        The code within this loop performs the following steps:
        1. Determine the label from the file name (derived from the substring after the last underscore).
        2. Read the code contents from the specified 'split_dir' directory.
        3. Construct paths to the corresponding parsed nodes and edges files.
        4. Read the nodes using csv.DictReader and store them in 'nodes'.
        5. Initialise sets to track lines of interest ('call_lines', 'array_lines', etc.).
        6. Iterate over 'nodes' to find lines matching specific node types or operators.
        7. Read the same nodes and edges again with 'read_csv' to create adjacency and combined graphs.
        8. Perform forward and backward slices for each line of interest.
        9. Tokenise the full code text.
        10. Construct a 'data_instance' dictionary with the tokens, slices, and label, 
            appending it to 'all_data'.
        11. Print progress every 1000 iterations.
        """

        # Extract the label integer from the file name based on the naming convention (last underscore)
        label = file_name.strip().split('_')[-1]

        # Read the raw code text from the split directory
        code_text = utils.read_file(split_dir + file_name.strip() + ".c")

        # Construct file paths for nodes and edges CSV files
        nodes_file_path = parsed + file_name.strip() + '.c/nodes.csv'
        edges_file_path = parsed + file_name.strip() + '.c/edges.csv'

        # Open the nodes file to read data as dictionaries
        nc = open(nodes_file_path)
        nodes_file = csv.DictReader(nc, delimiter='\t')
        nodes = [node for node in nodes_file]  # Read all nodes into a list
        
        # Initialise sets to collect line numbers of interest
        call_lines = set()
        array_lines = set()
        ptr_lines = set()
        arithmatic_lines = set()
        
        # If no nodes are present, skip to the next file
        if len(nodes) == 0:
            continue
        
        # Identify specific line numbers associated with calls, arrays, pointers, and arithmetic
        for node_idx, node in enumerate(nodes):
            ntype = node['type'].strip()
            
            # Identify function calls to certain library functions
            if ntype == 'CallExpression':
                function_name = nodes[node_idx + 1]['code']
                if function_name is None or function_name.strip() == '':
                    continue
                if function_name.strip() in l_funcs.l_funcs:
                    line_no = utils.extract_line_number(node_idx, nodes)
                    if line_no > 0:
                        call_lines.add(line_no)
            # Identify array index operations
            elif ntype == 'ArrayIndexing':
                line_no = utils.extract_line_number(node_idx, nodes)
                if line_no > 0:
                    array_lines.add(line_no)
            # Identify pointer member access operations
            elif ntype == 'PtrMemberAccess':
                line_no = utils.extract_line_number(node_idx, nodes)
                if line_no > 0:
                    ptr_lines.add(line_no)
            # Identify arithmetic operations based on operator symbols
            elif node['operator'].strip() in ['+', '-', '*', '/']:
                line_no = utils.extract_line_number(node_idx, nodes)
                if line_no > 0:
                    arithmatic_lines.add(line_no)
        
        # Re-read nodes and edges using read_csv for further processing
        nodes = utils.read_csv(nodes_file_path)
        edges = utils.read_csv(edges_file_path)
        
        # Extract node indices, IDs, and location information
        node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
        
        # Create an adjacency list for combined control/data flow
        adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
        combined_graph = combine_control_and_data_adjacents(adjacency_list)
        

        # Prepare data structures for storing slices of interest
        array_slices = []
        array_slices_bdir = []
        call_slices = []
        call_slices_bdir = []
        arith_slices = []
        arith_slices_bdir = []
        ptr_slices = []
        ptr_slices_bdir = []
        all_slices = []
        
        # Track unique slices to avoid duplicates
        all_keys = set()
        _keys = set()

        # Generate forward and backward slices for call_lines 
        for slice_ln in call_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            
            # Combine both slices, remove duplicates, and sort
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            
            # Create a unique key representing this slice
            key = ' '.join([str(i) for i in all_slice_lines])
            
            # If this particular slice hasn't been seen before, store it
            if key not in _keys:
                call_slices.append(backward_sliced_lines)
                call_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            # Add to overall slices if not encountered
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
        
        # Reset _keys for array slice tracking
        _keys = set()
        for slice_ln in array_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            
            key = ' '.join([str(i) for i in all_slice_lines])
            
            if key not in _keys:
                array_slices.append(backward_sliced_lines)
                array_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
        
        # Reset _keys for arithmetic slice tracking
        _keys = set()
        for slice_ln in arithmatic_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            
            key = ' '.join([str(i) for i in all_slice_lines])
            
            if key not in _keys:
                arith_slices.append(backward_sliced_lines)
                arith_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
        
        # Reset _keys for pointer slice tracking
        _keys = set()
        for slice_ln in ptr_lines:
            forward_sliced_lines = create_forward_slice(combined_graph, slice_ln)
            backward_sliced_lines = create_backward_slice(combined_graph, slice_ln)
            
            all_slice_lines = forward_sliced_lines
            all_slice_lines.extend(backward_sliced_lines)
            all_slice_lines = sorted(list(set(all_slice_lines)))
            
            key = ' '.join([str(i) for i in all_slice_lines])
            
            if key not in _keys:
                ptr_slices.append(backward_sliced_lines)
                ptr_slices_bdir.append(all_slice_lines)
                _keys.add(key)
            if key not in all_keys:
                all_slices.append(all_slice_lines)
                all_keys.add(key)
        
        # Tokenise the entire code text
        t_code = tokenize(code_text)
        if t_code is None:
            continue

        # Construct a data instance with all curated information
        data_instance = {
            'file_path': split_dir + file_name.strip(),
            'code': code_text,
            'tokenized': t_code,
            'call_slices_vd': call_slices,
            'call_slices_sy': call_slices_bdir,
            'array_slices_vd': array_slices,
            'array_slices_sy': array_slices_bdir,
            'arith_slices_vd': arith_slices,
            'arith_slices_sy': arith_slices_bdir,
            'ptr_slices_vd': ptr_slices,
            'ptr_slices_sy': ptr_slices_bdir,
            'label': int(label)
        }
        

        # Add the data instance to the global collection
        all_data.append(data_instance)
        
        # Print a progress report every 1000 files
        if i % 100 == 0:
            print(
                i,
                len(call_slices),
                len(call_slices_bdir),
                len(array_slices),
                len(array_slices_bdir),
                len(arith_slices),
                len(arith_slices_bdir),
                sep='\t'
            )
    return all_data


def inputGeneration(nodeCSV, edgeCSV, target, wv, edge_type_map, cfg_only=False):
    gInput = dict()
    gInput["targets"] = list()
    gInput["graph"] = list()
    gInput["node_features"] = list()
    gInput["targets"].append([target])
    with open(nodeCSV, 'r') as nc:
        nodes = csv.DictReader(nc, delimiter='\t')
        nodeMap = dict()
        allNodes = {}
        node_idx = 0
        for idx, node in enumerate(nodes):
            cfgNode = node['isCFGNode'].strip()
            if not cfg_only and (cfgNode == '' or cfgNode == 'False'):
                continue
            nodeKey = node['key']
            node_type = node['type']
            if node_type == 'File':
                continue
            node_content = node['code'].strip()
            node_split = nltk.word_tokenize(node_content)
            nrp = np.zeros(64)
            for token in node_split:
                try:
                    embedding = wv.wv[token]
                except:
                    embedding = np.zeros(64)
                nrp = np.add(nrp, embedding)
            if len(node_split) > 0:
                fNrp = np.divide(nrp, len(node_split))
            else:
                fNrp = nrp
            node_feature = l_funcs.type_one_hot[l_funcs.type_map[node_type] - 1].tolist()
            node_feature.extend(fNrp.tolist())
            allNodes[nodeKey] = node_feature
            nodeMap[nodeKey] = node_idx
            node_idx += 1
        if node_idx == 0 or node_idx >= 500:
            return None
        all_nodes_with_edges = set()
        trueNodeMap = {}
        all_edges = []
        with open(edgeCSV, 'r') as ec:
            reader = csv.DictReader(ec, delimiter='\t')
            for e in reader:
                start, end, eType = e["start"], e["end"], e["type"]
                if eType != "IS_FILE_OF":
                    if not start in nodeMap or not end in nodeMap or not eType in edge_type_map:
                        continue
                    all_nodes_with_edges.add(start)
                    all_nodes_with_edges.add(end)
                    edge = [start, edge_type_map[eType], end]
                    all_edges.append(edge)
        if len(all_edges) == 0:
            return None
        for i, node in enumerate(all_nodes_with_edges):
            trueNodeMap[node] = i
            gInput["node_features"].append(allNodes[node])
        for edge in all_edges:
            start, t, end = edge
            start = trueNodeMap[start]
            end = trueNodeMap[end]
            e = [start, t, end]
            gInput["graph"].append(e)
    return gInput

def applyInputGeneration(json_file_path: str, w2v_path: str, csv_dir: str, output_dir: str):
    data = json.load(open(json_file_path))
    model = Word2Vec.load(w2v_path)
    final_data = []
    v, nv, vd_present, syse_present, cg_present, dg_present, cdg_present = 0, 0, 0, 0, 0, 0, 0
    data_shard = 1
    for didx, entry in enumerate(tqdm.tqdm(data)):
        file_name = entry['file_path'].split('/')[-1]
        nodes_path = os.path.join(csv_dir, file_name + ".c", 'nodes.csv')
        edges_path = os.path.join(csv_dir, file_name + ".c", 'edges.csv')
        label = int(entry['label'])
        if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
            continue
        linized_code = {}
        for ln, code in enumerate(entry['code'].split('\n')):
            linized_code[ln + 1] = code
        vuld_slices = extract_slices(linized_code, entry['call_slices_vd'])
        syse_slices = extract_slices(
            linized_code, unify_slices(
                [entry['call_slices_sy'], entry['array_slices_sy'], entry['arith_slices_sy'], entry['ptr_slices_sy']]
            )
        )
        graph_input_full = inputGeneration(
            nodes_path, edges_path, label, model, l_funcs.edgeType_full, False)
        graph_input_control = inputGeneration(
            nodes_path, edges_path, label, model, l_funcs.edgeType_control, True)
        graph_input_data = inputGeneration(nodes_path, edges_path, label, model, l_funcs.edgeType_data, True)
        graph_input_cd = inputGeneration(
            nodes_path, edges_path, label, model, l_funcs.edgeType_control_data, True)
        draper_code = entry['tokenized']
        if graph_input_full is None:
            continue
        if label == 1:
            v += 1
        else:
            nv += 1
        if len(vuld_slices) > 0: vd_present += 1
        if len(syse_slices) > 0: syse_present += 1
        if graph_input_control is not None: cg_present += 1
        if graph_input_data is not None: dg_present += 1
        if graph_input_cd is not None: cdg_present += 1
        data_point = {
            'id': didx,
            'file_name': file_name, 'file_path': os.path.abspath(entry['file_path']),
            'code': entry['code'],
            'vuld': vuld_slices, 'vd_present': 1 if len(vuld_slices) > 0 else 0,
            'syse': syse_slices, 'syse_present': 1 if len(syse_slices) > 0 else 0,
            'draper': draper_code,
            'full_graph': graph_input_full,
            'cgraph': graph_input_control,
            'dgraph': graph_input_data,
            'cdgraph': graph_input_cd,
            'label': int(entry['label'])
        }
        final_data.append(data_point)
        if len(final_data) == 5000:
            output_path = output_dir + '.shard' + str(data_shard)
            with open(output_path, 'w') as fp:
                json.dump(final_data, fp)
                fp.close()
            print('Saved Shard %d to %s' % (data_shard, output_path), '=' * 100, 'Done', sep='\n')
            final_data = []
            data_shard += 1
    print("Vulnerable:\t%d\n"
          "Non-Vul:\t%d\n"
          "VulDeePecker:\t%d\n"
          "SySeVr:\t%d\n"
          "Control: %d\tData: %d\tBoth: %d" % \
          (v, nv, vd_present, syse_present, cg_present, dg_present, cdg_present))
    output_path = output_dir + '.shard' + str(data_shard)
    with open(output_path, 'w') as fp:
        json.dump(final_data, fp)
        fp.close()
    print('Saved Shard %d to %s' % (data_shard, output_path), '=' * 100, 'Done', sep='\n')

def extract_line_graph_data(
                        shard_directory,
                        output_dir,
                        split_dir,
                        w2v_path
    ):
    wv_model_original = Word2Vec.load(w2v_path)
    shards = os.listdir(shard_directory)
    shard_count = len(shards)
    total_functions, in_scope_function = set(), set()
    vnt, nvnt = 0, 0
    graphs = []
    for sc in range(1, shard_count + 1):
        shard_file = open(os.path.join(shard_directory + 'sysevr.shard' + str(sc)))
        shard_data = json.load(shard_file)
        try:
            for data in tqdm.tqdm(shard_data):
                file_name = data['file_name']    
                label = int(file_name.strip().split('_')[-1])
                code_text = utils.read_code_file(split_dir + file_name.strip() + ".c")
                nodes_file_path = parsed + file_name.strip() + ".c" + '/nodes.csv'
                edges_file_path = parsed + file_name.strip() + ".c" + '/edges.csv'
                nc = open(nodes_file_path)
                nodes_file = csv.DictReader(nc, delimiter='\t')
                nodes = [node for node in nodes_file]
                if len(nodes) == 0:
                    continue
                nodes = utils.read_csv(nodes_file_path)
                edges = utils.read_csv(edges_file_path)
                node_indices, node_ids, line_numbers, node_id_to_ln = extract_nodes_with_location_info(nodes)
                adjacency_list = create_adjacency_list(line_numbers, node_id_to_ln, edges, False)
                combined_graph = combine_control_and_data_adjacents(adjacency_list)
                data_point = reformat_code_line_graph(code_text, adjacency_list, wv_model_original, label)
                graphs.append(data_point)
        finally:
            pass
        del shard_data
    with open(os.path.join(output_dir, 'ggnn.json'), 'w') as output_file:
        json.dump(graphs, output_file)
    print(len(graphs))
#     return graphs
            

if __name__ == "__main__":

    files = utils.files_to_list("/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/renamed_code")
    split_dir = "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/renamed_code/"
    parsed = "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/code-slicer/joern/parsed/renamed_code/"
    json_file_path = "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/sysevr/process_slices.json"
    w2v_path = "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/saved_models/li_et_al_wv"
    shard_directory = "/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/sysevr/shards/"


    #data = process_slices(files, split_dir, parsed)

    #with open("/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/sysevr/process_slices_fixed_labels.json", 'w') as f:
    #    json.dump(data, f)

    #applyInputGeneration(json_file_path=json_file_path, 
    #               w2v_path=w2v_path,
    #               csv_dir=parsed,
    #               output_dir="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/sysevr/shards")

    extract_line_graph_data(shard_directory=shard_directory,
                            output_dir="/home/rob/Documents/PhD/Work/MyReVeal/ReVeal/fixed/data/processed/sysevr/",
                            split_dir=split_dir,
                            w2v_path=w2v_path) 
