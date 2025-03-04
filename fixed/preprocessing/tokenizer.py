import re
import nltk
nltk.download('punkt_tab')
import clang
from l_funcs import *

class Tokenizer:
    """
    A Tokenizer class that uses Clang's Python bindings (libclang) to parse and tokenise C/C++ source code.
    It supports tokenising entire translation units or specific functions within the source file.
    """
    # Creates the object, does the initial parse
    def __init__(self, path, tokenizer_type='original'):
        """
        Initialises the Tokenizer by creating a Clang index and parsing the specified file.

        Args:
            path (str): The filesystem path to the C/C++ source file to be parsed.
            tokenizer_type (str, optional): The type or mode of tokeniser. Defaults to 'original'.
        """
        # Create a Clang index object
        self.index = clang.cindex.Index.create()
        # Parse the input file into a translation unit
        self.tu = self.index.parse(path)
        # Store the trimmed path (extracting up to the last two folders)
        self.path = self.extract_path(path)
        # Maintain a symbol table for any necessary symbol tracking
        self.symbol_table = {}
        self.symbol_count = 1
        # Specify the tokenizer type
        self.tokenizer_type = tokenizer_type

    # To output for split_functions, must have the same path up to the last two folders
    def extract_path(self, path):
        """
        Extracts the partial filesystem path for consistent matching in split_functions.

        Args:
            path (str): The full filesystem path.

        Returns:
            str: The concatenated path string up to, but excluding, the last two folders.
        """
        return "".join(path.split("/")[:-2])

    def full_tokenize_cursor(self, cursor):
        """
        Tokenises the code associated with a given Clang cursor fully,
        skipping comments and returning processed literals and identifiers.

        Args:
            cursor (clang.cindex.Cursor): A cursor pointing to an element in the AST.

        Returns:
            list: A list of token strings representing the cursor's subtree.
        """
        # Retrieve all tokens for this cursor
        tokens = cursor.get_tokens()
        result = []
        for token in tokens:
            # Skip comment tokens
            if token.kind.name == "COMMENT":
                continue
            # Process literal tokens
            if token.kind.name == "LITERAL":
                result += self.process_literal(token)
                continue
            # Replace all identifiers with a generic "ID"
            if token.kind.name == "IDENTIFIER":
                result += ["ID"]
                continue
            # If elsewhere, use the actual spelling
            result += [token.spelling]
        return result

    def full_tokenize(self):
        """
        Tokenises the entire translation unit (TU).

        Returns:
            list: A list of tokens representing the entire file's source code,
                  excluding comments, with special handling of literals and identifiers.
        """
        # Get the root cursor from the translation unit
        cursor = self.tu.cursor
        return self.full_tokenize_cursor(cursor)

    def process_literal(self, literal):
        """
        Processes a literal token (e.g., integer, float, imaginary, or string),
        returning its spelling or a placeholder if relevant.

        Args:
            literal (clang.cindex.Token): A single token identified as a literal by Clang.

        Returns:
            list or str: The processed literal, either as its spelling (e.g., '123'),
                         or a generic placeholder like 'STRING' or 'NUM'.
        """
        cursor_kind = clang.cindex.CursorKind
        kind = literal.cursor.kind
        if kind == cursor_kind.INTEGER_LITERAL:
            return literal.spelling
        if kind == cursor_kind.FLOATING_LITERAL:
            return literal.spelling
        if kind == cursor_kind.IMAGINARY_LITERAL:
            return ["NUM"]
        if kind == cursor_kind.STRING_LITERAL:
            return ["STRING"]
        sp = literal.spelling
        if re.match('[0-9]+', sp) is not None:
            return sp
        return ["LITERAL"]

    def split_functions(self, method_only):
        """
        Splits a source file into tokens for each function-like entity.
        If 'method_only' is True, only C++ methods (CXX_METHOD) are considered; 
        otherwise, normal function declarations (FUNCTION_DECL) are also included.

        Args:
            method_only (bool): Flag indicating whether only methods should be captured
                                or all function declarations.

        Returns:
            list: A list, where each element is the list of tokens belonging to one function.
        """
        results = []
        cursor_kind = clang.cindex.CursorKind
        cursor = self.tu.cursor
        # Traverse the top-level children in the AST
        for c in cursor.get_children():
            filename = c.location.file.name if c.location.file != None else "NONE"
            extracted_path = self.extract_path(filename)

            # Check if this cursor is a function or method, and if it matches the path
            if (c.kind == cursor_kind.CXX_METHOD or (method_only == False and c.kind == cursor_kind.FUNCTION_DECL)) \
               and extracted_path == self.path:
                name = c.spelling
                # Tokenise the cursor's subtree
                tokens = self.full_tokenize_cursor(c)
                # Keep just the filename portion
                filename = filename.split("/")[-1]
                results += [tokens]

        return results


def tokenize(file_text):
    """
    Tokenises an in-memory string of C/C++ source code by writing it to a temporary file,
    creating a Tokenizer on that file, splitting out the functions, and returning tokens
    from the first function.

    Args:
        file_text (str): The source code content as a string.

    Returns:
        str or None: A space-separated string of tokens for the first discovered function,
                     or None if an error occurs (e.g., no functions).
    """
    try:
        # Write the provided source code to a temporary file
        c_file = open('/tmp/test1.c', 'w')
        c_file.write(file_text)
        c_file.close()

        # Create a Tokenizer instance on that file
        tok = Tokenizer('/tmp/test1.c')
        # Split out functions (including normal function declarations)
        results = tok.split_functions(False)

        # Return a space-joined string of the first function's tokens, if any
        return ' '.join(results[0])
    except:
        # If something went wrong (no functions or parse issue), return None
        return None

def symbolic_tokenize(code):
    tokens = nltk.word_tokenize(code)
    c_tokens = []
    for t in tokens:
        if t.strip() != '':
            c_tokens.append(t.strip())
    f_count = 1
    var_count = 1
    symbol_table = {}
    final_tokens = []
    for idx in range(len(c_tokens)):
        t = c_tokens[idx]
        if t in keywords:
            final_tokens.append(t)
        elif t in puncs:
            final_tokens.append(t)
        elif t in l_funcs:
            final_tokens.append(t)
        elif (idx+1) < len(c_tokens) and c_tokens[idx + 1] == '(':
            if t in keywords:
                final_tokens.append(t)
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t])
            idx += 1

        elif t.endswith('('):
            t = t[:-1]
            if t in keywords:
                final_tokens.append(t + '(')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '(')
        elif t.endswith('()'):
            t = t[:-2]
            if t in keywords:
                final_tokens.append(t + '( )')
            else:
                if t not in symbol_table.keys():
                    symbol_table[t] = "FUNC" + str(f_count)
                    f_count += 1
                final_tokens.append(symbol_table[t] + '( )')
        elif re.match("^\"*\"$", t) is not None:
            final_tokens.append("STRING")
        elif re.match("^[0-9]+(\.[0-9]+)?$", t) is not None:
            final_tokens.append("NUMBER")
        elif re.match("^[0-9]*(\.[0-9]+)$", t) is not None:
            final_tokens.append("NUMBER")
        else:
            if t not in symbol_table.keys():
                symbol_table[t] = "VAR" + str(var_count)
                var_count += 1
            final_tokens.append(symbol_table[t])
    return ' '.join(final_tokens)