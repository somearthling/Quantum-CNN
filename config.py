import argparse
import ast
import pennylane as qml

def int_or_none(x):
    '''
    Helper function to parse command line arguments.
    '''
    if x == 'None':
        return None
    return int(x)

def str_to_bool(x):
    '''
    Helper function to parse command line arguments.
    '''
    if x == 'True':
        return True
    elif x == 'False':
        return False
    else:
        raise ValueError("Invalid boolean value")
    
def str_to_list(x):
    '''
    Helper function to parse command line arguments.
    '''
    try:
        return ast.literal_eval(x)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid list value")

def get_args():
    '''
    Parses the command line arguments.
    '''
    parser = argparse.ArgumentParser()

    args_config = {
        "--batchsizes": {
            "type": int,
            "help": "batch sizes",
            "default": [32],
            "nargs": "+"
        },
        "--trainingruns": {
            "type": int,
            "help": "number of independent training runs",
            "default": 3
        },
        "--filters": {
            "type": int,
            "help": "numbers of filters",
            "default": [1],
            "nargs": "+"
        },
        "--convolutiontypes": {
            "type": str_to_bool,
            "help": "True for 246-parameter convolution, False for 82 parameter convolution",
            "default": [True],
            "nargs": "+"
        },
        "--pooltypes": {
            "type": str,
            "help": "pooling type measure or trace",
            "default": ['trace'],
            "nargs": "+"
        },
        "--separate": {
            "type": int,
            "help": "separate parameters for pooling gates? 0 for no, 1 for 2 sets of parameters when double pooling, 2 for separate parameters for each pooling gate",
            "default": [2],
            "nargs": "+"
        },
        "--doublepool": {
            "type": str_to_bool,
            "help": "double pooling?",
            "default": [True],
            "nargs": "+"
        },
        "--structurestring": {
            "type": str,
            "help": "structure string - 'p' for pooling, 'i' for independent U4 gates, 'r' for random U4 gates,"
            " 'b' to use the qml.broadcast function with all-to-all pattern, 's' for strongly entangling gates.",
            "default": ['ii ii'],
            "nargs": "+"
        },
        "--classes": {
            "type": int_or_none,
            "help": "number of classes",
            "default": [None],
            "nargs": "+"
        },
        "--ranges": {
            "type": str_to_list,
            "help": "ranges for strongly entangling gates",
            "default": [None],
            "nargs": "+"
        },
        "--imprimitives": {
            "type": str_to_list,
            "help": "lists of unparameterized 2-qubit gates for strongly entangling gates",
            "default": [None],
            "nargs": "+"
        },
        "--dense_pool_type": {
            "type": str,
            "help": "pooling type in dense layer meas or trace",
            "default": ['trace'],
            "nargs": "+"
        },
        "--trainperf": {
            "type": str_to_bool,
            "help": "evaluate train performance?",
            "default": False
        },
        "--testbatch": {
            "type": int,
            "help": "test batch size",
            "default": 32
        },
        "--datasets": {
            "type": str,
            "help": "torchvision dataset",
            "default": ['MNIST'],
            "nargs": "+"
        },
        "--iterations": {
            "type": int_or_none,
            "help": "numbers of iterations",
            "default": [None],
            "nargs": "+"
        },
        "--rates": {
            "type": float,
            "help": "learning rates",
            "default": [0.016],
            "nargs": "+"
        },
        "--testing": {
            "type": str_to_bool,
            "help": "testing?",
            "default": False
        },
        "--device": {
            "type": int,
            "help": "device",
            "default": 0
        },
    }

    for arg, config in args_config.items():
        parser.add_argument(arg, **config)

    try:
        return parser.parse_args(), parser
    except SystemExit as e:
        print(f"Error: {e}")
        parser.print_help()
        raise