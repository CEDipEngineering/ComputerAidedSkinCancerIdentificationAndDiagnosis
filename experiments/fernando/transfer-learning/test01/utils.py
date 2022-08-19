from os import listdir

def transform_diagnosis_to_numerical(diagnosis_result, dictonary):
    """
    Trasnform the diagnois to numerical numbers
    """
    return dictonary[diagnosis_result]

def add_prefix_to_string(string, prefix):
    """
    Add a sring prefix to the input string
    """
    return prefix+string

