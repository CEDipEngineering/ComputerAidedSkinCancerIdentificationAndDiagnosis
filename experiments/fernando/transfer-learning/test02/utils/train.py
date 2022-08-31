from tensorflow import keras
import pandas as pd

def series_to_array(pandas_series):
    classes_dictionary = get_classes_dictionary(pandas_series)
    pandas_series_to_integer = series_to_integer(pandas_series, classes_dictionary)
    return keras.utils.to_categorical(pandas_series_to_integer)

def series_to_integer(pandas_series, dictionary):
    pandas_series_tolist = pandas_series.tolist()
    transformation = list()
    for value in pandas_series_tolist:
        integer_value = categorical_to_int(value, dictionary)
        transformation.append(integer_value)
    return pd.Series(transformation)

def categorical_to_int(value, dictionary):
    return dictionary[value]

def get_classes_dictionary(targets):
    res = dict()
    classes = targets.unique().tolist()
    for index in range(len(classes)):
        class_name = classes[index]
        res[class_name] = index
    return res