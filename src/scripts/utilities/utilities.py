import pandas as pd
import numpy as np

## threshhold function for null values
def null_threshhold(df, threshhold):
    """
    This function takes in a dataframe and a threshhold value and returns a list of columns that have null values
    greater than the threshhold value.
    """
    null_values = df.isnull().sum()
    null_values = null_values[null_values > 0]
    null_values = null_values[null_values/len(df) > threshhold]
    return null_values.index.tolist()

## function to drop columns with null values greater than a threshhold value
def drop_null_columns(df, threshhold):
    """
    This function takes in a dataframe and a threshhold value and drops columns that have null values greater than the
    threshhold value.
    """
    null_values = null_threshhold(df, threshhold)
    df = df.drop(null_values, axis=1)
    return df

## function to map numerical value to a categorical value
def map_numerical_to_categorical(df, column, bins, labels):
    """
    This function takes in a dataframe, a column name, a list of bins and a list of labels and returns a new column
    with the categorical values.
    """
    df[column] = pd.cut(df[column], bins=bins, labels=labels)
    return df

## function to map categorical value to a numerical value
def map_categorical_to_numerical(df, column, mapping):
    """
    This function takes in a dataframe, a column name and a dictionary of mapping values and returns a new column with
    the numerical values.
    """
    df[column] = df[column].map(mapping)
    return df

## function to drop columns with a single unique value
def drop_single_unique_value_columns(df):
    """
    This function takes in a dataframe and drops columns that have a single unique value.
    """
    single_unique_value_columns = df.columns[df.nunique() == 1]
    df = df.drop(single_unique_value_columns, axis=1)
    return df

## function to drop columns with a high cardinality
def drop_high_cardinality_columns(df, threshhold):
    """
    This function takes in a dataframe and a threshhold value and drops columns that have a cardinality greater than the
    threshhold value.
    """
    high_cardinality_columns = df.columns[df.nunique()/len(df) > threshhold]
    df = df.drop(high_cardinality_columns, axis=1)
    return df

## function to drop columns with a high correlation
def drop_high_correlation_columns(df, threshhold):
    """
    This function takes in a dataframe and a threshhold value and drops columns that have a correlation greater than the
    threshhold value.
    """
    correlation_matrix = df.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
    high_correlation_columns = [column for column in upper.columns if any(upper[column] > threshhold)]
    df = df.drop(high_correlation_columns, axis=1)
    return df

## function to make pviot table from a dataframe
def pivot_table(df, index, columns, values, aggfunc):
    """
    This function takes in a dataframe, a list of index columns, a list of column columns, a list of value columns and
    an aggregation function and returns a pivot table.
    """
    pivot_table = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)
    return pivot_table

## function to return quantile values of a column
def quantile_values(df, column, quantiles):
    """
    This function takes in a dataframe, a column name and a list of quantiles and returns the quantile values.
    """
    quantile_values = df[column].quantile(quantiles)
    return quantile_values


