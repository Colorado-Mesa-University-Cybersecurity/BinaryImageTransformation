import time
from typing import List
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
import numpy as np
import bitstring
from PIL import Image
import os
import shutil


def print_library_versions() -> None:
    """
    Print out the versions for the libraries used
    """
    print(f"pandas version:".ljust(25), f"{pd.__version__}")
    print(f"matplotlib version:".ljust(25), f"{matplotlib.__version__}")
    print(f"numpy version:".ljust(25), f"{np.__version__}")
    print(f"bitstring version:".ljust(25), f"{bitstring.__version__}")
    print(f"joblib version:".ljust(25), f"{joblib.__version__}")
    print(f"PIL version:".ljust(25), f"{Image.__version__}")


def compare_3d_lists(list1: List[List[List]], list2: List[List[List]]) -> bool:
    """
    Compare two 3D lists and return True if they have the same shape and elements, and False otherwise.

    Args:
        list1: The first 3D list to compare.
        list2: The second 3D list to compare.

    Returns:
        A boolean indicating whether the two lists are equivalent or not.
    """
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if len(list1[i]) != len(list2[i]):
            return False
        for j in range(len(list1[i])):
            if len(list1[i][j]) != len(list2[i][j]):
                return False
            for k in range(len(list1[i][j])):
                if list1[i][j][k] != list2[i][j][k]:
                    return False
    return True


def runtime_analysis_between_two_functions(fun1 : callable, fun2 : callable, 
                                          df: pd.DataFrame, precision: 32 or 64 = 64, one: int = 128, zero: int = 0):
    """
    Compare the runtime of two functions on the same input list.

    Returns:
        A list containing the results of the two functions.
    """
    start_time = time.time()
    fun1_results = fun1(df, precision=precision, one=one, zero=zero)
    end_time = time.time()
    fun1_time = end_time - start_time

    start_time = time.time()
    fun2_results = fun2(df, precision=precision, one=one, zero=zero)
    end_time = time.time()
    fun2_time = end_time - start_time

    print(f"fun1 took {fun1_time} seconds to run and fun2 took {fun2_time} seconds")
    if fun1_time < fun2_time:
        print(f"fun1 is {fun2_time/fun1_time} times faster than fun2")
    else:
        print(f"fun2 is {fun1_time/fun2_time} times faster than fun1")
        
    return fun1_results, fun2_results


def display_images(images: List[List[List]], n_images: int):
    """
    Display a list of images.

    Args:
        images: A list of images to display.
        n_images: The number of images to display.
    """
    fig, axes = plt.subplots(1, n_images, figsize=(20, 20))
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray')
    plt.show()
    
    
def add_id_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Adds a new column to a pandas DataFrame with unique identifiers for each value in a specified column.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        column_name (str): The name of the column to use for generating unique identifiers.

    Returns:
        pandas.DataFrame: The modified DataFrame with the new 'Id' column.
    """
    if 'Id' not in df.columns:
        df['Id'] = ''

    unique_values = df[column_name].unique().tolist()
    values_dict = {value: 0 for value in unique_values}
    for index, row in df.iterrows():
        value = row[column_name]
        values_dict[value] += 1
        name = str(values_dict[value]) + "-" + value.split('-')[1]
        df.at[index, 'Id'] = name
    return df


def process_samples_as_floats(sample: List[float], label : str, directory : str, precision: int, one: int, zero: int) -> List[List[int]]:
    """
    Takes a sample and returns a binary representation of the sample as a 2D list of 1s and 0s.
    
    Args:
        sample (List[float]): A list of features for a sample.
        label (str): The label for the sample.
        directory (str): The directory to save the images to.
        
    Returns:
        List[List[int]]: A 2D list of 1s and 0s representing the binary values of the features in the sample.
    """
    sample_out = []
    for feature in sample:
        feature_out = [one if b == '1' else zero for b in bitstring.BitArray(float=feature, length=precision).bin]
        sample_out.append(feature_out)
    image = Image.new('L', (len(sample_out[0]), len(sample_out)))
    image.putdata([value for row in sample_out for value in row])
    image.save(directory + '/' + label +  '.png')


def process_samples_as_type(sample: List[float], label : str, directory : str, precision: int, one: int, zero: int, feature_types : List = None) -> List[List[int]]:
    """
    Takes a sample and returns a binary representation of the sample as a 2D list of 1s and 0s.
    
    Args:
        sample (List[float]): A list of features for a sample.
        label (str): The label for the sample.
        directory (str): The directory to save the images to.
        feature_types (List[int]): A list of integers representing the type of each feature in the dataframe.
        
    Returns:
        List[List[int]]: A 2D list of 1s and 0s representing the binary values of the features in the sample.
    """
    sample_out = []
    n_columns = len(sample)
    n_rows = n_columns

    for i, feature in enumerate(sample):
        if feature_types[i] == 0: # floating point
            feature_out = [one if b == '1' else zero for b in bitstring.BitArray(float=feature, length=precision).bin]
        elif feature_types[i] == 1: # integer
            feature_out = [one if b == '1' else zero for b in bitstring.BitArray(int=int(feature), length=precision).bin]
        else: # boolean value
            if feature == True:
                feature_out = [one]*precision
            else:
                feature_out = [zero]*precision
            
        sample_out.append(feature_out)
        
        # Update the number of rows if necessary
        if len(feature_out) > n_rows:
            n_rows = len(feature_out)

    # Append new rows with zeros if necessary
    for i in range(n_rows - n_columns):
        sample_out.append([zero]*precision)

    image = Image.new('L', (len(sample_out[0]), len(sample_out)))
    image.putdata([value for row in sample_out for value in row])
    image.save(directory + '/' + label +  '.png')


def convert_samples_to_binary(x: pd.DataFrame, labels: pd.DataFrame, directory : str, precision: 32 or 64 = 64, one: int = 128,
                                zero: int = 0, n_jobs: int = -1, feature_types : List[int] = None) -> np.ndarray:
    """
    Takes a pandas dataframe of numerical values and returns a binary representation of the dataframe as a matrix.
    
    Args:
        x (pd.DataFrame): The pandas dataframe x values to be converted to binary.
        labels (pd.DataFrame): The pandas dataframe ids values to be converted to binary.
        directory (str): The directory to save the images to.
        precision (int): The number of bits to use when converting a feature to binary.
        one (int): The value to use when the binary representation of a feature contains a 1.
        zero (int): The value to use when the binary representation of a feature contains a 0.
        n_jobs (int): The number of processors to use when converting the dataframe to binary.
        feature_types (List[int]): A list of integers representing the type of each feature in the dataframe.
        
    Returns:
        ndarray: A ndarray representing the binary values of the features in the dataframe.
    """
    initial_time = time.time()
    x_vals = x.values
    label_vals = labels.values
    if not os.path.exists(directory):
        os.makedirs(directory)
    if feature_types == None:
        Parallel(n_jobs=n_jobs)(delayed(process_samples_as_floats)(sample, label_vals[i], directory, precision, one, zero) for i,sample in enumerate(x_vals))    
    else:
        Parallel(n_jobs=n_jobs)(delayed(process_samples_as_type)(sample, label_vals[i], directory, precision, one, zero, feature_types) for i,sample in enumerate(x_vals))
    end_time = time.time()
    print(f"Time to convert {len(x_vals)} samples to binary: {end_time - initial_time} seconds. Seconds per sample = {(end_time - initial_time)/len(x_vals)}")

def get_column_data_types(df: pd.DataFrame) -> List[str]:
    """
    Given a pandas DataFrame, determine the data type of each column based on the types of its values.
    Returns a list of data types for each column in the DataFrame.

    Args:
        df: pandas.DataFrame The DataFrame to analyze.

    Returns:
        List[str]: A list of data types for each column in the DataFrame. Each data type is one of the following: 'float', 'int', 'bool', or 'other'.
    """
    col_types = []
    for col in df.columns:
        col_type = None
        for val in df[col]:
            val_type = type(val)
            if val_type == float:
                col_type = 'float'
            elif val_type == int:
                if col_type != None and (col_type == 'float' or col_type == 'other'):
                    break
                col_type = 'int'
            elif val_type == bool:
                if col_type != None and col_type != 'bool':
                    break
                col_type = 'bool'
            else:
                col_type = 'other'
                break
        if col_type == 'float':
            col_type = 0
        elif col_type == 'int':
            col_type = 1
        elif col_type == 'bool':
            col_type = 2
        else:
            #throw error
            raise ValueError(f'Column "{col}" contains values of an unsupported type. Try to look at your data or clean before using.')
        col_types.append(col_type)
    return col_types

def organize_photos_in_folders(image_directory : str, Y : pd.DataFrame) -> None:
    '''
        Organize the photos in the folder into subfolders based on the label
        
        Args:
            image_directory (str): The directory to save the images to.
            Y (pd.DataFrame): The pandas dataframe ids values to be converted to binary.
    '''
    dirs = Y.unique().tolist()
    if not os.path.exists(image_directory + '/data'):
        os.mkdir(image_directory + '/data')
    new_dir = image_directory + '/data/'
    if not os.path.exists(new_dir+'Train/'):
        os.mkdir(new_dir+'Train/')
    for i in dirs:
        i = str(i).split('-')[1]
        if not os.path.exists(new_dir+'Train/'+i):
            os.mkdir(new_dir+'Train/'+i)
    
    total_images = 0
    type_counts = {value.split("-")[1]: 0 for value in dirs}
    for file in os.listdir(image_directory):
        try:
            dir = file.split("-")[1].split(".")[0]
            type_counts[dir]+=1
        except:
            continue
        
        shutil.move(f"{image_directory}/{file}", f"{new_dir}Train/{dir}/{file}")
        total_images += 1
    print(total_images)
    print(type_counts)

def order_columns_by_correlation(df: pd.DataFrame, label: str, isIndx : bool = False) -> list:
    '''
        Order the columns of the dataframe in a sequence where the first element is the column most correlated with the label
            and every success element is the remaining column most correlated with its predecessor
            
        Args:
            df (pd.DataFrame): The dataframe to order the columns of
            label (str): The label to order the columns by (the column to be predicted)
            isIndx (bool): Whether or not an index label is present in the dataframe
    '''

    current_columns : pd.DataFrame = df.columns.copy()
    new_df          : pd.DataFrame = df.copy()
    new_column_order: list = []
    label_class_map : dict = {}

    print(f'ordering columns by correlation: {label}, {len(current_columns)}, {df[label].unique()}')

    for i, category in enumerate(df[label].unique()):
        label_class_map[category] = i

    new_df[label] = new_df[label].map(label_class_map)


    current: str = label
    last: str = None
    stop_condition = 2 if isIndx else 1

    while len(current_columns) > stop_condition:
        last = current
        current_columns = current_columns.drop(current)
        current = new_df[current_columns].corrwith(new_df[current]).abs().idxmax()
        new_column_order.append(current)

    if isIndx:
        new_column_order.insert(0, 'Id')
    new_column_order.append(label)
    return new_column_order