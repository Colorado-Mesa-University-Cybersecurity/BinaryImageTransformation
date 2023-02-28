import time
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
def read_dataframe(filename: str, nrows: int = None) -> pd.DataFrame:
    """
    Read a csv file into a pandas dataframe.

    Args:
        filename: The name of the file to read.
        nrows: The number of rows to read from the file.

    Returns:
        A pandas dataframe containing the data from the file.
    """
    if nrows is not None:
        return pd.read_csv(filename, index_col='Index', nrows=nrows)
    return pd.read_csv(filename, index_col='Index')

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