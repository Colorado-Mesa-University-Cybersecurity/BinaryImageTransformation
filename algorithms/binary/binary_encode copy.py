import datetime, pathlib, platform
import bitstring
import numpy as np
import pandas as pd

from PIL import (
    Image
)


seed: int = 14

# set up pandas display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# print library and python versions for reproducibility
print(
    f'''
    Last Execution: {datetime.datetime.now()}
    python:\t{platform.python_version()}

    \tnumpy:\t\t{np.__version__}
    \tpandas:\t\t{pd.__version__}
    '''
)


def convert_samples_to_binary(df: pd.DataFrame, precision: 32 or 64 = 64, one: int = 128, zero: int = 0) -> np.ndarray:
    '''
        Take every sample in the df and convert it into a 2d array with: 
            each sample being a row,
            each column contains the elements of that sample's binary representation with the given precision
    
        binary representation is 1 bit for sign, 8 bits for exponent, precision - 9 bits for mantissa

        This will return a 3d array with the following shape:
            (number of samples, number of features, precision)
    '''

    out = []

    # get the values of the dataframe
    vals = df.values

    # convert the values to binary
    for i, sample in enumerate(vals):
        sample_out = []
        for j, feature in enumerate(sample):
            feature_out = [one if b == '1' else zero for b in bitstring.BitArray(float=feature, length=precision).bin]
            sample_out.append(feature_out)
        out.append(sample_out)

    return np.array(out, dtype=np.uint8)


def save_sample_as_image(sample: np.ndarray, path: str, name: str) -> None:
    '''
        Save the sample as an image
    '''

    # create the path if it doesn't exist
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # create the image
    img = Image.fromarray(sample, 'L')

    # save the image
    img.save(f'{path}{name}.png')

    return None


def generate_binary_images_from_df(
    df: pd.DataFrame, 
    label: str, 
    precision: 32 or 64 = 64, 
    path: str = './', 
    name: str = 'binary_image', 
    one: int = 128, 
    zero: int = 0
) -> None:
    '''
        Generate binary images from a dataframe

        The dataframe must only have 1 label column with all other labels removed
        The data must be in either integer or float format
    '''

    # get the labels from the samples
    labels = df[label].values

    # remove the label column from the dataframe
    new_df = df.copy().drop(columns=[label])


    # convert the samples to binary
    # samples = convert_samples_to_binary(new_df, precision, one = zero, zero = one)
    samples = convert_samples_to_binary(new_df, precision, one = one, zero = zero)

    # save the samples as images
    for i, sample in enumerate(samples):
        sample_name = f'{name}_{labels[i]}_{i}'

        save_sample_as_image(sample, path, sample_name)


    return None