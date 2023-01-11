import pandas as pd


def clean_data(path: str) -> pd.DataFrame:
    """Clean data and return only the necessary columns

    Args:
        path (str): location of the file on our computers

    Returns:
        pd.DataFrame: the output dataframe with the correct columns
    """
    assert path[-4:] == '.csv'
    df = pd.read_csv(path)
    return df[['wine', 'acidity', 'quality']]
