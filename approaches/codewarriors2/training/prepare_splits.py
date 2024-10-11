def get_balanced_dataset(meta_df, column, fraction=0.2):
    """
    Splits a dataset into balanced training and validation sets.

    Ensures the validation set contains a specified fraction of examples from each unique value in `column`.

    Args:
        meta_df (pandas.DataFrame): The dataset containing the metadata.
        column (str): The column name to balance by.
        fraction (float, optional): The fraction of each category to include in the validation set. Defaults to 0.2.

    Returns:
        list: A list of image names for validation.
        list: A list of image names for training.
    """
    validation_examples = []
    for val in meta_df[column].unique():
        subdf = meta_df[meta_df[column] == val]
        validation_examples += list(subdf.sample(frac=fraction)['Image Name'])
    training_examples = set(meta_df['Image Name'].unique()).difference(set(validation_examples))
    return validation_examples, list(training_examples)