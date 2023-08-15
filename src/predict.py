import numpy as np
import pandas as pd
from typing import List
from config import paths
from utils import read_csv_in_directory, save_dataframe_as_csv
from logger import get_logger
from Classifier import Classifier, predict_with_model
from preprocessing.pipeline import create_pipeline, run_testing_pipeline
from schema.data_schema import load_saved_schema


logger = get_logger(task_name="predict")


def create_predictions_dataframe(
        predictions_arr: np.ndarray,
        class_names: List[str],
        ids: pd.Series,
        id_field_name: str,
        return_proba: bool = False
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Performs the following transformations:
    - converts to pandas dataframe
    - adds class labels as headers for columns containing predicted probabilities
    - inserts the id column

    Args:
        predictions_arr (np.ndarray): Predicted probabilities from predictor model.
        class_names List[str]: List of target classes (labels).
        ids (pd.Series): id field values for the provided data.
        id_field_name (str): Name to use for the id field.
        return_proba (bool): If true, returns the probabilities of the predicted classes.

    Returns:
        Predictions as a pandas dataframe
    """
    if predictions_arr.shape[1] != len(class_names):
        raise ValueError(
            "Length of class names does not match number of prediction columns"
        )
    predictions_df = pd.DataFrame(predictions_arr, columns=class_names)
    if len(predictions_arr) != len(ids):
        raise ValueError("Length of ids does not match number of predictions")
    if not return_proba:
        predictions_df['prediction'] = predictions_df.idxmax(axis=1)
        predictions_df.insert(0, id_field_name, ids)
        return predictions_df[[id_field_name, 'prediction']]

    predictions_df.insert(0, id_field_name, ids)
    return predictions_df


def run_batch_predictions() -> None:
    """
        Run batch predictions on test data, save the predicted probabilities to a CSV file.

        This function reads test data from the specified directory,
        loads the preprocessing pipeline and pre-trained predictor model,
        transforms the test data using the pipeline,
        makes predictions using the trained predictor model,
        adds ids into the predictions dataframe,
        and saves the predictions as a CSV file.
        """
    test_data = read_csv_in_directory(paths.TEST_DIR)
    data_schema = load_saved_schema(paths.SAVED_SCHEMA_DIR_PATH)
    model = Classifier.load(paths.PREDICTOR_DIR_PATH)
    pipeline = create_pipeline(data_schema)
    features = data_schema.features
    x_test = test_data[features]

    logger.info("Transforming the data...")
    x_test = run_testing_pipeline(x_test, data_schema, pipeline)
    logger.info("Making predictions...")
    predictions_arr = predict_with_model(model, x_test)
    predictions_df = create_predictions_dataframe(
        predictions_arr,
        data_schema.target_classes,
        test_data[data_schema.id],
        data_schema.id,
        return_proba=True
    )
    logger.info("Saving predictions...")
    save_dataframe_as_csv(
        dataframe=predictions_df, file_path=paths.PREDICTIONS_FILE_PATH
    )

    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()
