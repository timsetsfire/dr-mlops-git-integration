"""
Usage:
    python datarobot-predict.py <input-file.csv>
 
This example uses the requests library which you can install with:
    pip install requests
We highly recommend that you update SSL certificates with:
    pip install -U urllib3[secure] certifi
"""
import sys
import json
import requests

API_URL = "https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions"  # noqa
API_KEY = "NWUwYWJlNTkwMzM0MTkwNTE2MTc0Mzg2OmVmU3ZoakVkSTZzbHBWM0lCZFl4djk5UHZjVy83N1M2WGlIYjgvb00zTGc9"
DATAROBOT_KEY = "544ec55f-61bf-f6ee-0caf-15c7f919a45d"

DEPLOYMENT_ID = "5f2029666ac9eb0866055e48"

MAX_PREDICTION_FILE_SIZE_BYTES = 52428800  # 50 MB


class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""


def make_datarobot_deployment_predictions(data, deployment_id):
    """
    Make predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html
 
    Parameters
    ----------
    data : str
        Feature1,Feature2
        numeric_value,string
    deployment_id : str
        The ID of the deployment to make predictions with.
 
    Returns
    -------
    Response schema:
        https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html#response-schema
 
    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {
        "Content-Type": "text/plain; charset=UTF-8",
        "Authorization": "Bearer {}".format(API_KEY),
        "DataRobot-Key": DATAROBOT_KEY,
    }

    url = API_URL.format(deployment_id=deployment_id)
    # Parametrize Prediction Explanations with query parameters listed in the docs:
    # https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html#request-pred-explanations
    params = {
        "maxExplanations": 3,
        "thresholdHigh": 0.5,
        "thresholdLow": 0.15,
    }
    # Make API request for predictions
    predictions_response = requests.post(
        url, data=data, headers=headers, params=params,
    )
    _raise_dataroboterror_for_status(predictions_response)
    # Return a Python dict following the schema in the documentation
    return predictions_response.json()


def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = "{code} Error: {msg}".format(
            code=response.status_code, msg=response.text
        )
        raise DataRobotPredictionError(err_msg)


def main(filename, deployment_id):
    """
    Return an exit code on script completion or error. Codes > 0 are errors to the shell.
    Also useful as a usage demonstration of
    `make_datarobot_deployment_predictions(data, deployment_id)`
    """
    if not filename:
        print(
            "Input file is required argument. "
            "Usage: python datarobot-predict.py <input-file.csv>"
        )
        return 1
    data = open(filename, "rb").read()
    data_size = sys.getsizeof(data)
    if data_size >= MAX_PREDICTION_FILE_SIZE_BYTES:
        print(
            "Input file is too large: {} bytes. " "Max allowed size is: {} bytes."
        ).format(data_size, MAX_PREDICTION_FILE_SIZE_BYTES)
        return 1
    try:
        predictions = make_datarobot_deployment_predictions(data, deployment_id)
    except DataRobotPredictionError as exc:
        print(exc)
        return 1
    print(json.dumps(predictions, indent=4))
    return 0


if __name__ == "__main__":
    filename = sys.argv[1]
    sys.exit(main(filename, DEPLOYMENT_ID))
