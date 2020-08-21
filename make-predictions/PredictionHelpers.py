import requests
import os

DATAROBOT_API_TOKEN = os.environ["DATAROBOT_API_TOKEN"]

url = "https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/5f2029666ac9eb0866055e48/predictionExplanations"

payload = {}
files = [
  ('file', open('/Users/timothy.whittaker/Desktop/mlops-demo-2020-07/data/loss_cost_demo_short.csv','rb'))
]
headers = {
  'Content-Type': 'text/plain; charset=UTF-8',
  'Authorization': 'Bearer {}'.format(DATAROBOT_API_TOKEN),
  'DataRobot-Key': '544ec55f-61bf-f6ee-0caf-15c7f919a45d'
}

response = requests.request("POST", url, headers=headers, data = payload, files = files)
