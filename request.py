import requests
from requests.structures import CaseInsensitiveDict

url = "https://reqbin.com/echo/post/json"
headers = CaseInsensitiveDict()
headers["Accept"] = "application/json"
headers["Content-Type"] = "application/json"

data = """
{
  "Id": 78912,
  "Customer": "Jason Sweet",
}
"""

resp = requests.post(url, headers=headers, data=data)
print(resp.status_code)