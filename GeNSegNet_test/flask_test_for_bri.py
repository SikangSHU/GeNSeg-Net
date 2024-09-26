import requests

# 2. Program Launch Command for brightfield images.
resp = requests.post("http://localhost:9000/GeNSegNettest", data={'type': 'bri'})