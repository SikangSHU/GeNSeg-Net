import requests

# 1. Program Launch Command for fluorescence images.
resp = requests.post("http://localhost:9000/GeNSegNettest", data={'type': 'flu'})

