import requests
url = 'http://localhost:5000/results'
r = requests.post(url,json={'stockprice':5})
print(r.json())
