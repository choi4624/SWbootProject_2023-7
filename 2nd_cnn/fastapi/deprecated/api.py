import requests

image_file = open("path/to/your/image.jpg", "rb")
files = {"image_file": image_file}

response = requests.post("http://127.0.0.1:8000/predict/", files=files)
if response.status_code == 200:
    result = response.json()
    print("Predicted class:", result["predicted_class"])
else:
    print("Error:", response.text)
