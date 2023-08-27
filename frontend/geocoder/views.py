import requests

from django.shortcuts import render


def index(request):
    if request.method == 'POST':
        file = request.FILES['file']
        url = 'http://127.0.0.1:5000/'

        files = {'file': file}
        response = requests.post(url, files=files)
        if response.status_code == 200:
            return render(request, "geocoder/index.html")
    else:
        return render(request, "geocoder/index.html")


def learning(request):
    return render(request, "geocoder/learning.html")
