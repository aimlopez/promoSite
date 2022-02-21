import os
import subprocess

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse


def home(request):
    if request.method == 'POST':
        if "hand_tracking" in request.POST:
            os.system('python scripts/handtracking.py')
        elif "body_posture" in request.POST:
            os.system('python scripts/bodyposture.py')
        elif "object_recognition" in request.POST:
            os.system('python scripts/objectrecognition.py')

    return render(request, 'handtracking/index.html')
