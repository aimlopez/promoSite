import os
import subprocess

from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse


def home(request):
    if request.method == 'POST':
        if "hand_tracking" in request.POST:
            os.system('python scripts/handtracking.py')

    return render(request, 'handtracking/index.html')
