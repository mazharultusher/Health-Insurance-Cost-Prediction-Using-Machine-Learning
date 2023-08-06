from django.shortcuts import render
from app.utils import *

# Create your views here.
def index(request):
    context = {
        'age': None,
        'sex': None,
        'bmi': None,
        'children': None,
        'smoker': None,
        'region': None,
        'result' : "None"
    }

    if request.method == 'POST':
        age = int(request.POST['age'])
        sex = int(request.POST['sex'])
        bmi = float(request.POST['bmi'])
        children = int(request.POST['children'])
        smoker = int(request.POST['smoker'])
        region = int(request.POST['region'])

        context['age'] = age
        context['sex'] = sex
        context['bmi'] = bmi
        context['children'] = children
        context['smoker'] = smoker
        context['region'] = region
        result = health_cost_predictor(age, sex, bmi, children, smoker,region)
        context['result'] = str(round(result, 3))
    
    return render(request, 'index.html', context)
