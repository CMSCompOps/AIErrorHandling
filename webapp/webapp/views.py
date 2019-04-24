from django.http import JsonResponse, HttpResponseBadRequest
from AIErrorHandling.models import Predictor

def predict(request):
    try:
        response = Predictor( **{
                "wf" : request.POST.wf,
                "tsk" : request.POST.tsk,
                })
        return JsonResponse(response)
    except:
        return HttpResponseBadRequest()
