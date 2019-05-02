from django.http import JsonResponse, HttpResponseBadRequest

from AIErrorHandling.models.SiteErrorCodeModelLoader import *
from AIErrorHandling.models import Predictor

def predict(request):

        wf = request.POST['wf']
        tsk = request.POST['tsk']
        print(wf,tsk)
        pred = Predictor( **{"wf": wf , "tsk": tsk } )
        return JsonResponse( pred , safe=False )

