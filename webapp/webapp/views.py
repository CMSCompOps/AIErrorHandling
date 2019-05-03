from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse

#from AIErrorHandling.models.SiteErrorCodeModelLoader import *
from AIErrorHandling.models import Predictor
from AIErrorHandling.training.SitesErrorCodes import SitesErrorCodes_path
from os.path import isfile, join

def predict(request):
        wf = request.GET.get('wf' , None )
        if not wf:
                return HttpResponseBadRequest("please set the workflow first")

        jsoninputfile = request.GET.get("fromjson" , None)
        pred = None
        if jsoninputfile :
                inputjsonfile = join( SitesErrorCodes_path , "data" , jsoninputfile + ".json" )
                if isfile( inputjsonfile ) :
                        #actionshistory_03042019
                        pred = Predictor( **{"wf": wf , "sourcejson":inputjsonfile } )
                else :
                        return HttpResponseBadRequest("the requested input json file is not available")
        else :
                tsk = request.GET.get('tsk', None)
                if not tsk :
                        return HttpResponseBadRequest("tsk parameter is not set")

                #try:
                pred = Predictor( **{"wf": wf , "tsk":tsk } )
                #except BaseException as err:
                #        return HttpResponseBadRequest("can not load the predictions. Problem could be with the certificate. Please check log files." + "</br>" +  str(err))

        return JsonResponse( pred , safe=False )


def predict2(request):
        return JsonResponse( {'a':222 , 'b':2} , safe=True )

