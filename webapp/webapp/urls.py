#from django.contrib import admin
from django.urls import path
#from AIErrorHandling.webapp.webapp import views as views
from AIErrorHandling.models.StaticActions import view as static_view, summary as static_summary, listExistingWFs

import os.path
from django.http import HttpResponse, HttpResponseBadRequest, FileResponse

print("url is loaded")

def getFile(request):
    file_name = request.GET.get('f' , None )    
    if file_name:
        ext = file_name.split('.')[-1]
        file_name = '/home/aieh/AIErrorHandling/webapp/files/' + file_name
        print(file_name)
        if os.path.isfile( file_name ):
            f = open( file_name )
            if ext in ['html' , 'htm' , 'js' , 'css']:
                return HttpResponse( f )
            elif ext in ['xml' , 'xsl']:
                return HttpResponse( f , content_type= 'text/xml' )
            else:
                return FileResponse( f )
        else:
            return HttpResponseBadRequest("file {0} doesn't exist".format( file_name ) )
    else:
        return HttpResponseBadRequest( "please pass the file name via f argument")


urlpatterns = [
    #path('predict/', views.predict),
    path('static_action/', static_view),
    #path('test/' , views.predict2 ),
    path('listExistingWFs/', listExistingWFs ),
    path('static_action_details/' , static_summary ),
    path('getFile/' , getFile )
]
