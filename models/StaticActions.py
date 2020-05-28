#an implementation of https://docs.google.com/document/d/1SKHcrEnRBLwXJN_mdA9PDMeZtAmG5FrtQrbiP4E-LM0/edit

from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse, FileResponse
from AIErrorHandling.training.SitesErrorCodes.Tasks import Task
from workflowwebtools import workflowinfo
from cmstoolbox import sitereadiness
from collections import OrderedDict
import os
from xml.etree.ElementTree import ElementTree, Element, SubElement, Comment, tostring
import re
import xml.dom.minidom as minidom

def ParseErrors(wf):
    wfinfo = workflowinfo.WorkflowInfo( wf )
    err_summary={}
    #err_summary.setdefault(  )
    for tsk,errors in wfinfo.get_errors().items():
        for err_ in errors :
            try:
                err = int(err_)
            except :
                print( "error %s skipped" % err_ )
                continue

            if err not in err_summary:
                err_summary[err] = {'sum':0 , 'good_sites':[] , 'bad_sites':[]}
            err_summary[err]['sum'] += sum(errors[err_][site] for site in errors[err_] )
            err_summary[err]['good_sites'].extend( [ site for site in errors[err_] if sitereadiness.site_readiness( site ) == 'green' ] )
            err_summary[err]['bad_sites'].extend( [ site for site in errors[err_] if sitereadiness.site_readiness( site ) != 'green' ] )

    total_errs = sum( [ err_summary[err]['sum'] for err in err_summary ] )
    for err in err_summary:
        err_summary[err]['good_sites'] = list( set( err_summary[err]['good_sites'] ) )
        err_summary[err]['bad_sites'] = list( set( err_summary[err]['bad_sites'] ) )
        err_summary[err]['ratio'] = float(err_summary[err]['sum'])/total_errs
    err_summary_sorted = sorted( [(err,err_summary[err]) for err in err_summary ] , key= lambda x: x[1]['ratio'] , reverse=True)
    pred = OrderedDict( err_summary_sorted )
    main_err = err_summary_sorted[0][0] if len(err_summary_sorted) else -10
    main_err_ratio = err_summary_sorted[0][1]['ratio'] if len(err_summary_sorted) else 0
    pred["action"] = ""
    pred["description"] = ""

    if main_err_ratio > 0.3:
        pred['main_err'] = main_err
        bad_sites = err_summary[main_err]['bad_sites']
        if main_err in [99303, 99304]:
            pred['description'] = "condor issue"
            pred["action"] = "acdc"
        elif main_err in [99400, 99401]:
            pred["description"] = "condor issue"
            pred["action"] = "acdc"
        elif main_err in [50513]:
            pred["description"] = "scram/site issue"
            pred["action"] = "acdc"
        elif main_err in [71305 , 71304]:
            if len( bad_sites ) < 2:
                pred["action"] = "acdc"
                pred["description"] = "Timeout due to priority inversion, status is pending, sites in drain {0}".format( bad_sites )
            else:
                pred["action"] = "automatic-wait"
                pred["description"] = "Timeout due to priority inversion, status is pending, multiple sites in drain {0}".format( bad_sites )
        elif main_err in [99109]:
            if len(bad_sites) == 0:
                pred['action'] = 'acdc'
                pred['description'] = 'stage out error, site is not in drain'
            else:
                pred['action'] = 'automatic-wait'
                pred['description'] = 'stage out error, site is not backup'
        elif main_err in [71104, 71105]:
            if len(bad_sites) == 0:
                pred['action'] = 'acdc'
                pred['description'] = 'site put into the drain, site is backup'
            else:
                pred['action'] = 'automatic-wait'
                pred['description'] = 'site put into the drain, site is not backup'

    acdc_re = re.compile("(.*)_ACDC(.)_.*")
    acdc_re_parsed = acdc_re.match( wf )
    if acdc_re_parsed:
        username = acdc_re_parsed.groups()[0]
        acdc_index = acdc_re_parsed.groups()[1]
        pred['action'] = ''
        if pred['description'] != "":
            pred['description'] += " - no action recommended as the wf was acdc'ed before by {0}".format( username )

    return pred



def view(request):
    wf = request.GET.get('wf' , None )
    if not wf:
        return HttpResponseBadRequest("please set the workflow by passing the wf argument")

    pred = ParseErrors( wf )

    return JsonResponse( pred , safe=False )

def summary(request):
    wf = request.GET.get('wf' , None )
    if not wf:
        return HttpResponseBadRequest("please set the workflow by passing the wf argument")

    pred = ParseErrors( wf )
    root = Element('workflow')
    SubElement(root , 'name').text = wf
    errs = SubElement( root , 'errors' )
    for key in pred:
        if type(key) is int:
            err_el = SubElement( errs , 'error')
            err_id = SubElement( err_el , 'id' )
            err_id.text = str( key )
            err_gs = SubElement( err_el , 'good_sites' )
            for st in pred[key]['good_sites']:
                site_el = SubElement( err_gs , 'site')
                SubElement( site_el , 'name' ).text = st
            err_bs = SubElement( err_el , 'bad_sites' )
            for st in pred[key]['bad_sites']:
                site_el = SubElement( err_bs , 'site')
                SubElement( site_el , 'name' ).text = st
            err_sum = SubElement( err_el , 'count')
            err_sum.text = str( pred[key]['sum'] )
            err_ratio = SubElement( err_el , 'ratio' )
            err_ratio.text = '{0:.1f}%'.format( pred[key]['ratio']*100 )

    summary = SubElement(root , 'summary' )
    SubElement( summary , 'action' ).text = pred['action']
    SubElement( summary , 'main_error' ).text = str( pred.get( 'main_err' , -10 ) )
    SubElement( summary , 'description' ).text = pred.get( 'description' , "" )

    tree = ElementTree(root)
    xml_string = tostring( root , encoding='utf8', method='xml')
    xml_dom = minidom.parseString( xml_string )
    pi = xml_dom.createProcessingInstruction('xml-stylesheet',
                                             'type="text/xsl" href="/getFile/?f=wf_details.xsl"')
    xml_dom.insertBefore(pi, xml_dom.firstChild)
    response =  HttpResponse( xml_dom.toprettyxml(),
                              content_type= 'text/xml')
    return response


def listExistingWFs(request):
    root = Element('files')
    error_file_re = re.compile( 'workflowinfo_(.*)_errors.cache.json' )
    for wf_f in os.listdir( '/tmp/workflowinfo/' ):
        wf_f_err = error_file_re.match( wf_f )
        if wf_f_err:
            name = wf_f_err.groups()[0]
            errors = ParseErrors( name )
            wf_el = SubElement( root, 'wf')
            wf_name = SubElement( wf_el , 'name')
            wf_name.text = name
            lnk_el = SubElement( wf_el, 'lnk' )
            lnk_el.text = 'http://aieh.cern.ch:8052/static_action_details/?wf=' + name
            wf_action = SubElement( wf_el , 'action' )
            wf_action.text = errors['action']
            wf_action_description = SubElement( wf_el , 'action_description')
            wf_action_description.text = errors.get( 'description' , "" )
            wf_main_err = SubElement(wf_el , 'main_err')
            wf_main_err.text = str( errors.get( 'main_err' , -10 ) )
            wf_js = SubElement(wf_el , 'json')
            wf_js.text = 'http://aieh.cern.ch:8052/static_action/?wf=' + name 
            SubElement( wf_el , 'console' ).text = 'https://vocms0113.cern.ch/seeworkflow2/?workflow=' + name
            SubElement( wf_el , 'unified' ).text = 'https://cms-unified.web.cern.ch/cms-unified/report/' + name

    tree = ElementTree(root)
    xml_string = tostring( root , encoding='utf8', method='xml')
    xml_dom = minidom.parseString( xml_string )
    pi = xml_dom.createProcessingInstruction('xml-stylesheet',
                                             'type="text/xsl" href="/getFile/?f=wf_list.xsl"')
    xml_dom.insertBefore(pi, xml_dom.firstChild)
    response =  HttpResponse( xml_dom.toprettyxml(), #toprettyxml() ,
                              content_type= 'text/xml')
    return response
