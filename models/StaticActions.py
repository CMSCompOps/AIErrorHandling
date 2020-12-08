#an implementation of https://docs.google.com/document/d/1SKHcrEnRBLwXJN_mdA9PDMeZtAmG5FrtQrbiP4E-LM0/edit

from django.http import JsonResponse, HttpResponseBadRequest, HttpResponse, FileResponse
from AIErrorHandling.training.SitesErrorCodes.Tasks import Task
from workflowwebtools import workflowinfo, manageactions
from cmstoolbox import sitereadiness
from collections import OrderedDict
import os
from xml.etree.ElementTree import ElementTree, Element, SubElement, Comment, tostring
import re
import xml.dom.minidom as minidom

sitereadiness.TIMESTAMP = None

def trimSiteName(site):
    parts = site.split('_')
    if len(parts) == 4:
        if sitereadiness.site_drain_status( site ) in ['enabled' , 'disabled']:
            return site
        else:
            return '_'.join( parts[:3] )
    else:
        return site

def nBadSites(wfinfo):
    bad_sites = []
    for tsk,errors in wfinfo.get_errors(True).items():
        for err_ in errors :
            if err_ == 'NotReported':
                err = -2
            else:
                try:
                    err = int(err_)
                except :
                    print( "error %s skipped" % err_ )
                    continue

            bad_sites.extend( [ site for site in errors[err_] if sitereadiness.site_drain_status( trimSiteName(site) ) != 'enabled' and site != 'NoReportedSite' and "T3_" not in site] )
    return len( set(bad_sites) )

    

def ParseErrors(wfinfo, include_xrootds=False):
    sites_summary={}
    for tsk,info in wfinfo.get_recovery_info().items():
        tsk =tsk[ len(wfinfo.workflow)+2: ]
        sites_summary[tsk] = {
            'good_sites':[site for site in info['sites_to_run'] if sitereadiness.site_drain_status( trimSiteName(site) ) == 'enabled' and site != 'NoReportedSite' and "T3_" not in site] ,
            'bad_sites':[site for site in info['sites_to_run'] if sitereadiness.site_drain_status( trimSiteName(site) ) != 'enabled'  and site != 'NoReportedSite' and "T3_" not in site]
        }

    ret = {'workflow': wfinfo.workflow,
           'parameters': {
               'Action': '',
               'Reasons': ['AIEH'],
               'ACDCs': [],
               'Parameters': {
                   tsk: {'memory': '',
                         'cores': '',
                         'xrootd': '',
                         'secondary': '',
                         'sites': info['good_sites'],
                         'bad_sites':info['bad_sites']}
                   for tsk, info in sites_summary.items()
               }
           },
           'additional_info':{
               'errors': {},
               'nBadSites': nBadSites( wfinfo )
           }
    }

    ret2 = {'workflow': wfinfo.workflow,
           'parameters': {
               'Action': '',
               'Reasons': ['AIEH'],
               'ACDCs': [],
               'Parameters': {
                   tsk: {'memory': '',
                         'cores': '',
                         'xrootd': '',
                         'secondary': '',
                         'sites': info['good_sites'],
                         'bad_sites':info['bad_sites']}
                   for tsk, info in sites_summary.items()
               }
           },
           'additional_info':{
               'errors': {},
               'nBadSites': nBadSites( wfinfo )
           }
    }


    if ret['additional_info']['nBadSites'] != 0 and not include_xrootds:
        return ret

    err_summary={}
    err_summary_pertask={}
    for tsk,errors in wfinfo.get_errors(True).items():
        err_summary_pertask[tsk] = {}
        for err_ in errors :
            if err_ == 'NotReported':
                err = -2
            else:
                try:
                    err = int(err_)
                except :
                    print( "error %s skipped" % err_ )
                    continue


            nErrors = sum(errors[err_][site] for site in errors[err_] )
            err_gsites = [ site for site in errors[err_] if sitereadiness.site_drain_status( trimSiteName(site) ) == 'enabled' and site != 'NoReportedSite' and "T3_" not in site]
            err_bsites = [ site for site in errors[err_] if sitereadiness.site_drain_status( trimSiteName(site) ) != 'enabled' and site != 'NoReportedSite' and "T3_" not in site]

            if err not in err_summary:
                err_summary[err] = {'sum':nErrors , 'good_sites':err_gsites , 'bad_sites':err_bsites}
            else:
                err_summary[err]['sum'] += nErrors
                err_summary[err]['good_sites'].extend( err_gsites )
                err_summary[err]['bad_sites'].extend( err_bsites )

            if err not in err_summary_pertask[tsk]:
                err_summary_pertask[tsk][err] = {'sum':nErrors , 'good_sites':err_gsites , 'bad_sites':err_bsites}
            else:
                err_summary_pertask[tsk][err]['sum'] += sum(errors[err_][site] for site in errors[err_] )
                err_summary_pertask[tsk][err]['good_sites'].extend( err_gsites )
                err_summary_pertask[tsk][err]['bad_sites'].extend( err_bsites )

    if -2 in err_summary.keys() and err_summary[-2]['sum'] == 0:
        err_summary[-2]['sum'] = 1
    total_errs = sum( [ err_summary[err]['sum'] for err in err_summary ] )
    for err in err_summary:
        err_summary[err]['good_sites'] = list( set( err_summary[err]['good_sites'] ) )
        err_summary[err]['bad_sites'] = list( set( err_summary[err]['bad_sites'] ) )
        err_summary[err]['ratio'] = float(err_summary[err]['sum'])/total_errs
    for tsk,errs in err_summary_pertask.items():
        total_errs_tsk = sum( [ err_summary_pertask[tsk][err1]['sum'] for err1 in err_summary_pertask[tsk] ] )
        if total_errs_tsk == 0:
            total_errs_tsk = 1
        for err in errs:
            errs[err]['good_sites'] = list(set( errs[err]['good_sites'] ) )
            errs[err]['bad_sites'] = list(set( errs[err]['bad_sites'] ) )
            errs[err]['ratio'] = float( errs[err]['sum']/total_errs_tsk )


    #from now on, the phase I logic is followed
    err_summary_sorted = sorted( [(err,err_summary[err]) for err in err_summary ] , key= lambda x: x[1]['ratio'] , reverse=True)
    main_err = err_summary_sorted[0][0] if len(err_summary_sorted) else -10
    main_err_ratio = err_summary_sorted[0][1]['ratio'] if len(err_summary_sorted) else 0

    all_bad_sites = []
    for err in err_summary:
        all_bad_sites.extend( err_summary[err]['bad_sites'] )
    all_bad_sites = list( set( all_bad_sites ) )

    ret['additional_info']['errors'] = OrderedDict( err_summary_sorted )
    tsksWithXROOTD = 0
    for tsk in sites_summary:
        if len(sites_summary[tsk]['good_sites']) == 0:
            ret['parameters']['Action'] = '-'
            ret['parameters']['Reasons'].append( 'task {0} has empty sitelist'.format( tsk ) )

    if ret['parameters']['Action'] != '-':
        if main_err_ratio > 0.3:
            ret['additional_info']['main_err'] = main_err
            bad_sites = err_summary[main_err]['bad_sites']
            if main_err in [-1,-2]:
                if len( bad_sites ) == 0:
                    ret['parameters']['Reasons'].append( "unknown error, site is backup" )
                    ret['parameters']['Action'] = 'acdc'
                else:
                    ret['parameters']['Reasons'].append( 'unknown error, site(s) not backup yet {0}'.format( bad_sites ) )
                    ret['parameters']['Action'] = 'automatic-wait'
            if main_err in [99303, 99304]:
                ret['parameters']['Reasons'].append( "condor issue" )
                ret['parameters']['Action'] = "acdc"
            elif main_err in [99400, 99401]:
                ret['parameters']['Reasons'].append("condor issue" )
                ret['parameters']['Action'] = "acdc"
            elif main_err in [50513]:
                ret['parameters']['Reasons'].append( "scram/site issue" )
                ret['parameters']['Action'] = "acdc"
            elif main_err in [71305 , 71304]:
                bad_sites = all_bad_sites
                # if len( bad_sites ) == 1:
                #     ret['parameters']['Action'] = "acdc"
                #     ret['additional_info']["xrootd"] = "yes"
                #     ret['parameters']['Reasons'].append( "Timeout due to priority inversion, status is pending, one site in drain {0}".format( bad_sites ) )
                if len( bad_sites ) == 0:
                    ret['parameters']['Action'] = "acdc"
                    ret['parameters']['Reasons'].append( "Timeout due to priority inversion, status is pending, no site in drain" )
                else:
                    ret['parameters']['Action'] = "automatic-wait"
                    ret['parameters']['Reasons'].append("Timeout due to priority inversion, status is pending, multiple sites in drain {0}".format( bad_sites ) )
            elif main_err in [99109]:
                if len(bad_sites) == 0:
                    ret['parameters']['Action'] = 'acdc'
                    ret['parameters']['Reasons'].append( 'stage out error, site is not in drain' )
                else:
                    ret['parameters']['Action'] = 'automatic-wait'
                    ret['parameters']['Reasons'].append( 'stage out error, site is not backup' )
            elif main_err in [71104, 71105 , 71103]:
                if len(bad_sites) == 0:
                    ret['parameters']['Action'] = 'acdc'
                    ret['parameters']['Reasons'].append( 'site put into the drain, site is backup' )
                else:
                    ret['parameters']['Action'] = 'automatic-wait'
                    ret['parameters']['Reasons'].append( 'site put into the drain, site is not backup' )

    #phase II
    if ret['parameters']['Action'] not in ['acdc' , 'automatic-wait']:
        wfparams = wfinfo.get_workflow_parameters()
        primaryInputDS = ''
        if 'Task1' in wfparams:
            if 'InputDataset' in wfparams['Task1']:
                primaryInputDS = wfparams['Task1']['InputDataset']
        elif 'InputDataset' in wfparams:
            primaryInputDS = wfparams['InputDataset']
        ret['additional_info']['InputDataset'] = primaryInputDS

        for tsk,err_summary in err_summary_pertask.items():
            err_summary_sorted = sorted( [(err,err_summary[err]) for err in err_summary ] , key= lambda x: x[1]['ratio'] , reverse=True)
            main_err = err_summary_sorted[0][0] if len(err_summary_sorted) else -10
            main_err_ratio = err_summary_sorted[0][1]['ratio'] if len(err_summary_sorted) else 0

  
            if main_err_ratio<0.3 :
                continue

            all_bad_sites = []
            all_good_sites = []
            for err in err_summary:
                all_bad_sites.extend( err_summary[err]['bad_sites'] )
                all_good_sites.extend( err_summary[err]['bad_sites'] )
            all_bad_sites = list( set( all_bad_sites ) )
            all_good_sites = list( set( all_good_sites) )
            ret['parameters']['Parameters'][tsk] = {'xrootd':'disabled' , 'sites':all_good_sites , 'bad_sites':all_bad_sites }

            if primaryInputDS == '':
                if main_err in [8020 , 8021 , 8028 ] :
                    if len( all_bad_sites ) == 0:
                        ret['parameters']['Action'] = 'acdc'
                        ret['parameters']['Reasons'].append( 'main err 8020,8021,8028 for task {0} and no site in drain'.format( tsk ) )
                        
                        #ret['parameters']['Parameters'][tsk] = {'Action':'acdc'}
                # if main_err not in [ 8020 , 8021 , 8028 ]:
                #     if len( all_bad_sites ) != 0 and len(all_good_sites)!=0:
                #         ret['parameters']['Parameters'][tsk]['xrootd'] = 'enabled'
                #         #ret['parameters']['Parameters'][tsk]['secondary'] = 'enabled'
                #         tsksWithXROOTD += 1


    #if len(all_bad_sites) == 1 and ret['parameters']['Action'] == 'acdc':
    ret['additional_info']['xrootd'] = '({0}/{1})'.format( tsksWithXROOTD , len(sites_summary) )

    acdc_re = re.compile("(.*)_ACDC(.)_.*")
    acdc_re_parsed = acdc_re.match( wfinfo.workflow )
    if acdc_re_parsed:
        username = acdc_re_parsed.groups()[0]
        acdc_index = acdc_re_parsed.groups()[1]
        ret['parameters']['Action'] = ''
        ret['parameters']['Reasons'] += "no action recommended as the wf was acdc'ed before by {0}".format( username )


    #if ret['parameters']['Action'] != 'acdc' :
    #    return ret2

    return ret



def view(request):
    wf = request.GET.get('wf' , None )
    xrootd=request.GET.get('xrootd' , False)
    if not wf:
        return HttpResponseBadRequest("please set the workflow by passing the wf argument")

    wfinfo =  workflowinfo.WorkflowInfo( wf )
    pred = ParseErrors( wfinfo , xrootd )

    return JsonResponse( pred , safe=False )

def summary(request):
    wf = request.GET.get('wf' , None )
    if not wf:
        return HttpResponseBadRequest("please set the workflow by passing the wf argument")

    wfinfo = workflowinfo.WorkflowInfo( wf )
    pred = ParseErrors( wfinfo , True )
    root = Element('workflow')
    SubElement(root , 'name').text = wf
    errs = SubElement( root , 'errors' )
    for key,details in pred['additional_info']['errors'].items():
        err_el = SubElement( errs , 'error')
        err_id = SubElement( err_el , 'id' )
        err_id.text = str( key )
        err_gs = SubElement( err_el , 'good_sites' )
        for st in details['good_sites']:
            site_el = SubElement( err_gs , 'site')
            SubElement( site_el , 'name' ).text = st
        err_bs = SubElement( err_el , 'bad_sites' )
        for st in details['bad_sites']:
            site_el = SubElement( err_bs , 'site')
            SubElement( site_el , 'name' ).text = st
        err_sum = SubElement( err_el , 'count')
        err_sum.text = str( details['sum'] )
        err_ratio = SubElement( err_el , 'ratio' )
        err_ratio.text = '{0:.1f}%'.format( details['ratio']*100 )
    

    tsks = SubElement( root , 'tasks' )
    for tsk,details in pred['parameters']['Parameters'].items():
        task = SubElement( tsks , 'task')
        SubElement(task , 'name').text = tsk
        SubElement(task , 'xrootd').text = details['xrootd']
        tsk_gs = SubElement( task , 'good_sites' )
        for st in details['sites']:
            site_el = SubElement( tsk_gs , 'site')
            SubElement( site_el , 'name' ).text = st
        tsk_bs = SubElement( task , 'bad_sites' )
        for st in details['bad_sites']:
            site_el = SubElement( tsk_bs , 'site')
            SubElement( site_el , 'name' ).text = st
            

    summary = SubElement(root , 'summary' )
    SubElement( summary , 'action' ).text = pred['parameters']['Action']
    SubElement( summary , 'main_error' ).text = str( pred['additional_info'].get( 'main_err' , 'no-err' ) )
    SubElement( summary , 'description' ).text = ', '.join( pred['parameters'].get( 'Reasons' , "" ) )
    SubElement( summary , 'xrootd' ).text = pred['additional_info'].get( 'xrootd' , "" )
    SubElement( summary , 'nBadSites' ).text = str( pred['additional_info'].get( 'nBadSites' , -1 ) )

    tree = ElementTree(root)
    xml_string = tostring( root , encoding='utf8', method='xml')
    xml_dom = minidom.parseString( xml_string )
    pi = xml_dom.createProcessingInstruction('xml-stylesheet',
                                             'type="text/xsl" href="/getFile/?f=wf_details.xsl"')
    xml_dom.insertBefore(pi, xml_dom.firstChild)
    response =  HttpResponse( xml_dom.toprettyxml(),
                              content_type= 'text/xml')
    return response


def getCachedWorkflows():
    error_file_re = re.compile( 'workflowinfo_(.*)_errors.cache.json' )
    for wf_f in os.listdir( '/tmp/workflowinfo/' ):
        wf_f_err = error_file_re.match( wf_f )
        if wf_f_err:
            yield wf_f_err.groups()[0]

import requests
def getAssistantManualWorkflows():
    r = requests.get("https://vocms0113.cern.ch/assistance_manual" , verify=False)
    for l in r.json():
        yield l

def getAssistantManualRecoveredWorkflows():
    r = requests.get("https://vocms0113.cern.ch/assistance_manual_recovered" , verify=False)
    for l in r.json():
        yield l


def listExistingWFs(request):
    root = Element('files')
    fromcache = request.GET.get('cache' , False )
    assistanceManualRecovered = request.GET.get('recovered' , False )

    if fromcache:
        wflist = getCachedWorkflows
    else:
        if assistanceManualRecovered:
            wflist = getAssistantManualRecoveredWorkflows
        else:
            wflist = getAssistantManualWorkflows

    xrootd=request.GET.get('xrootd' , False)
    for name in wflist():
        wfinfo = workflowinfo.WorkflowInfo( name )
        errors = ParseErrors( wfinfo , xrootd )
        if errors['parameters']['Action'] != 'acdc':
            continue
        wf_el = SubElement( root, 'wf')
        wf_name = SubElement( wf_el , 'name')
        wf_name.text = name
        lnk_el = SubElement( wf_el, 'lnk' )
        lnk_el.text = 'http://aieh.cern.ch:8052/static_action_details/?wf=' + name
        wf_action = SubElement( wf_el , 'action' )
        wf_action.text = errors['parameters']['Action']
        wf_action_description = SubElement( wf_el , 'action_description')
        wf_action_description.text = ','.join( errors['parameters']['Reasons'] )

        wf_main_err = SubElement(wf_el , 'main_err')
        wf_main_err.text = str( errors['additional_info'].get( 'main_err' , 'no-err' ) )
        wf_js = SubElement(wf_el , 'json')
        wf_js.text = 'http://aieh.cern.ch:8052/static_action/?wf=' + name 
        SubElement( wf_el , 'console' ).text = 'https://vocms0113.cern.ch/seeworkflow2/?workflow=' + name
        SubElement( wf_el , 'unified' ).text = 'https://cms-unified.web.cern.ch/cms-unified/report/' + name
        SubElement( wf_el , 'wmstat' ).text = 'https://cmsweb.cern.ch/wmstatsserver/data/jobdetail/' + name
        SubElement( wf_el , 'xrootd' ).text = errors['additional_info'].get( 'xrootd' , '' ) 
        SubElement( wf_el , 'nBadSites' ).text = str( errors['additional_info'].get( 'nBadSites' , -1 ) )

    tree = ElementTree(root)
    xml_string = tostring( root , encoding='utf8', method='xml')
    xml_dom = minidom.parseString( xml_string )
    pi = xml_dom.createProcessingInstruction('xml-stylesheet',
                                             'type="text/xsl" href="/getFile/?f=wf_list.xsl"')
    xml_dom.insertBefore(pi, xml_dom.firstChild)
    response =  HttpResponse( xml_dom.toprettyxml(), #toprettyxml() ,
                              content_type= 'text/xml')
    return response
