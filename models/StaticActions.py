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

def wfs_dec21():
    ret = []
    with open('/home/aieh/AIErrorHandling/models/wfs_dec21') as f:
        for l in f:
            ret.append(l[:-1])
    return ret

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

    #ret['additional_info']['errs50660'] = err_summary_pertask
    if wfinfo.workflow in wfs_dec21():
        ret['additional_info']['errs50660'] = err_summary_pertask
        ret['additional_info']['in50660list'] = 1
        tsksw50660 = {}
        for tsk_,errs in err_summary_pertask.items():
            tsk = tsk_[ len(wfinfo.workflow)+2: ]
            if 50660 in errs:
                has50660 = True
                tsksw50660[tsk] = ret['parameters']['Parameters'][tsk]
                tsksw50660[tsk]['memory'] = 3750
                tsksw50660[tsk]['xrootd'] = 'enabled'
            else:
                tsksw50660[tsk] = ret['parameters']['Parameters'][tsk]
        if len(tsksw50660) > 0:
            return {'workflow': wfinfo.workflow,
                    'parameters': {
                        'Action': 'acdc',
                        'Reasons': ['AIEH', '50660, december 2021 accident'],
                        'ACDCs': [],
                        'Parameters': tsksw50660
                    },
                    'additional_info':{
                        'errors': {},
                        'nBadSites': nBadSites( wfinfo ),
                        'in50660list' : 1,
                        'main_err' : 50660
                        #'errs50660' : err_summary_pertask
                    } }
    else:
        ret['additional_info']['in50660list'] = 0
            
    if ret['additional_info']['nBadSites'] != 0 and not include_xrootds:
        return ret



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
    main_errs_pertask = {}
    for tsk,err_summary__ in err_summary_pertask.items():
        err_summary_sorted_ = sorted( [(err,err_summary__[err]) for err in err_summary__ ] , key= lambda x: x[1]['ratio'] , reverse=True)
        main_errs_pertask[ tsk ] = err_summary_sorted_[0] if len(err_summary_sorted_) else None

    #from now on, the phase I logic is followed
    err_summary_sorted = sorted( [(err,err_summary[err]) for err in err_summary ] , key= lambda x: x[1]['ratio'] , reverse=True)
    main_err = err_summary_sorted[0][0] if len(err_summary_sorted) else -10
    main_err_ratio = err_summary_sorted[0][1]['ratio'] if len(err_summary_sorted) else 0 #(-100 , 1)

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

    accident_22Nov2021 = False
    main_err_ratio_threshhold = 0.0 if accident_22Nov2021 else 0.3

    if ret['parameters']['Action'] != '-':
        if main_err_ratio > main_err_ratio_threshhold:
            ret['additional_info']['main_err'] = main_err
            bad_sites = err_summary[main_err]['bad_sites']
            if main_err in [-1,-2]:
                if len( bad_sites ) == 0:
                    ret['parameters']['Reasons'].append( "unknown error, site is backup" )
                    ret['parameters']['Action'] = 'acdc'                  
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
            # elif main_err in [99109]:
            #     if len(bad_sites) == 0:
            #         if any( [ste_ in err_summary[main_err]['good_sites'] for ste_ in ['T2_CH_CERN', 'T2_CH_CERN_HLT'] ] ):
            #             ret['parameters']['Action'] = 'acdc'
            #             ret['parameters']['Reasons'].append( 'stage out error, site is not in drain' )
            #     else:
            #         ret['parameters']['Action'] = 'automatic-wait'
            #         ret['parameters']['Reasons'].append( 'stage out error, site is not backup' )

    if main_err_ratio > 0.3:
        ret['additional_info']['main_err'] = main_err
        bad_sites = err_summary[main_err]['bad_sites']

        if main_err in [71104, 71105 , 71103 , 71102 , 71302]:
            if len(bad_sites) == 0:
                ret['parameters']['Action'] = 'acdc'
                ret['parameters']['Reasons'].append( 'site put into the drain, site is backup' )
            else:
                wfparams = wfinfo.get_workflow_parameters()
                tskInfo = None
                doAcdc = False
                anyTaskWithMCPU = False
                ret['parameters']['Reasons'].append( 'main error is {0} and {1} sites are in drain, trying to check other t1 sites {2}'.format( main_err , len(bad_sites) , len(sites_summary) ) )
                for tskId in range( len(sites_summary)+2 ):
                    tag = "Task{0}".format( tskId)
                    tskNameTag = "TaskName"
                    if tag not in wfparams:
                        tag = "Step{0}".format( tskId)
                        tskNameTag = "Step"
                        if tag not in wfparams:
                            ret['parameters']['Reasons'].append( 'Task/Step{0} not found in wfinfo'.format(tskId) )
                            continue
                    tskInfo_ = wfparams[ tag ]
                    tskName_ = tskInfo_[ "{0}Name".format(tskNameTag) ] 
                    GoodSites = []
                    AllSites = []
                    fullTaskName = ""
                    for tsk in sites_summary:
                        if tsk.endswith( tskName_ ):
                            GoodSites = sites_summary[tsk]['good_sites']
                            AllSites = sites_summary[tsk]['bad_sites']
                            

                    for tsk in err_summary_pertask:
                        if tsk.endswith( tskName_ ):
                            fullTaskName = tsk
                            ret['parameters']['Reasons'].append( 'tskSummary:{0}'.format(str(main_errs_pertask[tsk]) ))

                    if fullTaskName in err_summary_pertask:
                        mainTskErr = main_errs_pertask[fullTaskName][0]
                        if mainTskErr not in [71104, 71105 , 71103 , 71102 , 71302]:
                            ret['parameters']['Reasons'].append( 'task {0} skipped, main error is'.format(tskName_, mainTskErr) )
                            continue
                    else:
                        ret['parameters']['Reasons'].append( 'a)task {0} was not found in the taks list'.format(fullTaskName) )
                        continue
                    if fullTaskName == "":
                        ret['parameters']['Reasons'].append( 'b)task {0} was not found in the taks list'.format(tskName_) )
                        continue

                    ret['parameters']['Reasons'].append( 'task {0} is found to be the taks with this main error'.format(tskName_) )

                    if 'MCPileup' in tskInfo_:
                        secondaryInput = tskInfo_['MCPileup']
                        tskInfo = fullTaskName
                        anyTaskWithMCPU = True
                        if "minbias" in secondaryInput.lower():
                            #doAcdc = False
                            ret['parameters']['Reasons'].append( 'task {0} MCPileUp is minbias {1}'.format(tskName_ , secondaryInput) )
                            continue
                        #else: #len(GoodSites) == 0:


                    nUSSites = 0
                    for s in AllSites:
                        if "_US_" in s:
                            nUSSites += 1
                    sites_in_continent = ['T1_US_FNAL'] if nUSSites > 0 else ['T1_IT_CNAF' , 'T1_RU_JINR' , 'T1_DE_KIT' , 'T1_UK_RAL' , 'T1_ES_PIC' , 'T1_FR_CCIN2P3' ]
                    sites_in_continent += GoodSites
                    good_sites_in_continent = [trimSiteName(site) for site in sites_in_continent if sitereadiness.site_drain_status( trimSiteName(site) ) == 'enabled' ]
                    good_sites_in_continent = list( set(good_sites_in_continent) )

                    if len(good_sites_in_continent)>0:
                        doAcdc = True
                        #ret['parameters']['Reasons'].append( 'task {0} with non minbias MCPileUp can run on new sites with xrootd option = on / {1}'.format(tskName_ , ','.join([a for a in ret['parameters']['Parameters']])) )
                        ret['parameters']['Parameters'][tskName_]['xrootd'] = 'yes'
                        ret['parameters']['Parameters'][tskName_]['sites'] = good_sites_in_continent
                        ret['parameters']['Reasons'].append( 'task {0} with non minbias MCPileUp can run on new sites with xrootd option = on'.format(tskName_) )
                        ret['parameters']['Reasons'].append('while list: {0}'.format(','.join(good_sites_in_continent) ) )
                    else:
                        doAcdc = False
                        ret['parameters']['Reasons'].append( 'task {0} with non minbias MCPileUp has to run in {1}, but there is no T1 available there'.format(tskName_ , 'US' if nUSSites>0 else 'EU') )
                        #else:
                        #    ret['parameters']['Reasons'].append( 'task {0} with nonminbias MCPileUp has already some good site in the list'.format(tskName_) )
                        #else:
                        #ret['parameters']['Reasons'].append( 'task {0} has no MCPileUp'.format(tskName_) )

                if doAcdc:
                    ret['parameters']['Action'] = 'acdc'
                else:
                    ret['parameters']['Action'] = 'automatic-wait'

                        

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
                if main_err*100000 in [8020 , 8021 , 8028 ] : #these error codes are out since 2nd of Feb, requested by Hassan : https://mattermost.web.cern.ch/tools-and-integ/pl/9ebs1wxgctfs3kbx6f6qgzwtur
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

    xrootd=request.GET.get('xrootd' , False)
    pred = ParseErrors( wfinfo , xrootd )
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
    r = requests.get("https://wfrecovery.cern.ch/assistance_manual" , verify=False)
    for l in r.json():
        yield l

def getAssistantManualRecoveredWorkflows():
    r = requests.get("https://wfrecovery.cern.ch/assistance_manual_recovered" , verify=False)
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
        lnk_el.text = 'http://wfrecovery.cern.ch:80/static_action_details/?wf=' + name
        wf_action = SubElement( wf_el , 'action' )
        wf_action.text = errors['parameters']['Action']
        wf_action_description = SubElement( wf_el , 'action_description')
        wf_action_description.text = ','.join( errors['parameters']['Reasons'] )

        wf_main_err = SubElement(wf_el , 'main_err')
        wf_main_err.text = str( errors['additional_info'].get( 'main_err' , 'no-err' ) )
        wf_js = SubElement(wf_el , 'json')
        wf_js.text = 'http://wfrecovery.cern.ch:80/static_action/?wf=' + name 
        SubElement( wf_el , 'console' ).text = 'https://wfrecovery.cern.ch/seeworkflow2/?workflow=' + name
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
