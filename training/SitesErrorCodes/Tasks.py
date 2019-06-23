"""
This module contains classes to read contents of tasks and actions from the 'actionhistory' json file, produced by old-console
Here is the list of classes : 
1. Task : it gets one entry from json file and interpret it. It converts errors-sites to numpy array
2. Task.Action : details of the action taken by the operator 
3. Tasks : list of tasks. It includes lists of all sites, all errors and all actions in the json file. 
One can ask Tasks class to categorize tasks based on the site-tier instead of site-name. It is also capable of converting the information to 'acdc/non-acdc' binary decision.

:author: Hamed Bakhshiansohi <hbakhshi@cern.ch>
"""
import workflowwebtools as wwt
from pandas import DataFrame

import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from AIErrorHandling.training.SitesErrorCodes.ErrorCodes import ErrCategory , ErrorRange,ErrorCodes, ErrDescription, Categories
from tensorflow.keras.utils import to_categorical

class Task:
    """
    class to capsulate information about each task and the action took by the operator
    """
    class Action:
        """
        class to store Action taken by operator
        """
        def __init__(self,parameters, all_actions):
            """
            :param dict parameters: information from json file for the action
            :param list all_actions: list of the name of all the actions, so the index of the current action can be found
            """
            self.sites = parameters.get("sites")
            self.splitting  = parameters.get('splitting')
            self.xrootd  = parameters.get('xrootd')
            self.other  = parameters.get('other')
            self.memory  = parameters.get('memory')
            self.action  = str( parameters.get('action') )
            self.cores  = parameters.get('cores')
            self.secondary  = parameters.get('secondary')

            self.action_code = -1
            #self.SetCode( all_actions )

        def ShortDesciption(self):
            if hasattr( self , 'ShortDesc' ):
                return self.ShortDesc
            ret = [self.action , self.splitting , 0 if self.xrootd=='disabled' else 1  , 1 if self.sites else 0 , int( int(self.memory)/10000) if self.memory else -1 , 1 if self.cores else 0]
            self.ShortDesc = "_".join([ str(a) for a in ret])
            return self.ShortDesc
            
        def SetCode(self , all_actions , non_acdc_index = 0):
            """
            set the action code. here the action code is set to the corresponding index of non-acdc if the action is not found in the list of all_actions
            :param list all_actions: list of all actions. 
            :param int non_acdc_index: by default it assumes that 'non-acdc' is the first element in the list, but it can be modified by this parameter
            """
            try :
                self.action_code = all_actions.index( self.action )
            except ValueError as err :
                if all_actions[ non_acdc_index ] == "non-acdc" :
                    self.action_code = non_acdc_index
                else :
                    raise err

        def SetCodeByShortDesc(self,newcode):
            self.action_code = newcode
        def code(self):
            """
            return the action code
            """
            return self.action_code


    def normalize_errors(self, errs_goodsites , errs_badsites , all_errors = None , all_sites = None , TiersOnly=False):
        """
        converting the good/bad sites/errors matrix to numpy array. it creates an array with dim #errors X #sites X 2
        :param dict errs_goodsites: map of good sites and number of failed jobs in each site
        :param dict errs_goodsites: map of bad sites and number of failed jobs in each site
        :param list all_errors: the list of all errors. if is not given, the list from parent tasks object is used
        :param list all_sites: the list of all sites. if is not given, the list from parent tasks object is used
        :param bool TiersOnly: if it is true, site names are ignored and they are just categorized as T0, T1, T2, T3 and Others(just in case)
        """
        if not all_errors:
            all_errors = self.tasks.all_errors
        if not all_sites:
            all_sites = self.tasks.all_sites

        if TiersOnly :
            self.error_tensor = np.zeros( (len(all_errors) , 5 ,2 ) )
        else :
            self.error_tensor = np.zeros( (len(all_errors) , len(all_sites) ,2 ) )
        for good_bad,errs in {0:errs_goodsites,1:errs_badsites}.items() :       
            for err in errs :
                errcode = all_errors.index( int(err) )
                sites = self.error_tensor[errcode]
                for site in errs[err]:
                    if TiersOnly:
                        try:
                            site_index = int( site[1] )
                        except ValueError as errorrrr:
                            site_index = 4
                    else:
                        site_index = all_sites.index( site )
                    count = errs[err][site]
                    sites[site_index][good_bad] +=  count

    def GetActionRank(self):
        return self.tasks.AllTakenActions.index( self.action.ShortDesciption() )

    def GetErrorInfo(self):
        if hasattr(self,'ErrorInfo'):
            return self.ErrorInfo
        
        DominantErrorCodeInGoodSites = self.GetInfo( 'DominantErrorCodeInGoodSites' )
        
        DominantErrorCodeInBadSites = self.GetInfo( 'DominantErrorCodeInBadSites' )
        self.ErrorInfo = ErrDescription(DominantErrorCodeInGoodSites) + "_" + ErrDescription(DominantErrorCodeInBadSites)
        #"_".join( str(i) for i in [ ErrCategory( DominantErrorCodeInBadSites )  , ErrorRange( DominantErrorCodeInBadSites ) ,
        #                                         ErrCategory( DominantErrorCodeInGoodSites ) , ErrorRange( DominantErrorCodeInGoodSites ) ] )
        return self.ErrorInfo
        
    def GetErrorInfoRank(self):
        return self.tasks.AllErrorInfos.index( self.GetErrorInfo() )
    
    def __init__(self , tsk , name , tasks , site_statuses=['good_sites' , 'bad_sites'] ):
        """
        initialize the Task object.
        :param dict tsk: the dictionary from json file, which includes "parameters" and "errors" keys
        :param str name: the name of the task
        :param Tasks tasks: the parent tasks object
        """
        self.Name = name
        self.tasks = tasks
        #self.tttsssskkk = tsk
        
        if "parameters" in tsk.keys():
            params = tsk["parameters"]
            self.action = self.Action( params , tasks.all_actions )
        else:
            self.action = None

        if "errors" in tsk.keys() :
            errors = tsk["errors"]
            if 'good_sites' in site_statuses :
                goodsts = errors["good_sites"]
            else:
                goodsts = {}
            if 'bad_sites' in site_statuses :
                badsts = errors["bad_sites"]
            else:
                badsts = {}
            self.normalize_errors( goodsts , badsts , TiersOnly=tasks.TiersOnly )



    def GetNErrors(self , errcode , site = 'all' ):
        try:
            err_index = self.tasks.all_errors.index( errcode )
        except ValueError:
            return 0
        if site in ['good' , 'bad' , 'all' ]:
            if site != 'all' :
                site_index = ['good' , 'bad'].index( site )
                return self.Get2DArrayOfErrors(overridetiersonly=True)[ err_index ][site_index]
            else :
                return self.Get2DArrayOfErrors(overridetiersonly=True)[ err_index ][0]+self.Get2DArrayOfErrors(overridetiersonly=True)[ err_index ][1]
        else:
            try:
                site_index = self.tasks.all_sites.index(site)
                row = self.error_tensor[err_index][site_index]
                return row[0]+row[1]
            except ValueError:
                return 0

        
                
    def Get2DArrayOfErrors(self , force = True , overridetiersonly =False):
        """
        Converts the 3D numpy array to 2d array of errors by summing over the sites.
        :param bool force: force to calculate it, even it has been already calculated
        :ret numpy.array: a 2D numpy array where the first dimention indicates the index of the error and the second dimention has only size of two : bad_sites/good_sites
        """
        objname = 'sum_over_sites'
        if self.tasks.TiersOnly and not overridetiersonly:
            objname = 'sum_over_sites_tiersonly'
        
        if not hasattr( self , objname ) or force:
            if self.tasks.TiersOnly and not overridetiersonly:
                self.sum_over_sites_tiersonly = np.concatenate( (self.error_tensor[:,:,0] , self.error_tensor[:,:,1] ) , 1 )
            else:
                self.sum_over_sites = np.sum( self.error_tensor , axis=1 )

        return getattr( self , objname )

    
    
    def GetInfo(self , what = None , labelsOnly = False):
        """
        Converts the task to a one dimention list. 
        :param bool labelsOnly: if it is true, only the header which includes the name of the fields is returned
        """
        lbls = ['tsk' , 'action' , 'nErrors' , 'nErrorsInGoodSites' , 'nErrorsInBadSites' , 'DominantErrorCodeInGoodSites' , 'DominantErrorCodeInBadSites' , 'DominantBadSite' , 'DominantGoodSite']
        if labelsOnly:
            return lbls
        indx = -1
        if what:
            indx = lbls.index( what )
        info = [ self.Name ,
                 self.action.code() ,
                 np.sum( self.error_tensor ) ]
    
        info.extend( np.sum( self.error_tensor , axis=(0,1) )  )

        dominantErrors = np.argmax( self.Get2DArrayOfErrors(overridetiersonly=True) , axis=0 )
        info.extend([self.tasks.all_errors[i] for i in dominantErrors])

        sum_over_error = np.sum( self.error_tensor , axis=0 )
        dominantSite = np.argmax( sum_over_error , axis=0 )
        info.extend(dominantSite)

        if indx == -1:
            return info
        else:
            return info[indx]


class Tasks :
    """
    a class to read 'actionhistory' json file and convert it to numpy array
    it involves two loops over the actions. in the first loop it extracts the name of all the sites and actions and also the error codes.
    in the second loop, for each entry a Task item is created and stored
    """
    def __init__(self, _file , TiersOnly=False, all_sites_=[] , all_errors_=[] , all_actions_=[] , site_statuses = ['good_sites' , 'bad_sites'] , name = "tasks" , countclasses = False , classes = {} ):
        """
        initialize an instance of Taks
        :param str _file: the full path of the actionhistory json file
        :param bool binary: if true, converts actions to acdc/non-acdc
        :param bool TiersOnly: if true, only the tier index of the site is stored instead of the full name
        :param all_actions, all_errors, all_actions: to be able to add additional values to the list
        """

        binary = len(classes)==2
        
        self.Name = name
        self.TiersOnly = TiersOnly
        self.IsBinary = binary
        self.fIn = open(_file)
        self.js = json.load( self.fIn )

        self.all_sites = [] #all_sites_
        self.all_errors = [] #all_errors_
        self.all_actions = [] #all_actions_

        #print(self.all_sites)
        self.SiteStatuses = site_statuses
        self.FillSiteErrors(site_statuses)
        
        #if binary :
        self.all_actions = list( classes.keys() ) #["non-acdc", "acdc"]
        
        self.AllData = []

        self.SummaryTakenActions = {}
        self.SummaryDominantErrors = {}
        for tsk in self.js :
            self.AllData.append( Task( self.js[tsk] , tsk , self , site_statuses ) )
            shortdesc = self.AllData[-1].action.ShortDesciption()
            if shortdesc in self.SummaryTakenActions:
                self.SummaryTakenActions[ shortdesc ] += 1
            else:
                self.SummaryTakenActions[ shortdesc ] = 1
                
            errinfo = self.AllData[-1].GetErrorInfo()
            if errinfo in self.SummaryDominantErrors :
                self.SummaryDominantErrors[ errinfo ] += 1
            else:
                self.SummaryDominantErrors[ errinfo ] = 1
                
        import operator
        self.AllTakenActions = [ a for a,b in sorted( self.SummaryTakenActions.items() , key=operator.itemgetter(1) , reverse=True ) ]
        self.AllErrorInfos = [ a for a,b in sorted( self.SummaryDominantErrors.items() , key=operator.itemgetter(1) , reverse=True ) ]        

        all_classes = []
        the_one_with_minus1 = -1
        for cls in classes:
            all_classes.extend( classes[cls] )
            if -1 in classes[cls]:
                the_one_with_minus1 = cls
        if the_one_with_minus1 != -1 :
            classes[ the_one_with_minus1 ].remove( -1 )
            for indx in range(0,len(self.AllTakenActions)):
                if not indx in all_classes :
                    classes[ the_one_with_minus1 ].append( indx )

        
        
        from collections import defaultdict
        self.ClassCounts = defaultdict(int) if countclasses else None
        for tsk in self.AllData:
            class_index = -1
            tsk_rank = tsk.GetActionRank()
            for cls in classes:
                if tsk_rank in classes[cls]:
                    class_index = self.all_actions.index( cls )
            tsk.action.SetCodeByShortDesc(class_index)
            if countclasses:
                self.ClassCounts[tsk.action.code()] += 1
        if self.ClassCounts!=None:
            MIN = min( self.ClassCounts.values() )
            for idx in self.ClassCounts :
                self.ClassCounts[idx] = MIN/self.ClassCounts[idx]

        self.Classes = classes
        for cls in classes :
            print(cls , self.ClassCounts[ self.all_actions.index(cls) ]   , [ self.AllTakenActions[i] for i in classes[cls] ] )
                
                
        self.ErrorsGoodBadSites = np.array( [ tsk.Get2DArrayOfErrors() for tsk in self.AllData ] )
        self.AllActions = np.array( [tsk.action.code() for tsk in self.AllData ] )
        if not self.TiersOnly :
            self.df = DataFrame(data=[tsk.GetInfo() for tsk in self.AllData] , columns=self.AllData[0].GetInfo(labelsOnly=True))
        
    def GetShuffledDS(self , n ):
        p = np.random.permutation( len(self.AllData)  )
        return self.ErrorsGoodBadSites[ p[:n] ], self.AllActions[p[:n] ]
        
    def GetTrainTestDS(self , train_ratio , shuffle=False , val_ratio = 0):
        """
        convert the information to train/test
        :param float train_ratio: number between 0 and 1, the fraction to go for the training
        :ret: train_x, train_y, test_x , test_y
        """
        if shuffle:
            self.ErrorsGoodBadSites , self.AllActions = self.GetShuffledDS( len(self.AllData) )
        n = int(train_ratio*len(self.AllData))
        n1 = int(n*val_ratio)
        X_val , y_val, X_train, y_train, X_test, y_test = self.ErrorsGoodBadSites[:n1] , self.AllActions[:n1] , self.ErrorsGoodBadSites[n1:n] , self.AllActions[n1:n] , self.ErrorsGoodBadSites[n:] , self.AllActions[n:]

        X_val = X_val.astype('int16')
        X_train = X_train.astype('int16')
        X_test = X_test.astype('int16')

        if self.IsBinary :
            Y_val = y_val.astype('int16')
            Y_train = y_train.astype('int16')
            Y_test = y_test.astype('int16')
        else:
            Y_val = to_categorical( y_val , len(self.all_actions) , 'int8')
            Y_train = to_categorical(y_train, len(self.all_actions) , 'int8')
            Y_test = to_categorical(y_test, len(self.all_actions) , 'int8')

        return X_val , Y_val, X_train, Y_train, X_test, Y_test
        
    def FillSiteErrors(self , site_statuses , Print=False ):
        """
        For the first loop and fill the lists of errors, sites and actions
        :param bool Print: do printing after it has been done
        """
        #print(site_statuses)
        for tsk in self.js :
            errors = self.js[tsk]["errors"]
            for site_status in site_statuses:
                sites = errors[site_status]
                for err in sites :
                    if int(err) not in self.all_errors:
                        self.all_errors.append(int(err))
                    for site in sites[err]:
                        if site not in self.all_sites :
                            self.all_sites.append( site )
            action = self.js[tsk]['parameters']['action']
            if action not in self.all_actions :
                self.all_actions.append( str(action) )
        self.all_sites.sort()
        self.all_errors.sort()
        self.all_actions.sort()

        if Print:
            print(self.all_sites)
            print(self.all_errors)
            print(self.all_actions)

    def PlotDominantErrors( self ):
        import ROOT
        good_site_errors = [a.split("_")[0] for a in self.AllErrorInfos]
        bad_site_errors = [a.split("_")[1] for a in self.AllErrorInfos]
        all_errors = set( good_site_errors + bad_site_errors )
        n_all_errors = len( all_errors ) - 1
        
        self.hDominantErrors = ROOT.TH2D("hDominantErrors" , 'hDominantErrors;Good Sites;Bad Sites' , n_all_errors + 1 , 0 , n_all_errors +1 , n_all_errors , 0 , n_all_errors )
        for err_g in Categories :
            if err_g == 'Others':
                continue
            for err_b in Categories :
                if err_b == 'Others':
                    continue
                err = err_g + "_" + err_b
                nnn = 0
                if err in self.AllErrorInfos :
                    nnn = self.SummaryDominantErrors[ err ]
                self.hDominantErrors.Fill( err_g , err_b , nnn )

        return self.hDominantErrors

    def PlotActionVsErrorInfo(self , label ):
        import ROOT
        self.hActionsVsErrorInfo = ROOT.TH2D('hActionsVsErrorInfo' , label + ";Action;DominantError" , len( self.SummaryTakenActions ) , 0 , len(self.SummaryTakenActions ) ,
                                             len( self.SummaryDominantErrors ) , 0 , len(self.SummaryDominantErrors ) )
        for tsk in self.AllData:
            #self.hActionsVsErrorInfo.Fill( tsk.GetActionRank() , tsk.GetErrorInfoRank() )
            self.hActionsVsErrorInfo.Fill( tsk.action.ShortDesciption() , tsk.GetErrorInfo() , 1 )

        return self.hActionsVsErrorInfo

    def PlotErrorInfoForAction(self , action , color , marker , label = None ):
        import ROOT
        if not label:
            label = action
        name = 'hErrorInfoForAction_' + label
        hRet = ROOT.TH1D( name , label + ";DominantErrors"  , len( self.SummaryDominantErrors ) , 0 , len(self.SummaryDominantErrors ) )
        hRet.SetMarkerStyle( marker )
        hRet.SetMarkerColor( color )
        hRet.SetLineColor( color )
        hRet.SetMarkerSize( 1.4 )
        if type(action) is str :
            action = [self.AllTakenActions.index( action )]
        elif type(action) is int:
            action = [action]
        setattr( self , name , hRet )

        for i,err in enumerate(self.AllErrorInfos) :
            hRet.GetXaxis().SetBinLabel( i+1 , err )
        for tsk in self.AllData:
            if tsk.GetActionRank() in action :
                #self.hActionsVsErrorInfo.Fill( tsk.GetActionRank() , tsk.GetErrorInfoRank() )
                hRet.Fill( tsk.GetErrorInfo() , 1 )

        return hRet
    
    
    def PlotNTotalErrs(self , label , sitestats , errs = None ):
        if errs == None:
            errs = self.all_errors
        import ROOT
        sums = np.zeros( len(errs) )
        for tsk in self.AllData:
            if sitestats == 'good':
                sums += tsk.Get2DArrayOfErrors(overridetiersonly=True)[:,0]
            elif sitestats == 'bad' :
                sums += tsk.Get2DArrayOfErrors(overridetiersonly=True)[:,1]
            elif sitestats == 'sum' :
                sums += np.sum( tsk.Get2DArrayOfErrors(overridetiersonly=True) , 1 )


        self.hNTotalErrors = ROOT.TH1D( "hNTotalErrors" + self.Name + "_" + sitestats , label , len(errs) , 0 , len(errs) )
        for i,err in enumerate(errs) :
            if err in ErrorCodes:
                self.hNTotalErrors.GetXaxis().SetBinLabel( i+1 , ErrorCodes[err] ) # + " " + str(err) )
                self.hNTotalErrors.GetXaxis().ChangeLabel( i+1 , 45 )
            else:
                self.hNTotalErrors.GetXaxis().SetBinLabel( i+1 , str(err) )
            self.hNTotalErrors.SetBinContent( i+1 , sums[i] )

        self.hNTotalErrors.SetEntries( self.hNTotalErrors.Integral() )
        return self.hNTotalErrors
        
    def PlotCorrelation(self):
        """
        produce and show the correlation plot, based on the output of GetInfo method of the Task object
        """
        plt.matshow(self.df.corr())
        plt.show()

    def GroupBy( self,  var1 , var2 ):
        """
        group by var1 and var2 and plot the counts
        """
        groupby = self.df.groupby([var1 , var2])
        var3 = "nErrorsInGoodSites" if "nErrorsInBadSites" in [var1,var2] else "nErrorsInBadSites"
        df_action_error_count = groupby[var3].count().reset_index()
        df_action_error_count.plot.scatter(x=var1 , y=var2 , s=df_action_error_count[var3] )
        plt.show()

