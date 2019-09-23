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
from PIL import Image

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

        def ShortDescription(self):
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

    def filter(self , action_rank=None , action_code=None , mem=None , splitting=None ):
        ar = True
        if action_rank:
            if type(action_rank) is int:
                if self.GetActionRank() != action_rank:
                    ar = False
            if type(action_rank) is list:
                if self.GetActionRank() not in action_rank:
                    ar = False

        ac = True
        if action_code:
            if action_code != self.action.action:
                ac = False

        mm = True
        if mem:
            if int(self.action.memory) == 0:
                mm = False

        splt = True
        if splitting:
            if not self.action.splitting:
                splt = False

        return ar and ac and mm and splt
            
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
                if int(err) in self.ExcludeErrors:
                    continue
                errcode = all_errors.index( int(err) )
                sites = self.error_tensor[errcode]
                for site in errs[err]:
                    if site in self.ExcludeSites:
                        continue
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
        return self.tasks.AllTakenActions.index( self.action.ShortDescription() )

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
    
    def __init__(self , tsk , name , tasks , site_statuses=['good_sites' , 'bad_sites'] , exclude_errors = [] , exclude_sites = [] ):
        """
        initialize the Task object.
        :param dict tsk: the dictionary from json file, which includes "parameters" and "errors" keys
        :param str name: the name of the task
        :param Tasks tasks: the parent tasks object
        """
        self.Name = name
        self.tasks = tasks

        self.ExcludeErrors = exclude_errors
        self.ExcludeSites = exclude_sites
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


    def GetNErrorCodes(self , good_bad_sum):
        mtx = None
        if good_bad_sum == 0 :
            mtx = self.Get2DArrayOfErrors()[:,0]
        elif good_bad_sum == 1 :
            mtx = self.Get2DArrayOfErrors()[:,1]
        elif good_bad_sum == 2 :
            mtx = np.sum( self.Get2DArrayOfErrors() , 1 )
            
        return np.sum( mtx != 0. )
        

    def GetImage(self , plot=True , error_sorting=None , repeat = False , blue_channel=True ):
        red   = self.error_tensor[:,:,0]*255.0/(self.error_tensor[:,:,0].max() if self.error_tensor[:,:,0].max() else 1.0)
        green = self.error_tensor[:,:,1]*255.0/(self.error_tensor[:,:,1].max() if self.error_tensor[:,:,1].max() else 1.0)
        img = 255*np.ones( (red.shape[0] , red.shape[1] , 3 if blue_channel else 2  ) )
        img[:,:,0] -= red if error_sorting is None else red[error_sorting]
        img[:,:,1] -= green if error_sorting is None else green[error_sorting]

        if repeat :
            ImageMatrix = np.repeat( np.repeat( img , repeat[0] , 0 ) , repeat[1] , 1 )
        else:
            ImageMatrix = img
        if plot:
            self.Image = plt.figure()
            plt.imshow( ImageMatrix, interpolation='nearest' )
            return self.Image
        else :
            return ImageMatrix
    
    def GetDominantError(self , good_bad_sum , nth ):
        '''
        :param int good_bad_sum: 0 for good sites/ 1 for bad sites/ 2 for sum of them
        :param int nth: 1 returns the most dominant error code, 2 return the next dominant one and so on
        '''
        if self.GetNErrorCodes( good_bad_sum ) < nth :
            return -2
        
        mtx = None
        if good_bad_sum == 0 :
            mtx = self.Get2DArrayOfErrors()[:,0]
        elif good_bad_sum == 1 :
            mtx = self.Get2DArrayOfErrors()[:,1]
        elif good_bad_sum == 2 :
            mtx = np.sum( self.Get2DArrayOfErrors() , 1 )
            
        #return self.tasks.all_errors[ np.argpartition( mtx , -nth )[ -nth ] ]
        return np.argpartition( mtx , -nth )[ -nth ]
            
    
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
    def __init__(self, _file , TiersOnly=False, all_sites_=[] , all_errors_=[] , all_actions_=[] , site_statuses = ['good_sites' , 'bad_sites'] , name = "tasks" , countclasses = False , classes = {}
                 , exclude_errors = [] , exclude_sites=[]):
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
        self.ExcludeErrors = exclude_errors
        self.ExcludeSites = exclude_sites

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
        print('total number of tasks in the beginning:' , len(self.js) )
        for tsk in self.js :
            ttsskk = Task( self.js[tsk] , tsk , self , site_statuses , exclude_errors , exclude_sites )
            if np.sum(ttsskk.error_tensor)==0:
                continue
            self.AllData.append(ttsskk)
            shortdesc = self.AllData[-1].action.ShortDescription()
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
            if 'action_rank' not in classes[cls].keys():
                classes[cls]['action_rank'] = []

            if len(classes[cls])!=1 :
                for taken_action in self.AllTakenActions:
                    action_,splitting_ ,xrootd_ , sites_, memory_ , cores_ = taken_action.split('_')

                    ac = True
                    if 'action_code' in classes[cls].keys() :
                        tst_code = classes[cls]['action_code']
                        if type(tst_code).__name__ == 'function' :
                            ac = tst_code( action_ )
                        else:
                            ac = action_ == tst_code
                    mm = True
                    if 'mem' in classes[cls].keys() :
                        tst_code = classes[cls]['mem']
                        if type(tst_code).__name__ == 'function':
                            mm = tst_code( memory_ )
                        else:
                            mm = int(memory_) > tst_code
                    splt = True
                    if 'splitting' in classes[cls].keys():
                        tst_code = classes[cls]['splitting']
                        if type(tst_code).__name__ == 'function':
                            splt = tst_code( splitting_ )
                        else:
                            splt = splitting_ != 'None'

                    if ac and mm and splt :
                        print(taken_action, cls, mm, ac , splt)
                        classes[cls]['action_rank'].append( self.AllTakenActions.index( taken_action ) )
            all_classes.extend( classes[cls]['action_rank'] )
            if -1 in classes[cls]['action_rank']:
                the_one_with_minus1 = cls
                
                
        if the_one_with_minus1 != -1 :
            classes[ the_one_with_minus1 ]['action_rank'].remove( -1 )
            for indx in range(0,len(self.AllTakenActions)):
                if not indx in all_classes :
                    classes[ the_one_with_minus1 ]['action_rank'].append( indx )
        
        from collections import defaultdict
        self.ClassCounts = defaultdict(int) if countclasses else None
        toRemove = []
        print('total number of tasks before filterring:' , len(self.AllData) )
        
        for tsk in self.AllData:
            class_index = -1
            tsk_rank = tsk.GetActionRank()
            for cls in classes:
                #if tsk.filter( **classes[cls] ):
                if tsk_rank in classes[cls]['action_rank']:
                    class_index = self.all_actions.index( cls )
                    break
            if class_index==-1:
                toRemove.append( tsk )
                continue
            tsk.action.SetCodeByShortDesc(class_index)
            if countclasses:
                self.ClassCounts[tsk.action.code()] += 1

        self.AllData=[tsk for tsk in self.AllData if tsk not in toRemove]
            
        if self.ClassCounts!=None:
            MIN = min( self.ClassCounts.values() )
            print('Class with minimum value:' , MIN )
            for idx in self.ClassCounts :
                self.ClassCounts[idx] = MIN/self.ClassCounts[idx]

        self.Classes = classes
        for cls in classes :
            print(cls , self.ClassCounts[ self.all_actions.index(cls) ]   , [ self.AllTakenActions[i] for i in classes[cls]['action_rank'] ] )
        print('Filterred:' , len(toRemove), set([a.action.ShortDescription() for a in toRemove]) )
        print('Total numbers now:' , len(self.AllData) )
            
                
        self.ErrorsGoodBadSites = np.array( [ tsk.Get2DArrayOfErrors() for tsk in self.AllData ] )
        self.AllActions = np.array( [tsk.action.code() for tsk in self.AllData ] )
        if not self.TiersOnly :
            self.df = DataFrame(data=[tsk.GetInfo() for tsk in self.AllData] , columns=self.AllData[0].GetInfo(labelsOnly=True))
        
    def GetShuffledDS(self , n ):
        p = np.random.permutation( len(self.AllData)  )
        return self.ErrorsGoodBadSites[ p[:n] ], self.AllActions[p[:n] ]


    def GetShuffledImages(self , n , err_srt ):
        xxx = []
        yyy = []
        for ii,i in enumerate( np.random.permutation( len(self.AllData) ) ):
            if ii < n :
                xxx.append( self.AllData[i].GetImage( plot=False , error_sorting=err_srt , repeat=False , blue_channel=False ).astype( np.uint8 ) )
                yyy.append( self.AllActions[i] )

        return np.array( xxx ) , np.array( yyy )
    
    def GetTrainTestImages(self , train_ratio , error_sorting , shuffle=False , val_ratio = 0):
        lst = np.random.permutation(len(self.AllData)) if shuffle else range(0,len(self.AllData))
        xtrain = []
        ytrain = []
        xtest  = []
        ytest  = []
        xval   = []
        yval   = []
        ntrain = int(train_ratio*len(self.AllData))
        nval   = int(ntrain*val_ratio)
        for i,n in enumerate(lst):
            xx = self.AllData[n].GetImage( plot=False , error_sorting=error_sorting , repeat=False , blue_channel=False )
            xx = xx.astype( np.uint8 )
            
            yy = self.AllActions[n]
            if self.IsBinary :
                yy = yy.astype(np.uint8)
            else:
                yy = to_categorical( yy , len(self.all_actions) , np.uint8 )

            if i<nval :
                xval.append( xx )
                yval.append( yy )
            elif i < ntrain:
                xtrain.append( xx )
                ytrain.append( yy )
            else :
                xtest.append( xx )
                ytest.append( yy )

        
                
        return np.array( xtrain ) , np.array( ytrain ) , np.array( xtest ) , np.array( ytest ) , np.array( xval ) , np.array( yval )
    
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
                    if int(err) in self.ExcludeErrors:
                        continue
                    if int(err) not in self.all_errors:
                        self.all_errors.append(int(err))
                    for site in sites[err]:
                        if site in self.ExcludeSites:
                            continue
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
            self.hActionsVsErrorInfo.Fill( tsk.action.ShortDescription() , tsk.GetErrorInfo() , 1 )

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


    def GetSumErrorsForAction(self, action , lbl ):
        setattr( self , lbl , np.zeros( ( len(self.all_errors) , 3 ) ) )
        ret = getattr( self , lbl )

        for tsk in self.AllData:
            if tsk.GetActionRank() in action :
                atsk = tsk.Get2DArrayOfErrors(overridetiersonly =True)
                ret[:,0] += atsk[:,0]
                ret[:,1] += atsk[:,1]
                ret[:,2] += atsk[:,0]+atsk[:,1]

        setattr( self , lbl + "_argsort" , np.argsort( ret , 0 ) )
        return ret

    def SortErrorCodes(self , lbls , good_bad_sum , aggregate_function ):
        self.ErrorRanks = np.zeros( ( len(self.all_errors) ) )
        for err_index in range( 0 , len(self.all_errors) ):
            ranks = []
            for lbl in lbls :
                rnk_mtx = getattr( self , lbl + "_argsort" )[:,good_bad_sum]
                ranks.append( np.where( rnk_mtx == err_index )[0][0] )
            #print self.ErrorRanks
            #print err_index
            self.ErrorRanks[ err_index ] = aggregate_function( ranks )

        
        return np.argsort(self.ErrorRanks)
                
    def PlotErrorIndexForAction(self , action , color , marker , dominant = 1 , label = None ):
        import ROOT
        if not label:
            label = action
        name = 'hErrorIndexForAction_' + label
        hRet = ROOT.TH1D( name , label + ";DominantErrors"  , len( self.all_errors ) , 0 , len(self.all_errors ) )
        hRet.SetMarkerStyle( marker )
        hRet.SetMarkerColor( color )
        hRet.SetLineColor( color )
        hRet.SetMarkerSize( 1.4 )
        if type(action) is str :
            action = [self.AllTakenActions.index( action )]
        elif type(action) is int:
            action = [action]
        setattr( self , name , hRet )

        for i,err in enumerate(self.all_errors) :
            hRet.GetXaxis().SetBinLabel( i+1 , str(err) )
        for tsk in self.AllData:
            if tsk.GetActionRank() in action :
                #self.hActionsVsErrorInfo.Fill( tsk.GetActionRank() , tsk.GetErrorInfoRank() )
                hRet.Fill( tsk.GetDominantError( 2 , dominant) , 1 )

        return hRet
    
    def ImageForAction(self , action , lbl , error_sorting ):
        label = action
        name = 'figure_' + lbl
        fig = plt.figure()
        setattr( self , name , fig )

        img_name = 'img_' + lbl
        if hasattr( self , img_name ):
            img = getattr( self , img_name )
        else:
            img = None
            nnn = 0
            for tsk in self.AllData:
                if tsk.GetActionRank() in action :
                    nnn += 1
                    iii = tsk.GetImage( False , None ,repeat=False )
                    if img is None :
                        img = iii
                    else:
                        img += iii
                    del iii

            img /= nnn
            img = img.astype( np.uint8 )

            mmin_0=np.min(img[:,:,0])
            mmax_0=np.max(img[:,:,0])
            
            # Make a LUT (Look-Up Table) to translate image values
            LUT_0=np.zeros(256,dtype=np.uint8)
            LUT_0[mmin_0:mmax_0+1]=np.linspace(start=0,stop=255,num=(mmax_0-mmin_0)+1,endpoint=True,dtype=np.uint8)

            img[:,:,0] = LUT_0[ img[:,:,0] ]

            mmin_1=np.min(img[:,:,1])
            mmax_1=np.max(img[:,:,1])
            
            # Make a LUT (Look-Up Table) to translate image values
            LUT_1=np.zeros(256,dtype=np.uint8)
            LUT_1[mmin_1:mmax_1+1]=np.linspace(start=0,stop=255,num=(mmax_1-mmin_1)+1,endpoint=True,dtype=np.uint8)

            img[:,:,1] = LUT_1[ img[:,:,1] ]
            
            setattr( self , img_name , img )
        if error_sorting is None:
            plt.imshow( np.repeat( np.repeat( img , 10 , 0 ) , 10 , 1 ) , interpolation='nearest' )
        else:
            plt.imshow( np.repeat( np.repeat( img[error_sorting] , 10 , 0 ) , 100 , 1 ) , interpolation='nearest' )
        return fig
    
    
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

