#https://workflowwebtools.readthedocs.io/en/latest/workflow_procedures.html
#https://docs.google.com/spreadsheets/d/1onZky6rW2z7NRvvQaLrBGaS8iFfYSg_EL71F1l3mTWE/edit#gid=0
#https://twiki.cern.ch/twiki/bin/view/CMSPublic/JobExitCodes
ErrorCodes = {
    73:"RootEmbeddedFileSequence: secondary file list seems to be missing. When overflow procedure is enables, the agent should be able to see the secondary file set as present at those sites as well.",
    84:"The file could not be found",
    85:"If FileReadError, the file could not be read. If R__unzip: error, most likely a bad node with corrupted intermediate files.",
    92:"File was not found and fallback procedure also failed.",

    134:"Strong failure to report immediately",
    137:"Likely an unrelated batch system kill. Sometimes a site-related problem.",
    139:"Segmentation Violation, Segmentation Violation (usually problems with Singularity or CVMFS)",
    
    7000:"CommandLineProcessing",
    7001:"ConfigFileNotFound",
    7002:"ConfigFileReadError",
    
    8001:"OtherCMS, EventGenerationFailure, External LHEpProducer Error,",
    8002:"StdException, Site Issue",
    8003:"Unknown",
    8004:"BadAlloc,virtual memory exhaustion",
    8005:"BadExceptionType",
    
    8006:"ProductNotFound",
    8007:"DictionaryNotFound",
    8008:"InsertFailure",
    8009:"Configuration",
    8010:"LogicError",
    8011:"UnimplementedFeature",
    8012:"InvalidReference",
    8013:"NullPointerError",
    8014:"NoProductSpecified",
    8015:"EventTimeout",
    8016:"EventCorruption Geant4 error",

    8017:"ScheduleExecutionFailure",
    8018:"EventProcessorFailure",

    8019:"FileInPathError",
    8020:"FileOpenError",
    8021:"FileReadError, site issue",
    8022:"FatalRootError",
    8023:"MismatchedInputFiles",
    
    8024:"ProductDoesNotSupportViews",
    8025:"ProductDoesNotSupportPtr",
    
    8026:"NotFound",
    8027:"FormatIncompatibility",
    8028:"FallbackFileOpenError",

    8033:"FileWriteError", #Could not write output file (FileWriteError) (usually local disk problem)",
    8501:"EventGenerationFailure",
    
    10001:"Connectivity problems",
    10002:"CPU load is too high",
    10003:"CMS software initialisation script cmsset_default.sh failed",
    10004:"CMS_PATH not defined",
    10005:"CMS_PATH directory does not exist",
    10006:"scramv1 command not found",
    10007:"Some CMSSW files are corrupted/non readable",
    10008:"Scratch dir was not found",
    10009:"Less than 5 GB/core of free space in scratch dir",
    10010:"Could not find X509 certificate directory",
    10011:"Could not find X509 proxy certificate",
    10012:"Unable to locate the glidein configuration file",
    10013:"No sitename detected! Invalid SITECONF file",
    10014:"No PhEDEx node name found for local or fallback stageout",
    10015:"No LOCAL_STAGEOUT section in site-local-config.xml",
    10016:"No frontier-connect section in site-local-config.xml",
    10017:"No callib-data section in site-local-config.xml",
    10018:"site-local-config.xml was not found",
    10019:"TrivialFileCatalog string missing",
    10020:"event_data section is missing",
    10021:"no proxy string in site-local-config.xml",
    10022:"Squid test was failed",
    10023:"Clock skew is bigger than 60 sec",
    10031:"Directory VO_CMS_SW_DIR not found",
    10032:"Failed to source CMS Environment setup script such as cmssset_default.sh, grid system or site equivalent script",
    10034:"Required application version is not found at the site (see HERE.)",
    10040:"failed to generate cmsRun cfg file at runtime",
    10042:"Unable to stage-in wrapper tarball.",
    10043:"Unable to bootstrap WMCore libraries (most likely site python is broken).",
    10050:"WARNING test_squid.py: One of the load balance Squid proxies",
    10051:"WARNING less than 20 GB of free space per core in scratch dir",
    10052:"WARNING less than 10MB free in /tmp",
    10053:"WARNING CPU load of last minutes + pilot cores is higher than number of physical CPUs",
    10054:"WARNING proxy shorther than 6 hours",

    11003:"JobExtraction failures",
    
    
    
    50110:"Executable not found",
    50111:"no exe permissions", #Executable file has
    50113:"Executable did not get enough arguments",
    50115:"cmsRun segfaulted", #cmsRun did not produce a valid job report at runtime (often means 
    50116:"No exit code from cmsRun", #Could not determine exit code of cmsRun executable at runtime",
    50513:"Failure in SCRAM setup scripts", #"Failure to run SCRAM setup scripts",
    50660:"RAM problem (wrapper)", #Application terminated by wrapper because using too much RAM (RSS)", Performance kill: Job exceeding maxRSS
    50661:"VSIZE problem (wrapper)" , #Application terminated by wrapper because using too much Virtual Memory (VSIZE)",
    50662:"DISK usage (wrapper)", #Application terminated by wrapper because using too much disk",
    50663:"CPU Time (wrapper)", #Application terminated by wrapper because using too much CPU time",
    50664:"Wall clock time (wrapper) stage out error" , #Application terminated by wrapper because using too much Wall Clock time",
    50665:"Application terminated by wrapper because it stay idle too long",
    50669:"Application terminated by wrapper for not defined reason",

    
    60302:"Output file(s) not found (see HERE.)",
    60307:"Failed to copy an output", # file to the SE (sometimes caused by timeout issue). Or by the issues mentioned HERE.",
    60311:"Local Stage Out Failure using site specific plugin",
    60312:"Failed to get file TURL via lcg-lr command, LogCollect error",
    60315:"ProdAgent StageOut initialisation error (Due to TFC, SITECONF etc)",
    60316:"Failed to create a directory on the SE",
    60317:"Forced timeout for stuck stage out",
    60318:"Internal error in Crab cmscp.py stageout script",
    60319:"Failure to do AlcaHarvest stageout (WMAgent)",
    60320:"Failure to communicate with ASO server",
    60321:"Site related issue: no space, SE down, refused connection.",
    60322:"User is not authorized to write to destination site.",
    60323:"User quota exceeded.",
    60324:"Other stageout exception.",
    60401:"Failure to assemble LFN in direct-to-merge by size (WMAgent)",
    60402:"Failure to assemble LFN in direct-to-merge by event (WMAgent)",
    60403:"Timeout during attempted file transfer - status unknown (WMAgent)",
    60404:"Timeout during staging of log archives - status unknown (WMAgent)",
    60405:"Failed to stage out log", #General failure to stage out log archives (WMAgent)",
    60407:"Timeout in staging in log files during log collection (WMAgent)",
    60408:"Failure to stage out of log files during log collection (WMAgent)",
    60409:"Timeout in stage out of log files during log collection (WMAgent)",
    60450:"No output reported skipped file",# files present in the report",
    60451:"Output file lacked adler32 checksum (WMAgent)",

    61202:"Cant determine proxy filename. X509 user proxy required for job.",
    
    71101:"No suitable site found to submit", #No sites are available to submit the job because the location of its input(s) do not pass the site whitelist/blacklist restrictions (WMAgent) Twas 61101",
    71102:"Suitable site is aborted" , #The job can only run at a site that is currently in Aborted state (WMAgent)",
    71103:"Job pickle can't be loaded by JobSubmitter",
    71104:"Suitable site is in Draining" , #The job can run only at a site that is currently in Draining state (WMAgent)",Site containing only available copy of input datasets is in drain.
    71300:"The job was killed by the WMAgent, reason is unknown (WMAgent)",
    71301:"The job was killed by the WMAgent because the site it was running at was set to Aborted (WMAgent)",
    71302:"Site is going to Draining state", #The job was killed by the WMAgent because the site it was running at was set to Draining (WMAgent)",
    71303:"Site is going Down", #The job was killed by the WMAgent because the site it was running at was set to Down (WMAgent)",
    71304:"Wall clock time", #The job was killed by the WMAgent for using too much wallclock time (WMAgent)",
    71305:"Wall clock time", #The job was killed by the WMAgent for using too much wallclock time (WMAgent)",
    71306:"Wall clock time", #The job was killed by the WMAgent for using too much wallclock time (WMAgent)",
    70318:"Failure in DQM upload.",
    70452:"No run/lumi information in file - Merge Issue",

    
    80000:"Internal error in CRAB job wrapper",
    80001:"No exit code set by job wrapper.",
    80453:"Unable to determine pset hash from output file (CRAB3).",

    
    90000:"Error in CRAB3 post-processing step (currently includes basically errors in stage out and file metadata upload).",
    99109:"Uncaught exception by WMAgent,stage out error, site error", # step executor" (often staging out problems),
    99303:"The agent has not found the JobReport.x.pkl file. Regardless of the status of the job, it will fail the job with 99303. Work is needed to make this error as rare as possible since there are very limited cases to make the file fail to appear.",
    99304:"Site issue	No Job Report & specific case we don't even have job cache directory",
    
}

def ErrorRange(err):
    if err in [-1, 0]:
        return 0
    elif 0 < err <= 512 :
        return 1
    elif 7000 <= err <=9000 :
        return 2
    elif 10000 <= err <=19999:
        return 3
    elif 50000 <= err <=59999:
        return 4
    elif 60000 <= err <= 69999:
        return 5
    elif 70000 <= err <=79999 :
        return 6
    elif 80000 <= err <= 89999 :
        return 7
    elif 90000 <= err <= 99999 :
        return 8

ErrRanges = [ "-1,0",
              "Unix", #standard ones in Unix and indicate a CMSSW abort that the cmsRun did not catch as exception
              "cmsRun",
              "environment",
              "executable",
              "stagingOUT",
              "WMAgent" ,
              "CRAB3" ,
              "Others" ]
def ErrRangeStr(err):
    return ErrRanges[ ErrorRange(err) ]

def ErrCategory(err):
    if ErrorRange(err) in [0,1]:
        return 0
    if not err in ErrorCodes :
        return 0
    errdesc = ErrorCodes[err].lower()

    if any( x in errdesc for x in ['ram', 'vsize' , 'memory'] ):
        return 1
    if any( x in errdesc for x in ['cpu', 'clock'] ):
        return 2
    if any( x in errdesc for x in ['stage' , 'file'] ):
        return 3
    if any( x in errdesc for x in ['site'] ):
        return 4

    return 0
Categories = [ 'Unk' , 'MEM' , 'Time' , 'File' , 'Site' , 'Others' ]
def ErrDescription(err):
    #rng = ErrRangeStr( err )
    cat = Categories[ ErrCategory( err ) ]
    #return cat + "-" + rng
    return cat
