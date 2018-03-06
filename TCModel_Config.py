'''
    TCModel_Config.py contains simulation configurations for TCModel_Run.py.

    Contributors: Vergil R. Haynes, vrhaynes.tech@gmail.com

'''

# Import modules
import os
from os.path import join
from TCModel_Params import PopulationParams




################################################################################
##### General batch configurations
################################################################################
class BatchConfigs(PopulationParams):
    def __init__(self):
        ''' Class defining rules for how single simulation protocols change '''

        # Inherit params from *_Params file
        PopulationParams.__init__(self)

    ####################################
    #                                  #
    #                                  #
    #      DEVELOPMENT PARAMETERS      #
    #                                  #
    #                                  #
    ####################################

        # Used as conditional flag for development (inherited from GeneralParams)
        # self.testing = True

        # Important paths for simulations
        if self.testing is True:
            self.path2output = 'scratch'                 # dir for overwriting testing
            self.path2figs = 'scratch'
            self.path2data = 'scratch'
        else:
            self.path2output = 'output'                  # grandparent output dir
            self.path2figs = join(path2output,'figs')    # parent figure dir
            self.path2data = join(path2output,'dat')     # parent data dir

            # TODO: subdirectories for various batch results

    ####################################
    #                                  #
    #                                  #
    #      GENERAL SIM PARAMETERS      #
    #                                  #
    #                                  #
    ####################################

        # Global batch specification
        self.num_sims = 1

        # # Necessary for NETPYNE batch object
        # self.cfgFile
        # self.netParamsFile
        # self.params
        # self.groupedParams



################################################################################
##### Simulation configurations
################################################################################
class SimulationConfigs(BatchConfigs):
    def __init__(self):
        ''' Class defining single simulation protocols - used by specs.SimConfig() from netpyne '''

        # Inherit parent class params
        BatchConfigs.__init__(self)


    ####################################
    #                                  #
    #                                  #
    #        CONFIGURATION SPECS       #
    #                                  #
    #                                  #
    ####################################


        ###################################
        #  SIMULATIONS
        ###################################
        self.duration = (.2 + .8)*1e3         # runtime (equilibriation + simulation)*(s -> ms)
        self.dt = 0.25                        # internal timestep (ms)

        self.verbose = False                   # show detailed messages

        ###################################
        #  DATA ACQUISITION
        ###################################

        # get population list for including examplars from each
        includedPopList = []

        # Unpackage .Y_pop_ids into one list (for recordTraces)
        for layerLabels in self.Y_pop_ids:
            for popLabel in layerLabels:
                includedPopList.append(popLabel)

        # include only first cell for traces
        includedCellList = []
        for pop in includedPopList:
            includedCellList.append( (pop,0) )

        self.recordTraces = {'Somatic Potential (mv)': {
                                        'sec' : 'comp_1',
                                        'loc' : 0.5,
                                        'var' : 'v'}
                            }

        # Print/save population average firing rates
        self.printPopAvgRates = False

        self.recordStep = 1             # downsampling for saved data (ms)


        ###################################
        #  OUTPUT DETAILS
        ###################################

        self.filename = 'TC_output'
        self.saveFolder = self.path2data
        if not self.testing:
            self.savePickle = True       # save params, network and sim output to pickle file
        else:
            self.savePickle = False



    ####################################
    #                                  #
    #                                  #
    #          ANALYSIS SPECS          #
    #                                  #
    #                                  #
    ####################################


        # Dictionary for specs.SimConfig() in NETPYNE
        self.analysis = {}
        self.analysis.update({'plotRaster' : {'orderBy' : 'y', 'orderInverse' : True,          # ordered L23 on top and L6 on bottom
                                              'timeRange' : [0,1500],
                                              'saveFig' : join(self.path2figs,'TC_Raster.png'), # TODO: change this for batches
                                              'showFig' : False,    # default True
                                              },
                              'plotTraces' : {'include' : includedCellList,      # plot traces for list of cells
                                              'timeRange' : [0,1500],
                                              'oneFigPer' : 'cell',
                                              'saveFig' : join(self.path2figs,'TC_popTraces.png'), # TODO: change this for batches
                                              'showFig' : False,
                                              },
                              # 'plotSpikeHist' : {'timeRange' : [50,500],
                              #                    'graphType' : 'bar',
                              #                    'overlay' : False,
                              #                    'saveFig' : join(self.path2figs,'TC_bl_Rate.png'),
                              #                    'showFig' : False,
                              #                     },
                              # 'plotRatePSD' : {'timeRange' : [50,500],
                              #                  'Fs' : 20,
                              #                  'overlay' : False,
                              #                  'saveFig' : join(self.path2figs,'TC_bl_ratePSD.png'),
                              #                  'showFig' : False,
                              #                 },
                              'plotConn' : {'groupBy' : 'pop',
                                            'saveFig' : join(self.path2figs,'TC_synConn.png'), # TODO: change this for batches
                                            # 'feature' : 'numConns',           # default: strength = weight*probability NOTE: using weight and showing each cell to test
                                            'showFig' : False,
                                            },
                             })

        # Parent dictionary for specs.SimConfig() class
        self.simConfigDict = {
            'duration' : self.duration,
            'dt' : self.dt,
            'verbose' : self.verbose,
            'recordTraces' : self.recordTraces,
            'recordStep' : self.recordStep,
            'filename' : self.filename,
            'savePickle' : self.savePickle,
            'analysis' : self.analysis,
        }





if __name__ == '__main__':

    fullConfigs = SimulationConfigs()

    # Used to reference and checking on appropriate formatting of data structures
    # Will not print when imported using run file
    print '\nFull Configuration Dictionary: fullConfigs\n'
    print dir(fullConfigs)
