'''
    TCModel_Run.py imports parameter dictionaries from TCModel_Params.py and TCModel_Config.py
    then performs automated simulations in batch.

    Simulation designed to reconstruct network model from
    http://www.opensourcebrain.org/projects/thalamocortical/wiki
    using NetPyNe.

    Contributors: Vergil R. Haynes, vrhaynes.tech@gmail.com

'''


# Import modules
from netpyne import specs, sim
import neuron
import os
from os.path import join
import numpy as np
import random
from GeneratedSynapseParams import cellsec_comps
from time import time



################################################################################
#### FUNCTION DECLARATIONS
################################################################################
def createImports(importDict):
    ''' Take dictionary to turn into function input vars '''
    # NOTE: Clunkier than I thought it would be

    label = importDict['label']
    conds = importDict['conds']
    cellName = importDict['cellName']
    fileName = importDict['fileName']

    return label, conds, cellName, fileName

def convertIntListToCompList(list):
    ''' Takes list of integer compartment values and converts into list of str compartment values for NETPYNE '''
    new_list = ['comp_' + str(list[i]) for i in np.arange(len(list))]

    return new_list

def stationaryPoisson(nsyn,lambd,tstart,tstop):
    ''' Generates nsyn stationary possion processes with rate lambda between tstart and tstop'''
    interval_s = (tstop-tstart)*.001
    spiketimes = []
    for i in range(nsyn):
        spikecount = np.random.poisson(interval_s*lambd)
        spikevec = np.empty(spikecount)
        if spikecount==0:
            spiketimes.append(spikevec)
        else:
            spikevec = tstart + (tstop-tstart)*np.random.random(spikecount)
            spiketimes.append(np.sort(spikevec))

    return spiketimes


################################################################################
### LOAD MAIN PARAMETER SETS
################################################################################

# Import parameter dictionaries for NETPYNE
from TCModel_Params import PopulationParams

fullParams = PopulationParams()

# Create class NetParams object to store imported parameters
netParams = specs.NetParams(netParamsDict=fullParams.netParamsDict)

# Importing cell parameters from .hoc files
for labels in fullParams.Y_pop_ids:
    for label_id in labels:

        label, conds, cellName, fileName = createImports(fullParams.Y_importParams[label_id])

        # NOTE: cellParams[X] must be the label and not the label_id
        netParams.cellParams[label] = netParams.importCellParams(label=label,
                                                                 conds=conds,
                                                                 cellName=cellName,
                                                                 fileName=fileName)




################################################################################
##### MAIN SIMULATION
################################################################################
if __name__ == '__main__':

    ####################################
    #                                  #
    #                                  #
    #    CURRENT INJECTION PROTOCOL    #
    #                                  #
    #                                  #
    ####################################

    current_injection = True

    if current_injection:

        # Stimulating electrode source and target
        netParams.stimSourceParams['depol_step_current'] =  {
                    'type': 'IClamp',
                    'del' : 200, 'dur': 200,
                    'amp': 1.5} # del/dur in ms and amp in nA

        # choose section from potential targets (can only be one section)
        # basal_dends = netParams.cellParams['L23_RS_PYR_rule']['secList']['basal_dends']
        # stim_target = random.choice(basal_dends)

        netParams.stimTargetParams['curr_inj->L23_RS_PYR'] = {
                'source' : 'depol_step_current',
                'sec' : 'comp_1',
                'loc' : 0.5,
                'conds' : {'pop' : 'L23_RS_PYR'} # , 'cellList': range(8)}}
        }


    ####################################
    #                                  #
    #                                  #
    #        ECTOPIC AXON SPIKES       #
    #                                  #
    #                                  #
    ####################################

    # Generate random "ectopic" spikes for all glutamatergic axons following Traub et al. 2005
    # NOTE: superifical pyramids mean interval 10s, all other glutamatergic cells 1s
    # TODO: Should be pulse synapse mod but don't think I need that.

    # Synaptic mechanism parameters
    netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 1.0, 'e': 0}

    # Ectopic axonal spike generating stimulation protocols
    netParams.stimSourceParams['ectopic1'] = {'type': 'NetStim', 'interval' : 10000, 'start' : 0, 'noise': 10} # mean interval of 10 s
    netParams.stimSourceParams['ectopic2'] = {'type': 'NetStim', 'interval' : 1000,  'start' : 0, 'noise': 10} # mean interval of 1 s
    # netParams.stimSourceParams['ectopic3'] = {'type': 'NetStim', 'interval' : 20000,  'start' : 0, 'noise': 0.5}  # mean interval of 20 s (this is bare minimum 0.05 spikes/s)

    # Superficial pyramids
    netParams.stimTargetParams['ectopic1->L23_E'] = {'source': 'ectopic1', 'conds': {'cellType': 'PYR', 'yrange' : [  81.6,  587.1]},
                                                     'sec' : 'comp_69', 'loc' : 0.5,                # axon
                                                     'weight': 1, 'delay': 0}
    # Infragranular pyramids
    netParams.stimTargetParams['ectopic2->L5_E'] = {'source': 'ectopic2', 'conds': {'cellType': 'PYR', 'yrange' : [  922.2, 1170.0]},
                                                    'sec' : 'comp_56', 'loc' : 0.5,                # axon
                                                    'weight': 1, 'delay': 0}
    netParams.stimTargetParams['ectopic2->L6_E'] = {'source': 'ectopic2', 'conds': {'cellType': 'PYR', 'yrange' : [  1170.0, 1491.7]},
                                                    'sec' : 'comp_45', 'loc' : 0.5,                # axon
                                                    'weight': 1, 'delay': 0}

    # Granular cells
    netParams.stimTargetParams['ectopic2->L4_E'] = {'source': 'ectopic2', 'conds': {'cellType': 'STEL'},
                                                  'sec' : 'comp_54', 'loc' : 0.5,                # axon
                                                  'weight': 1, 'delay': 0}

    # # Interneurons aren't firing at al so here's a fix
    # netParams.stimTargetParams['ectopic3->all_I'] = {'source' : 'ectopic3', 'conds' : {'cellType' : ['BASK', 'AXO', 'IN']}
    #                                                   'sec' : 'comp_1', 'loc' : 0.5,
    #                                                   'weight' : 1, 'delay' : 0}



    ####################################
    #                                  #
    #                                  #
    #             RUN MODEL            #
    #                                  #
    #                                  #
    ####################################

    # Import configuration dictionaries for NETPYNE
    from TCModel_Config import SimulationConfigs

    fullConfigs = SimulationConfigs()

    # Creare class SimConfig object to store imported configurations
    simConfig  = specs.SimConfig(simConfigDict=fullConfigs.simConfigDict)

    # Build network and run simulation
    sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)
