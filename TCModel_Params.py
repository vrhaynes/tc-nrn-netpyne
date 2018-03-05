'''
    TCModel_Params.py contains parameter values for TCModel_Run.py.
    Here parameter classes are created to imported in main simulation program.

    Contributors: Vergil R. Haynes, vrhaynes.tech@gmail.com

'''
################################################################################
#### TODO LIST
################################################################################
#   (1) Extend to include thalamic portion
#   (5) Update _Y_project() to include special cases for synaptic params
#   (8) Interneuron segment "levels" are defined in Cunningham et al 2004 SOMEWHERE
#       - plus need for every cell type the axonal intial segment
#       compartments which aren't defined - this is necessary for
#       axoaxonic syanptic connections -> just use levels description from groucho.hoc and compare to
#       figures in Traub appendix for finding synapse lists
#   (9) TODO IMPORTANT! Does weight in connParams override g in synMechParams??


# Import modules
from collections import OrderedDict as ODict
from itertools import product
import math
import neuron
import numpy as np
import os
from os.path import join
from GeneratedSynapseParams import *


# Important paths for simulations
path2cells = 'cells/generatedNEURON'

# Compile mod files and load them
# Uncomment 'os.system()' on first run
os.system('''
          cd mod/generatedNEURON
          nrnivmodl
          ''')
neuron.load_mechanisms('mod/generatedNEURON/')


################################################################################
#### Function declarations
################################################################################
def findNumPops(poplist,pop):
    ''' Sums over totals in pop list to find current pop id '''
    count = 0

    for i in np.arange(pop):
        count += poplist[i]

    return count


def convertIntListToCompList(list):
    ''' Takes list of integer compartment values and converts into list of str compartment values for NETPYNE '''
    new_list = ['comp_' + str(list[i]) for i in np.arange(len(list))]

    return new_list


################################################################################
##### General parameters
################################################################################
class GeneralParams(object):
    def __init__(self):
        ''' Class defining global domain and network parameters used in sub-classes '''

    ####################################
    #                                  #
    #                                  #
    #      DEVELOPMENT PARAMETERS      #
    #                                  #
    #                                  #
    ####################################

        # Used as conditional flag for development (Used in *_Config file also)
        self.testing = True
        self.includeGJ = True


    ####################################
    #                                  #
    #                                  #
    #     GENERAL MODEL PARAMETERS     #
    #                                  #
    #                                  #
    ####################################


        ###################################
        #  DOMAIN-LEVEL DESCRIPTIONS
        ###################################

        # Geometry
        # (Cartesian axes)
        #          y    z
        #          ^  ^
        #          | /
        #   x <--- o --
        #         /|
        self.shape = 'cylinder' # Domain geometry
        self.sizeX = 100        # x-dimension (um)
        self.sizeY = 1500       # y-dimension (um)
        self.sizeZ = 100        # z-dimension (um)

        # Cortical depths (NOTE: Notice negative signs - expressed as absolute depth so positive)
        self.layerBoundaries = -1*np.array([[    0.0,   -81.6],       # Pia layer
                                            [  -81.6,  -587.1],       # Layer 2/3
                                            [ -587.1,  -922.2],       # Layer 4
                                            [ -922.2, -1170.0],       # Layer 5
                                            [-1170.0, -1491.7]])      # Layer 6



        # MISC (NOTE: if adding additional spatial features - remember to add keyargs to neParamsDict)
        # self.propVelocity = 100.0      # propagation velocity (um/ms)
        # self.probLengthConst = 150.0   # length constant for conn probability (um)


        # Parent dictionary for specs.NetParams() class (Doesn't include everything)
        self.netParamsDict = {
                'shape' : self.shape,
                'sizeX' : self.sizeX,
                'sizeY' : self.sizeY,
                'sizeZ' : self.sizeZ,
        }



################################################################################
##### Network parameters
################################################################################
class NetworkParams(GeneralParams):
    def __init__(self):
        ''' Class defining network-level model parameters '''

        # Inherit parent class params
        GeneralParams.__init__(self)

    ####################################
    #                                  #
    #                                  #
    #       ALL POPS PARAMETERS        #
    #                                  #
    #                                  #
    ####################################

        # Population counds
        self.num_pops = 12
        self.num_layers = 4 # between 1 and 4 (no pia layer + layers 2 and 3 are one)
        self.num_layer_pops = np.array([5, 1, 2, 4])

        # Population sizes organized by populations within layers
        self.num_cells = [[1000,          # Layer 23 RS pyramids
                             50,          # Layer 23 FRB pyramids
                             90,          # superficial (L23) FS interneurons (bask)
                             90,          # superficial (L23) FS interneurons (axo-axo)
                             90],         # superficial (L23) LTS interneurons
                           [240],         # Layer 4 spiny stellate
                           [800,          # Layer 5 tufted IB pyramids
                            200],         # Layer 5 tufted RS pyramids
                           [100,          # deep (L56) FS interneurons (bask)
                            100,          # deep (L56) FS interneurons (axo-axo)
                            100,          # deep (L56) LTS interneurons
                            500]]         # Layer 6 nontufted RS pyramids
        if self.testing:
            # Maximum cell count is a tenth of full model
            self.num_cells = [[100,          # Layer 23 RS pyramids
                                 5,          # Layer 23 FRB pyramids
                                 9,          # superficial (L23) FS interneurons (bask)
                                 9,          # superficial (L23) FS interneurons (axo-axo)
                                 9],         # superficial (L23) LTS interneurons
                               [24],         # Layer 4 spiny stellate
                               [80,          # Layer 5 tufted IB pyramids
                                20],         # Layer 5 tufted RS pyramids
                               [10,          # deep (L56) FS interneurons (bask)
                                10,          # deep (L56) FS interneurons (axo-axo)
                                10,          # deep (L56) LTS interneurons
                                50]]         # Layer 6 nontufted RS pyramids

        # NOTE: Currently pre- and post-synaptic populations are the same + not always the case, e.g., pops outside the network
        self.N_X = np.array(self.num_cells) # convert pre-synaptic list into array
        self.N_Y = np.array(self.num_cells) # convert post-synaptic list into array

        # Layers in which cells exist
        self.layers = [['L23', 'L23', 'L23', 'L23', 'L23'],
                       ['L4'],
                       ['L5', 'L5'],
                       ['L56', 'L56', 'L56', 'L6']]

        # principal cell types pseudo-dict: {
        # NOTE: Not entirely accurate, TODO: Consider Izhikevich & Edelman (2008) nomenclature
        #         'PYR' : pyramid
        #         'IN' : interneuron
        #         'STEL' : stellate
        #         'BASK' : basket
        #         'AXO' : axoaxonic
        # }
        self.p_types = [['PYR', 'PYR', 'BASK', 'AXO', 'IN'],    # layer 23
                        ['STEL'],                               # layer 4
                        ['PYR', 'PYR'],                         # layer 5
                        ['BASK', 'AXO', 'IN', 'PYR']]           # layer (56)6


        # electrophysiological types pseudo-dict: {
        # NOTE: Not entirely accurate, TODO: Consider Markram et al. (2015) nomenclature
        #         'RS' : regular spiking
        #         'FRB' : fast rhythmic bursting
        #         'IB' : intrinsic bursting
        #         'FS' : fast spiking
        #         'LTS' : low-threshold spiking
        # }
        self.e_types = [['RS', 'FRB', 'FS', 'FS', 'LTS'],          # layer 23
                        ['RS'],                                    # layer 4
                        ['IB', 'RS'],                              # layer 5
                        ['FS', 'FS', 'LTS', 'RS']]                 # layer (56)6

        # Template names (from *_template.hoc files in cells/ directory)
        # self.templatenames = [['suppyrRS', 'suppyrFRB', 'supbask', 'supaxax','supLTS'],
        #                       ['spinstell'],
        #                       ['tuftIB', 'tuftRS'],
        #                       ['deepbask', 'deepaxax','deepLTS','nontuftRS']]

        # # TODO: Use generated NEURON models
        self.templatenames = [['L23PyrRS', 'L23PyrFRB_varInit', 'SupBasket', 'SupAxAx','SupLTSInter'],
                              ['L4SpinyStellate'],
                              ['L5TuftedPyrIB', 'L5TuftedPyrRS'],
                              ['DeepBasket', 'DeepAxAx','DeepLTSInter','L6NonTuftedPyrRS']]

        # Assemble into single zip list
        self.Y_ziplist = zip(self.layers,
                            self.p_types, self.e_types, self.N_Y, self.templatenames)


        ###################################
        #  CONNECTIVITY DESCRIPTIONS
        ###################################

        # Rows are presynaptic population and columns are postsynaptic population

        # Number of connections from PRE pop to POST pop
        #                          L23E L23E L23I L23I L23I L4E  L5E  L5E  L56I L56I L56I  L6E
        self.num_conns = np.array([[50,  50,  90,  90,  90,  3,   60,  60,  30,  30,  30,  3],   # L23 RS PYR
                                   [5,   5,   5,   5,   5,   1,   3,   3,   3,   3,   3,   1],   # L23 FRB PYR
                                   [20,  20,  20,  20,  20,  20,  0,   0,   0,   0,   0,   0],   # L23 FS BASK
                                   [20,  20,  0,   0,   0,   5,   5,   5,   0,   0,   0,   5],   # L23 FS AXO
                                   [20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20,  20],  # L23 LTS IN
                                   [20,  20,  20,  20,  20,  30,  20,  20,  20,  20,  20,  20],  # L4 RS STELL
                                   [2,   2,   20,  20,  20,  30,  50,  20,  20,  20,  20,  20],  # L5 IB PYR
                                   [2,   2,   20,  20,  20,  30,  20,  10,  20,  20,  20,  20],  # L5 RS PYR
                                   [0,   0,   0,  0,    0,   20,  20,  20,  20,  20,  20,  20],  # L56 FS BASK
                                   [5,   5,   0,  0,    0,   5,   5,   5,   0,   0,   0,   5],   # L56 FS AXO
                                   [10,  10,  10, 10,   10,  20,  20,  20,  20,  20,  20,  20],  # L56 LTS IN
                                   [10,  10,  10,  10,  10,  10,  10,  10,  10,  10,  1,   20]]) # L6 RS PYR

        if self.testing:
            # Proportional to population size reduction above
            self.num_conns = self.num_conns/10.

            # bad way to do this but lazy right now
            for i, rows in enumerate(self.num_conns):
                for j, elem in enumerate(rows):
                    if elem > 0 and elem < 1:
                        self.num_conns[i][j] = 1

            self.num_conns = self.num_conns.astype(int)

        # Intra-columnar connection probabilities
        self.conn_probs = self._CYX_get()

        # List of all synapse mods -> self.syn_list is created later to handle specific versions
        self.mod_list = ['AMPA', 'NMDA', 'GABAA', 'gGapPar']   # name of synapse mod

        # specify ranges for random initialization of potentials
        self.v_range = [-70,20] # mV


    ###################################
    #  LOCAL HELPER FUN
    ###################################

    def _CYX_get(self):
        ''' Return matrix with elements of the connection rate, i.e.,
            the probability that a member of the postsynaptic population, Y, receives input
            from a member of the presynaptic population, X.

            Connection prob = N_INPUTS_PRE_TO_POST*NUM_POST/(NUM_PRE*NUM_POST)
                            = N_INPUTS_PRE_TO_POST/NUM_PRE

            This is an approximation to the scheme described in Traub et al. 2005.

            NOTE: implemented in _Y_project() is traubCellConn.
            See net.py in NETPYNE tool for details.
        '''
        dim = (self.num_pops,self.num_pops)
        C_XY = np.zeros(dim)

        N_pre = [N for layer in self.num_cells for N in layer]

        for pre_i in np.arange(self.num_pops):
            for post_j in np.arange(self.num_pops):

                C_XY[pre_i][post_j] = self.num_conns[pre_i][post_j]/float(N_pre[pre_i])

        return C_XY



################################################################################
##### Population parameters
################################################################################
class PopulationParams(NetworkParams):
    def __init__(self):
        ''' Class defining population-level model parameters - used by specs.NetParams() from netpyne '''

        # Inherit parent class params
        NetworkParams.__init__(self)

    ####################################
    #                                  #
    #                                  #
    #        MODEL PARAMETERS          #
    #                                  #
    #                                  #
    ####################################

        # Cell specific description dictionary as nested dictionary
        self.Y_populationParams = self._Y_describePop()        # sorted to make sure cells are ordered by layer

        # Cell specific description for importing NEURON templates
        self.Y_importParams = self._Y_import()


        ###################################
        #  SYNAPTIC MECHANISMS
        ###################################

        # Load synaptic mechanisms
        self.Y_synapseMechParams = self._Y_describeSyn()


        ###################################
        #  CONNECTIVITY PARAMETERS
        ###################################

        # Set up cell specific parameters in terms of model, syn loc, and type
        self.Y_projectParams = self._Y_project()

        # Update netParamsDict
        self.netParamsDict.update({
                # 'cellParams' :    # NOTE: Not used because importing using specs.NetParams.importCellParams()
                'popParams' : self.Y_populationParams,
                'synMechParams' : self.Y_synapseMechParams,
                'connParams' : self.Y_projectParams,
        })



    ###################################
    #  LOCAL HELPER FUN
    ###################################

    def _Y_describePop(self):
        ''' Return nested dictionary with necessary object NETPYNE descriptors

            Used for NETPYNE popParams() functionality
            INPUTS:

            Example parameter dictionary for run file:
            >> from netpyne import specs
            >> netParams = specs.NetParams()
            >> netParams.popParams['L23_RS_PYR'] = {'cellType': 'PYR', 'numCells': 36,
            >>                                     'yRange': [  81.6,  587.1],
            >>                                     'vInit': [ -70, 20]
            >>                                     'cellModel': 'suppyrRS_mod'}

            Keyargs are stored in 'Y_populationParams'

            See http://neurosimlab.org/netpyne/index.html for more details
        '''

        # Populaton dictionary for NETPYNE instantiation
        Y_populationParams = ODict()    # Maintain order for plotting later

        # Construct full cell ids for nesting
        self.Y_pop_ids = [[],[],[],[]]
        for i, (layers, ptypes, etypes, _, _) in enumerate(self.Y_ziplist):
            for layer, ptype, etype in zip(layers, ptypes, etypes):
                self.Y_pop_ids[i].append(layer + '_' + etype + '_' + ptype)

        # Calculate population depths
        self.depths = self._calcDepths()

        # Update ziplist
        self.Y_ziplist = zip(self.Y_pop_ids, self.depths, self.layers,
                            self.p_types, self.e_types, self.N_Y, self.templatenames)

        # Create data structure
        for i, (pops, depths, layers, ptypes, _, sizes, temps) in enumerate(self.Y_ziplist):
            for pop, depth, ptype, size, temp in zip(pops, depths, ptypes, sizes, temps):
                Y_populationParams.update({pop : {
                    'cellType' : ptype,
                    'numCells' : size,
                    'yRange' : depth.tolist(),      # NETPYNE accepts yRange as a list
                    'vInit' : self.v_range,
                    'cellModel' : temp + '_mod',
                    }
                })

        return Y_populationParams

    def _calcDepths(self):
        ''' Return cortical depth for each population

            TODO: No need for the conversion from array to list.
            NETPYNE uses lists so I could refactor later
        '''

        # Range for each layer EXCEPT pia layer
        depths = self.layerBoundaries[1:]

        pop_depths = [[],[],[],[]]
        for i, layers in enumerate(self.layers):
            for layer in layers:
                pop_depth = []
                if layer.rfind('23')>=0:
                    pop_depth = np.r_[pop_depth, depths[0]]
                    pop_depths[i].append(pop_depth)
                elif layer.rfind('4')>=0:
                    pop_depth = np.r_[pop_depth, depths[1]]
                    pop_depths[i].append(pop_depth)
                elif layer.rfind('56')>=0:
                    lb, ub = depths[3][1], depths[2][0]
                    pop_depth = np.r_[pop_depth, np.array([ub, lb])]
                    pop_depths[i].append(pop_depth)
                elif layer.rfind('5')>=0:
                    pop_depth = np.r_[pop_depth, depths[2]]
                    pop_depths[i].append(pop_depth)
                elif layer.rfind('6')>=0:
                    pop_depth = np.r_[pop_depth, depths[3]]
                    pop_depths[i].append(pop_depth)
                else:
                    raise Exception, 'Cortical depth not found'

        return pop_depths

    def _Y_describeSyn(self):
        ''' Return nested dictionary with necessary object NETPYNE descriptors

            Used for NETPYNE importCellParams() functionality
            INPUTS:

            Example parameter dictionary for run file:
            >> from netpyne import specs
            >> netParams = specs.NetParams()
            >> netParams.synMechParams['AMPA_L23_RS_PYR_to_L23_LTS_IN'] = {
            >>                                     'mod': 'AMPA'
            >>                                     'tau' : 1.e0,
            >>                                     'g' : 2.e-3
            >>                                   }

            Keyargs are stored in 'Y_synapseMechParams'

            See http://neurosimlab.org/netpyne/index.html for more details
        '''

        # Synapse Mechanisms dictionary for NETPYNE instantiation
        Y_synapseMechParams = {}

        # Synapse list for project labels
        self.syn_list = []


        for pops_i, _, _, ptypes_i, etypes_i, _, temps_i in self.Y_ziplist:       # presynaptic layer
            for pops_j, _, _, ptypes_j, etypes_j, _, temps_j in self.Y_ziplist:   # postsynaptic layer

                for pop_i, ptype_i, etype_i, temp_i in zip(pops_i, ptypes_i, etypes_i, temps_i):         # presynaptic cell x

                    # Gap Junctions (NOTE: all but axo-axonic interneurons)
                    if self.includeGJ:
                        if not ptype_i == 'AXO':
                            label = 'GJ_' + pop_i
                            g_id = 'gGAP_' + temp_i

                            # Update synapse list
                            self.syn_list.append(label)

                            Y_synapseMechParams.update({label : {'mod' : self.mod_list[3],
                                                                 'g' : g_gap[g_id]
                                                                }
                                                        })

                    for pop_j, ptype_j, etype_j, temp_j in zip(pops_j, ptypes_j, etypes_j, temps_j):     # postsynaptic cell


                        if ptype_i in ['PYR','STEL']:
                            # AMPARs
                            label1 = 'AMPA_' + pop_i + '_to_' + pop_j
                            var_id1 = 'AMPA_' + temp_i + '_to_' + temp_j
                            tau_id1 = 'tau' + var_id1
                            g_id1 = 'g' + var_id1

                            # Update synapse list
                            self.syn_list.append(label1)

                            # NMDARs
                            label2 = 'NMDA_' + pop_i + '_to_' + pop_j
                            var_id2 = 'NMDA_' + temp_i + '_to_' + temp_j
                            tau_id2 = 'tau' + var_id2
                            g_id2 = 'g' + var_id2

                            # Update synapse list
                            self.syn_list.append(label2)

                            if tau_id1 in tau_syn: # Check if in SynapseParams
                                Y_synapseMechParams.update({label1 : {'mod' : self.mod_list[0], # AMPA
                                                                      'tau' : tau_syn[tau_id1],
                                                                      'g' : g_syn[g_id1]*2,# if ptype_i == 'PYR' else g_syn[g_id1],
                                                                      },
                                                            label2 : {'mod' : self.mod_list[1], # NMDA
                                                                      'tau' : tau_syn[tau_id2],
                                                                      'g' : g_syn[g_id2]*.2 if ptype_j in ['BASK','AXO','IN'] else g_syn[g_id2]*2.5,
                                                                      }
                                                            })
                            else:
                                continue

                        if ptype_i in ['BASK', 'AXO', 'IN']:
                            # GABARs
                            label = 'GABA_' + pop_i + '_to_' + pop_j
                            var_id = 'GABA_' + temp_i + '_to_' + temp_j
                            tau_id = 'tau' + var_id
                            g_id = 'g' + var_id

                            # update synapse list
                            self.syn_list.append(label)

                            if tau_id in tau_syn:   # Check if in SynapseConfig
                                Y_synapseMechParams.update({label : {'mod' : self.mod_list[2],
                                                                    'tau' : tau_syn[tau_id],
                                                                    'g' : g_syn[g_id]
                                                                    }
                                                            })
                            else:
                                continue


        return Y_synapseMechParams

    def _Y_import(self):
        ''' Return nested parameter dictionary for NEURON template usage

            Used for NETPYNE importCellParams() functionality
            INPUTS:
                - label : name of rule for later references
                - conds : dict of cell properties to use for applying the rule
                - fileName : name of .py or .hoc file including format
                - cellName : name of object or template in model file
                - importSynMechs : boolean

            Example parameter dictionary for run file:
            >> from netpyne import specs
            >> netParams = specs.NetParams()
            >> netParams.cellParams['L23_RS_PYR_rule'] = netParams.importCellParams(
            >>          label='L23_RS_PYR_rule',
            >>          conds={'cellType': 'PYR', 'cellModel': 'suppyrRS_mod'},
            >>          fileName=join(path2cells,'suppyrRS.hoc'), cellName='suppyrRS')

            Keyargs are stored in 'Y_importParams'

            See http://neurosimlab.org/netpyne/index.html for more details
        '''

        # Population dictionary for NEURON templates
        Y_importParams = {}

        # Create data structure
        for pops, _, _, ptypes, _, _, temps in self.Y_ziplist:
            for pop, ptype, temp in zip(pops, ptypes, temps):
                Y_importParams.update({ pop : {
                    'label' : pop + '_rule',
                    'conds' : {'cellType' : ptype, 'cellModel' : temp + '_mod'},
                    'fileName' : path2cells + '/' + temp + '.hoc',
                    'cellName' : temp,
                    }
                })

        return Y_importParams

    def _Y_project(self):
        ''' Return nested parameter dictionary for NETPYNE connection parameters

            Used for NETPYNE connParams() functionality
            INPUTS:
                - preConds : dict of cell properties to use for presynaptic cells
                - postConds : dict of cell properties to use for postsynaptic cells
                - probability : connection rate if using random connectivity
                - weight : synaptic weight (e.g, random value between 0 and 1 scaled by maximal synaptic conductance)
                - delay : transmission delay (can be function of distance if desired)
                - sec : location of synapse on postsynaptic cell
                - synMech : name of synapse model
                - and others

            Example parameter dictionary for run file:
            >> # Recurrent connectivity
            >> netParams.connParams['recurrent_E'] = {
            >>     'preConds': {'cellType': 'L23_PYR_RS', 'yRange' : [  81.6,  587.1]},  # redudant example as L23 cellType => this yRange
            >>     'postConds': {'cellType': 'L23_PYR_RS', 'yRange' : [  81.6,  587.1]}, # randomly connect PYR->PYR
            >>     'probability' : 0.06,                               # connection rate
            >>     'weight': '0.001635*uniform(0,1)',                                 # maximal synaptic conductace (microS) * uniform(0,1)
            >>     'delay': '6*exp(-dist_3D/probLengthConst)',         # transmission delay (ms)
            >>     'sec': 'Soma',
            >>     'synMech' : 'AMPA'
            >> }

            Keyargs are stored in 'Y_projectParams'

            See http://neurosimlab.org/netpyne/index.html for more details
        '''

        Y_projectParams = {}


        # Create data structure
        for i, (pops_i, _, _, ptypes_i, etypes_i, _, temps_i) in enumerate(self.Y_ziplist):       # presynaptic layer
            for j, (pops_j, _, _, ptypes_j, etypes_j, _, temps_j) in enumerate(self.Y_ziplist):   # postsynaptic layer


                for k, (pop_i, ptype_i, etype_i, temp_i) in enumerate(zip(pops_i, ptypes_i, etypes_i, temps_i)):         # presynaptic population x

                    # pre cell id
                    pre_id = findNumPops(self.num_layer_pops,i) + k

                    # create HOMOLOGOUS gap junction dictionary
                    # TODO: number of synsPerConn could be handled with more randomness like in Traub et al., 2005
                    if self.includeGJ:
                        if not ptype_i == 'AXO':
                            gjtype = 'gj_' + pop_i
                            comp_ids = 'compallow_' + temp_i
                            secs = convertIntListToCompList(comp_gap[comp_ids])
                            num_gj = n_gap['nGAP_' + temp_i]

                            Y_projectParams.update({ gjtype : {
                                    'preConds' : {'pop' : pop_i}, 'postConds' : {'pop' : pop_i},
                                    'weight' : 1.,
                                    'synMech' : 'GJ_' + pop_i,
                                    'connFunc' : 'traubCellConn',
                                    'numPreToPost' : self.num_conns[pre_id][pre_id],
                                    'synsPerConn' : num_gj,
                                    'gapJunction' : True,       # TODO True vs 'pre' vs 'post'?
                                    'sec' : secs,
                                    'preSec' : secs
                                }
                            })

                    for l, (pop_j, ptype_j, etype_j, temp_j) in enumerate(zip(pops_j, ptypes_j, etypes_j, temps_j)):     # postsynaptic population

                        # post cell id
                        post_id = findNumPops(self.num_layer_pops,j) + l


                        # Avoid trying to create synapses without models
                        if self.num_conns[pre_id][post_id] > 0:

                            # key for compartments that can have synapses between any two cell types
                            comp_ids = 'compallow_' + temp_i + '_to_' + temp_j
                            secs = convertIntListToCompList(comp_syn[comp_ids])


                            # Determine type of synapse
                            if ptype_i in ['PYR','STEL']:
                                syntype = 'E'
                            elif ptype_i in ['BASK','AXO','IN']:
                                syntype = 'I'
                            else:
                                # Flag for updating classification of principal cell-types
                                # Example: 'IN' class may be subdivided into 'SOM' and 'PV'
                                #           for somatostatin- and parvalbumin-expressing, respectively, at some point
                                #           so this section needs to be changed
                                raise Exception, 'UNSPECIFIED PRINCIPAL CELL-TYPE'

                            # Need to adjust for E-type connections having both AMPARs and NMDARs
                            if syntype == 'E':

                                # Name connection label
                                # NOTE: label used to create connection but first part is to find connection types later, such as all FF_E
                                if pop_i == pop_j:
                                    conntype = 'recurrent_' + syntype + '_' + pop_i    # AMPA + NMDA feedback connections
                                else:
                                    conntype = 'FF_' + syntype + '_' + pop_i + '->' + pop_j    # AMPA + NMDA feedforward connections

                                # Create connParams dict (TODO: check whether 'weight' below overrides *.mod file)
                                Y_projectParams.update({ conntype : {
                                    'preConds': {'pop': pop_i}, 'postConds': {'pop': pop_j},
                                    'connFunc' : 'traubCellConn',
                                    'numPreToPost' : self.num_conns[pre_id][post_id],
                                    # 'weight': [self.Y_synapseMechParams['AMPA_' + pop_i + '_to_' + pop_j]['g'],     # weight is unitary conductance here
                                    #            self.Y_synapseMechParams['NMDA_' + pop_i + '_to_' + pop_j]['g']],
                                    'delay': 0,             # ignore axonal conduction delays within column (NOTE: defaults to 1 if omitted)
                                    'sec': secs,
                                    'synMech' : ['AMPA_' + pop_i + '_to_' + pop_j, 'NMDA_' + pop_i + '_to_' + pop_j],       # makes 2 synapses per sec
                                    }
                                })

                            else:

                                # Name connection label
                                # NOTE: label used to create connection but first part is to find connection types later, such as all FF_E
                                if pop_i == pop_j:
                                    conntype = 'recurrent_' + syntype + '_' + pop_i    # GABA feedback connections
                                else:
                                    conntype= 'FF_' + syntype + '_' + pop_i + '->' + pop_j    # GABA feedforward connections



                                # Create connParams dict
                                Y_projectParams.update({ conntype : {
                                    'preConds': {'pop': pop_i}, 'postConds': {'pop': pop_j},
                                    'connFunc' : 'traubCellConn',
                                    'numPreToPost' : self.num_conns[pre_id][post_id],
                                    # 'weight': self.Y_synapseMechParams['GABA_' + pop_i + '_to_' + pop_j]['g'],
                                    'delay': 0,         # ignore axonal conduction delays within column (NOTE: defaults to 1 if omitted)
                                    'sec': secs,
                                    'synMech' : 'GABA_' + pop_i + '_to_' + pop_j,
                                    }
                                })


                        ###################################
                        #  SPECIAL CASES TO OVERWRITE
                        ###################################
                        # NOTE: Update special cases as needed. Easier to do statically than with if-elif-else conditionals

                        # Special case 1
                        # if conntype == 'recurrent_E_L23_RS_PYR':
                        #     Y_projectParams['recurrent_E_L23_PYR_RS']['delay'] = '6*exp(-dist_3D/probLengthConst)'


        return Y_projectParams



if __name__ == '__main__':

    fullParams = PopulationParams()

    # Used to reference and checking on appropriate formatting of data structures
    # Will not print when imported using run file
    print '\nFull Parameter Dictionary: fullParams\n'
    print dir(fullParams)
