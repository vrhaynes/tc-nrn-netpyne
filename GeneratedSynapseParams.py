'''
    GeneratedSynapseParams.py contains parameter values for time constants, conductances,
    and permitted compartment locations for chemical (i.e., AMPA, NMDA, and GABA)
    and electrical (i.e., gap junctions) synapses.

    Contributors: Vergil R. Haynes, vrhaynes.tech@gmail.com
'''

# Import modules
import numpy as np
from numpy import arange

################################################################################
##### SYNAPTIC CONDUCTANCE TIME CONSTANTS (in ms)
################################################################################
tau_syn = dict(

    ####################################
    #                                  #
    #      superficial RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

          tauAMPA_L23PyrRS_to_L23PyrRS  =2.e0,
          tauNMDA_L23PyrRS_to_L23PyrRS  =130.5e0,
          tauAMPA_L23PyrRS_to_L23PyrFRB_varInit =2.e0,
          tauNMDA_L23PyrRS_to_L23PyrFRB_varInit =130.e0,
          tauAMPA_L23PyrRS_to_SupBasket   =.8e0,
          tauNMDA_L23PyrRS_to_SupBasket   =100.e0,
          tauAMPA_L23PyrRS_to_SupAxAx   =.8e0,
          tauNMDA_L23PyrRS_to_SupAxAx   =100.e0,
          tauAMPA_L23PyrRS_to_SupLTSInter    =1.e0,
          tauNMDA_L23PyrRS_to_SupLTSInter    =100.e0,
          tauAMPA_L23PyrRS_to_L4SpinyStellate =2.e0,
          tauNMDA_L23PyrRS_to_L4SpinyStellate =130.e0,
          tauAMPA_L23PyrRS_to_L5TuftedPyrIB    =2.e0,
          tauNMDA_L23PyrRS_to_L5TuftedPyrIB    =130.e0,
          tauAMPA_L23PyrRS_to_L5TuftedPyrRS    =2.e0,
          tauNMDA_L23PyrRS_to_L5TuftedPyrRS    =130.e0,
          tauAMPA_L23PyrRS_to_DeepBasket  =.8e0,
          tauNMDA_L23PyrRS_to_DeepBasket  =100.e0,
          tauAMPA_L23PyrRS_to_DeepAxAx  =.8e0,
          tauNMDA_L23PyrRS_to_DeepAxAx  =100.e0,
          tauAMPA_L23PyrRS_to_DeepLTSInter   =1.e0,
          tauNMDA_L23PyrRS_to_DeepLTSInter   =100.e0,
          tauAMPA_L23PyrRS_to_L6NonTuftedPyrRS =2.e0,
          tauNMDA_L23PyrRS_to_L6NonTuftedPyrRS =130.e0,

    ####################################
    #                                  #
    #     superficial FRB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

          tauAMPA_L23PyrFRB_varInit_to_L23PyrRS     =2.e0,
          tauNMDA_L23PyrFRB_varInit_to_L23PyrRS     =130.e0,
          tauAMPA_L23PyrFRB_varInit_to_L23PyrFRB_varInit    =2.e0,
          tauNMDA_L23PyrFRB_varInit_to_L23PyrFRB_varInit    =130.e0,
          tauAMPA_L23PyrFRB_varInit_to_SupBasket      =.8e0,
          tauNMDA_L23PyrFRB_varInit_to_SupBasket      =100.e0,
          tauAMPA_L23PyrFRB_varInit_to_SupAxAx      =.8e0,
          tauNMDA_L23PyrFRB_varInit_to_SupAxAx      =100.e0,
          tauAMPA_L23PyrFRB_varInit_to_SupLTSInter       =1.e0,
          tauNMDA_L23PyrFRB_varInit_to_SupLTSInter       =100.e0,
          tauAMPA_L23PyrFRB_varInit_to_L4SpinyStellate    =2.e0,
          tauNMDA_L23PyrFRB_varInit_to_L4SpinyStellate    =130.e0,
          tauAMPA_L23PyrFRB_varInit_to_L5TuftedPyrIB       =2.e0,
          tauNMDA_L23PyrFRB_varInit_to_L5TuftedPyrIB       =130.e0,
          tauAMPA_L23PyrFRB_varInit_to_L5TuftedPyrRS       =2.e0,
          tauNMDA_L23PyrFRB_varInit_to_L5TuftedPyrRS       =130.e0,
          tauAMPA_L23PyrFRB_varInit_to_DeepBasket     =.8e0,
          tauNMDA_L23PyrFRB_varInit_to_DeepBasket     =100.e0,
          tauAMPA_L23PyrFRB_varInit_to_DeepAxAx     =.8e0,
          tauNMDA_L23PyrFRB_varInit_to_DeepAxAx     =100.e0,
          tauAMPA_L23PyrFRB_varInit_to_DeepLTSInter      =1.e0,
          tauNMDA_L23PyrFRB_varInit_to_DeepLTSInter      =100.e0,
          tauAMPA_L23PyrFRB_varInit_to_L6NonTuftedPyrRS    =2.e0,
          tauNMDA_L23PyrFRB_varInit_to_L6NonTuftedPyrRS    =130.e0,

    ####################################
    #                                  #
    #     sup Basketet FS interneurons   #
    #               (GABA)             #
    #                                  #
    ####################################

          tauGABA_SupBasket_to_L23PyrRS   =6.e0,
          tauGABA_SupBasket_to_L23PyrFRB_varInit  =6.e0,
          tauGABA_SupBasket_to_SupBasket    =3.e0,
          tauGABA_SupBasket_to_SupAxAx    =3.e0,
          tauGABA_SupBasket_to_SupLTSInter     =3.e0,
          tauGABA_SupBasket_to_L4SpinyStellate  =6.e0,

    ####################################
    #                                  #
    #   sup axoaxonic FS internerons   #
    #              (GABA)              #
    #                                  #
    ####################################

          tauGABA_SupAxAx_to_L23PyrRS   =6.e0,
          tauGABA_SupAxAx_to_L23PyrFRB_varInit  =6.e0,
          tauGABA_SupAxAx_to_L4SpinyStellate  =6.e0,
          tauGABA_SupAxAx_to_L5TuftedPyrIB     =6.e0,
          tauGABA_SupAxAx_to_L5TuftedPyrRS     =6.e0,
          tauGABA_SupAxAx_to_L6NonTuftedPyrRS  =6.e0,

    ####################################
    #                                  #
    #  superficial LTS interneurons    #
    #             (GABA)               #
    #                                  #
    ####################################

          tauGABA_SupLTSInter_to_L23PyrRS    =20.e0,
          tauGABA_SupLTSInter_to_L23PyrFRB_varInit   =20.e0,
          tauGABA_SupLTSInter_to_SupBasket     =20.e0,
          tauGABA_SupLTSInter_to_SupAxAx     =20.e0,
          tauGABA_SupLTSInter_to_SupLTSInter      =20.e0,
          tauGABA_SupLTSInter_to_L4SpinyStellate   =20.e0,
          tauGABA_SupLTSInter_to_L5TuftedPyrIB      =20.e0,
          tauGABA_SupLTSInter_to_L5TuftedPyrRS      =20.e0,
          tauGABA_SupLTSInter_to_DeepBasket    =20.e0,
          tauGABA_SupLTSInter_to_DeepAxAx    =20.e0,
          tauGABA_SupLTSInter_to_DeepLTSInter     =20.e0,
          tauGABA_SupLTSInter_to_L6NonTuftedPyrRS   =20.e0,

    ####################################
    #                                  #
    #      spiny stellate cells        #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

          tauAMPA_L4SpinyStellate_to_L23PyrRS     =2.e0,
          tauNMDA_L4SpinyStellate_to_L23PyrRS     =130.e0,
          tauAMPA_L4SpinyStellate_to_L23PyrFRB_varInit    =2.e0,
          tauNMDA_L4SpinyStellate_to_L23PyrFRB_varInit    =130.e0,
          tauAMPA_L4SpinyStellate_to_SupBasket      =.8e0,
          tauNMDA_L4SpinyStellate_to_SupBasket      =100.e0,
          tauAMPA_L4SpinyStellate_to_SupAxAx      =.8e0,
          tauNMDA_L4SpinyStellate_to_SupAxAx      =100.e0,
          tauAMPA_L4SpinyStellate_to_SupLTSInter       =1.e0,
          tauNMDA_L4SpinyStellate_to_SupLTSInter       =100.e0,
          tauAMPA_L4SpinyStellate_to_L4SpinyStellate    =2.e0,
          tauNMDA_L4SpinyStellate_to_L4SpinyStellate    =130.e0,
          tauAMPA_L4SpinyStellate_to_L5TuftedPyrIB       =2.e0,
          tauNMDA_L4SpinyStellate_to_L5TuftedPyrIB       =130.e0,
          tauAMPA_L4SpinyStellate_to_L5TuftedPyrRS       =2.e0,
          tauNMDA_L4SpinyStellate_to_L5TuftedPyrRS       =130.e0,
          tauAMPA_L4SpinyStellate_to_DeepBasket     =.8e0,
          tauNMDA_L4SpinyStellate_to_DeepBasket     =100.e0,
          tauAMPA_L4SpinyStellate_to_DeepAxAx     =.8e0,
          tauNMDA_L4SpinyStellate_to_DeepAxAx     =100.e0,
          tauAMPA_L4SpinyStellate_to_DeepLTSInter      =1.e0,
          tauNMDA_L4SpinyStellate_to_DeepLTSInter      =100.e0,
          tauAMPA_L4SpinyStellate_to_L6NonTuftedPyrRS    =2.e0,
          tauNMDA_L4SpinyStellate_to_L6NonTuftedPyrRS    =130.e0,

    ####################################
    #                                  #
    #      deep tufted IB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

          tauAMPA_L5TuftedPyrIB_to_L23PyrRS    =2.e0,
          tauNMDA_L5TuftedPyrIB_to_L23PyrRS    =130.e0,
          tauAMPA_L5TuftedPyrIB_to_L23PyrFRB_varInit   =2.e0,
          tauNMDA_L5TuftedPyrIB_to_L23PyrFRB_varInit   =130.e0,
          tauAMPA_L5TuftedPyrIB_to_SupBasket     =.8e0,
          tauNMDA_L5TuftedPyrIB_to_SupBasket     =100.e0,
          tauAMPA_L5TuftedPyrIB_to_SupAxAx     =.8e0,
          tauNMDA_L5TuftedPyrIB_to_SupAxAx     =100.e0,
          tauAMPA_L5TuftedPyrIB_to_SupLTSInter      =1.e0,
          tauNMDA_L5TuftedPyrIB_to_SupLTSInter      =100.e0,
          tauAMPA_L5TuftedPyrIB_to_L4SpinyStellate   =2.e0,
          tauNMDA_L5TuftedPyrIB_to_L4SpinyStellate   =130.e0,
          tauAMPA_L5TuftedPyrIB_to_L5TuftedPyrIB      =2.e0,
          tauNMDA_L5TuftedPyrIB_to_L5TuftedPyrIB      =130.e0,
          tauAMPA_L5TuftedPyrIB_to_L5TuftedPyrRS      =2.0e0,
          tauNMDA_L5TuftedPyrIB_to_L5TuftedPyrRS      =130.e0,
          tauAMPA_L5TuftedPyrIB_to_DeepBasket    =.8e0,
          tauNMDA_L5TuftedPyrIB_to_DeepBasket    =100.e0,
          tauAMPA_L5TuftedPyrIB_to_DeepAxAx    =.8e0,
          tauNMDA_L5TuftedPyrIB_to_DeepAxAx    =100.e0,
          tauAMPA_L5TuftedPyrIB_to_DeepLTSInter     =1.e0,
          tauNMDA_L5TuftedPyrIB_to_DeepLTSInter     =100.e0,
          tauAMPA_L5TuftedPyrIB_to_L6NonTuftedPyrRS   =2.0e0,
          tauNMDA_L5TuftedPyrIB_to_L6NonTuftedPyrRS   =130.e0,

    ####################################
    #                                  #
    #      deep tufted RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

          tauAMPA_L5TuftedPyrRS_to_L23PyrRS    =2.e0,
          tauNMDA_L5TuftedPyrRS_to_L23PyrRS    =130.e0,
          tauAMPA_L5TuftedPyrRS_to_L23PyrFRB_varInit   =2.e0,
          tauNMDA_L5TuftedPyrRS_to_L23PyrFRB_varInit   =130.e0,
          tauAMPA_L5TuftedPyrRS_to_SupBasket     =.8e0,
          tauNMDA_L5TuftedPyrRS_to_SupBasket     =100.e0,
          tauAMPA_L5TuftedPyrRS_to_SupAxAx     =.8e0,
          tauNMDA_L5TuftedPyrRS_to_SupAxAx     =100.e0,
          tauAMPA_L5TuftedPyrRS_to_SupLTSInter      =1.e0,
          tauNMDA_L5TuftedPyrRS_to_SupLTSInter      =100.e0,
          tauAMPA_L5TuftedPyrRS_to_L4SpinyStellate   =2.e0,
          tauNMDA_L5TuftedPyrRS_to_L4SpinyStellate   =130.e0,
          tauAMPA_L5TuftedPyrRS_to_L5TuftedPyrIB      =2.e0,
          tauNMDA_L5TuftedPyrRS_to_L5TuftedPyrIB      =130.e0,
          tauAMPA_L5TuftedPyrRS_to_L5TuftedPyrRS      =2.e0,
          tauNMDA_L5TuftedPyrRS_to_L5TuftedPyrRS      =130.e0,
          tauAMPA_L5TuftedPyrRS_to_DeepBasket    =.8e0,
          tauNMDA_L5TuftedPyrRS_to_DeepBasket    =100.e0,
          tauAMPA_L5TuftedPyrRS_to_DeepAxAx    =.8e0,
          tauNMDA_L5TuftedPyrRS_to_DeepAxAx    =100.e0,
          tauAMPA_L5TuftedPyrRS_to_DeepLTSInter     =1.e0,
          tauNMDA_L5TuftedPyrRS_to_DeepLTSInter     =100.e0,
          tauAMPA_L5TuftedPyrRS_to_L6NonTuftedPyrRS   =2.e0,
          tauNMDA_L5TuftedPyrRS_to_L6NonTuftedPyrRS   =130.e0,

    ####################################
    #                                  #
    #    deep Basketet FS interneurons   #
    #              (GABA)              #
    #                                  #
    ####################################

          tauGABA_DeepBasket_to_L4SpinyStellate =6.e0,
          tauGABA_DeepBasket_to_L5TuftedPyrIB    =6.e0,
          tauGABA_DeepBasket_to_L5TuftedPyrRS    =6.e0,
          tauGABA_DeepBasket_to_DeepBasket  =3.e0,
          tauGABA_DeepBasket_to_DeepAxAx  =3.e0,
          tauGABA_DeepBasket_to_DeepLTSInter   =3.e0,
          tauGABA_DeepBasket_to_L6NonTuftedPyrRS =6.e0,

    ####################################
    #                                  #
    #  deep axoaxonic FS interneurons  #
    #              (GABA)              #
    #                                  #
    ####################################

          tauGABA_DeepAxAx_to_L23PyrRS   =6.e0,
          tauGABA_DeepAxAx_to_L23PyrFRB_varInit  =6.e0,
          tauGABA_DeepAxAx_to_L4SpinyStellate  =6.e0,
          tauGABA_DeepAxAx_to_L5TuftedPyrIB     =6.e0,
          tauGABA_DeepAxAx_to_L5TuftedPyrRS     =6.e0,
          tauGABA_DeepAxAx_to_L6NonTuftedPyrRS  =6.e0,

    ####################################
    #                                  #
    #       deep LTS interneurons      #
    #              (GABA)              #
    #                                  #
    ####################################

          tauGABA_DeepLTSInter_to_L23PyrRS    =20.e0,
          tauGABA_DeepLTSInter_to_L23PyrFRB_varInit   =20.e0,
          tauGABA_DeepLTSInter_to_SupBasket     =20.e0,
          tauGABA_DeepLTSInter_to_SupAxAx     =20.e0,
          tauGABA_DeepLTSInter_to_SupLTSInter      =20.e0,
          tauGABA_DeepLTSInter_to_L4SpinyStellate   =20.e0,
          tauGABA_DeepLTSInter_to_L5TuftedPyrIB      =20.e0,
          tauGABA_DeepLTSInter_to_L5TuftedPyrRS      =20.e0,
          tauGABA_DeepLTSInter_to_DeepBasket    =20.e0,
          tauGABA_DeepLTSInter_to_DeepAxAx    =20.e0,
          tauGABA_DeepLTSInter_to_DeepLTSInter     =20.e0,
          tauGABA_DeepLTSInter_to_L6NonTuftedPyrRS   =20.e0,

    ####################################
    #                                  #
    #      thalamic relay neurons      #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

          tauAMPA_TCR_to_L23PyrRS        =2.e0,
          tauNMDA_TCR_to_L23PyrRS        =130.e0,
          tauAMPA_TCR_to_L23PyrFRB_varInit       =2.e0,
          tauNMDA_TCR_to_L23PyrFRB_varInit       =130.e0,
          tauAMPA_TCR_to_SupBasket         =1.e0,
          tauNMDA_TCR_to_SupBasket         =100.e0,
          tauAMPA_TCR_to_SupAxAx         =1.e0,
          tauNMDA_TCR_to_SupAxAx         =100.e0,
          tauAMPA_TCR_to_L4SpinyStellate       =2.0e0,
          tauNMDA_TCR_to_L4SpinyStellate       =130.e0,
          tauAMPA_TCR_to_L5TuftedPyrIB          =2.e0,
          tauNMDA_TCR_to_L5TuftedPyrIB          =130.e0,
          tauAMPA_TCR_to_L5TuftedPyrRS          =2.e0,
          tauNMDA_TCR_to_L5TuftedPyrRS          =130.e0,
          tauAMPA_TCR_to_DeepBasket        =1.e0,
          tauNMDA_TCR_to_DeepBasket        =100.e0,
          tauAMPA_TCR_to_DeepAxAx        =1.e0,
          tauNMDA_TCR_to_DeepAxAx        =100.e0,
          tauAMPA_TCR_to_nRT             =2.0e0,
          tauNMDA_TCR_to_nRT             =150.e0,
          tauAMPA_TCR_to_L6NonTuftedPyrRS       =2.0e0,
          tauNMDA_TCR_to_L6NonTuftedPyrRS       =130.e0,

    ####################################
    #                                  #
    #           nRT neurons            #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

#          tauGABA1_nRT_to_TCR             =10.e0,
#          tauGABA2_nRT_to_TCR             =30.e0,
#          tauGABA1_nRT_to_nRT             =18.e0,
#          tauGABA2_nRT_to_nRT             =89.e0,
#   See notebook entry of 17 Feb. 2004.
#   Speed these up per Huntsman & Huguenard (2000)
          tauGABA1_nRT_to_TCR             =3.30e0,
          tauGABA2_nRT_to_TCR             =10.e0,
          tauGABA1_nRT_to_nRT             = 9.e0,
          tauGABA2_nRT_to_nRT             =44.5e0,

    ####################################
    #                                  #
    #      deep nontuft RS pyramids    #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

          tauAMPA_L6NonTuftedPyrRS_to_L23PyrRS  =2.e0,
          tauNMDA_L6NonTuftedPyrRS_to_L23PyrRS  =130.e0,
          tauAMPA_L6NonTuftedPyrRS_to_L23PyrFRB_varInit =2.e0,
          tauNMDA_L6NonTuftedPyrRS_to_L23PyrFRB_varInit =130.e0,
          tauAMPA_L6NonTuftedPyrRS_to_SupBasket   =.8e0,
          tauNMDA_L6NonTuftedPyrRS_to_SupBasket   =100.e0,
          tauAMPA_L6NonTuftedPyrRS_to_SupAxAx   =.8e0,
          tauNMDA_L6NonTuftedPyrRS_to_SupAxAx   =100.e0,
          tauAMPA_L6NonTuftedPyrRS_to_SupLTSInter    =1.0e0,
          tauNMDA_L6NonTuftedPyrRS_to_SupLTSInter    =100.e0,
          tauAMPA_L6NonTuftedPyrRS_to_L4SpinyStellate =2.e0,
          tauNMDA_L6NonTuftedPyrRS_to_L4SpinyStellate =130.e0,
          tauAMPA_L6NonTuftedPyrRS_to_L5TuftedPyrIB    =2.e0,
          tauNMDA_L6NonTuftedPyrRS_to_L5TuftedPyrIB    =130.e0,
          tauAMPA_L6NonTuftedPyrRS_to_L5TuftedPyrRS    =2.e0,
          tauNMDA_L6NonTuftedPyrRS_to_L5TuftedPyrRS    =130.e0,
          tauAMPA_L6NonTuftedPyrRS_to_DeepBasket  =.8e0,
          tauNMDA_L6NonTuftedPyrRS_to_DeepBasket  =100.e0,
          tauAMPA_L6NonTuftedPyrRS_to_DeepAxAx  =.8e0,
          tauNMDA_L6NonTuftedPyrRS_to_DeepAxAx  =100.e0,
          tauAMPA_L6NonTuftedPyrRS_to_DeepLTSInter   =1.e0,
          tauNMDA_L6NonTuftedPyrRS_to_DeepLTSInter   =100.e0,
          tauAMPA_L6NonTuftedPyrRS_to_TCR       =2.e0,
          tauNMDA_L6NonTuftedPyrRS_to_TCR       =130.e0,
          tauAMPA_L6NonTuftedPyrRS_to_nRT       =2.0e0,
          tauNMDA_L6NonTuftedPyrRS_to_nRT       =100.e0,
          tauAMPA_L6NonTuftedPyrRS_to_L6NonTuftedPyrRS =2.e0,
          tauNMDA_L6NonTuftedPyrRS_to_L6NonTuftedPyrRS =130.e0

# End definition of synaptic time constants
)



################################################################################
##### SYNAPTIC UNITARY CONDUCTANCE (in nS)
################################################################################
g_syn = dict(

    ####################################
    #                                  #
    #      superficial RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

       gAMPA_L23PyrRS_to_L23PyrRS   =0.25e-3,
       gNMDA_L23PyrRS_to_L23PyrRS   =0.025e-3,
       gAMPA_L23PyrRS_to_L23PyrFRB_varInit  =0.25e-3,
       gNMDA_L23PyrRS_to_L23PyrFRB_varInit  =0.025e-3,
       gAMPA_L23PyrRS_to_SupBasket    =3.00e-3,
       gNMDA_L23PyrRS_to_SupBasket    =0.15e-3,
       gAMPA_L23PyrRS_to_SupAxAx    =3.0e-3,
       gNMDA_L23PyrRS_to_SupAxAx    =0.15e-3,
       gAMPA_L23PyrRS_to_SupLTSInter     =2.0e-3,
       gNMDA_L23PyrRS_to_SupLTSInter     =0.15e-3,
       gAMPA_L23PyrRS_to_L4SpinyStellate  =0.10e-3,
       gNMDA_L23PyrRS_to_L4SpinyStellate  =0.01e-3,
       gAMPA_L23PyrRS_to_L5TuftedPyrIB     =0.10e-3,
       gNMDA_L23PyrRS_to_L5TuftedPyrIB     =0.01e-3,
       gAMPA_L23PyrRS_to_L5TuftedPyrRS     =0.10e-3,
       gNMDA_L23PyrRS_to_L5TuftedPyrRS     =0.01e-3,
       gAMPA_L23PyrRS_to_DeepBasket   =1.00e-3,
       gNMDA_L23PyrRS_to_DeepBasket   =0.10e-3,
       gAMPA_L23PyrRS_to_DeepAxAx   =1.00e-3,
       gNMDA_L23PyrRS_to_DeepAxAx   =0.10e-3,
       gAMPA_L23PyrRS_to_DeepLTSInter    =1.00e-3,
       gNMDA_L23PyrRS_to_DeepLTSInter    =0.15e-3,
       gAMPA_L23PyrRS_to_L6NonTuftedPyrRS  =0.50e-3,
       gNMDA_L23PyrRS_to_L6NonTuftedPyrRS  =0.05e-3,

    ####################################
    #                                  #
    #     superficial FRB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

       gAMPA_L23PyrFRB_varInit_to_L23PyrRS  =0.25e-3,
       gNMDA_L23PyrFRB_varInit_to_L23PyrRS  =0.025e-3,
       gAMPA_L23PyrFRB_varInit_to_L23PyrFRB_varInit =0.25e-3,
       gNMDA_L23PyrFRB_varInit_to_L23PyrFRB_varInit =.025e-3,
       gAMPA_L23PyrFRB_varInit_to_SupBasket   =3.00e-3,
       gNMDA_L23PyrFRB_varInit_to_SupBasket   =0.10e-3,
       gAMPA_L23PyrFRB_varInit_to_SupAxAx   =3.0e-3,
       gNMDA_L23PyrFRB_varInit_to_SupAxAx   =0.10e-3,
       gAMPA_L23PyrFRB_varInit_to_SupLTSInter    =2.0e-3,
       gNMDA_L23PyrFRB_varInit_to_SupLTSInter    =0.10e-3,
       gAMPA_L23PyrFRB_varInit_to_L4SpinyStellate = 0.10e-3,
       gNMDA_L23PyrFRB_varInit_to_L4SpinyStellate = 0.01e-3,
       gAMPA_L23PyrFRB_varInit_to_L5TuftedPyrIB    =0.10e-3,
       gNMDA_L23PyrFRB_varInit_to_L5TuftedPyrIB    =0.01e-3,
       gAMPA_L23PyrFRB_varInit_to_L5TuftedPyrRS    =0.10e-3,
       gNMDA_L23PyrFRB_varInit_to_L5TuftedPyrRS    =0.01e-3,
       gAMPA_L23PyrFRB_varInit_to_DeepBasket  =1.00e-3,
       gNMDA_L23PyrFRB_varInit_to_DeepBasket  =0.10e-3,
       gAMPA_L23PyrFRB_varInit_to_DeepAxAx  =1.00e-3,
       gNMDA_L23PyrFRB_varInit_to_DeepAxAx  =0.10e-3,
       gAMPA_L23PyrFRB_varInit_to_DeepLTSInter   =1.00e-3,
       gNMDA_L23PyrFRB_varInit_to_DeepLTSInter   =0.10e-3,
       gAMPA_L23PyrFRB_varInit_to_L6NonTuftedPyrRS =0.50e-3,
       gNMDA_L23PyrFRB_varInit_to_L6NonTuftedPyrRS =0.05e-3,

    ####################################
    #                                  #
    #    sup Basketet FS internerons     #
    #              (GABA)              #
    #                                  #
    ####################################

       gGABA_SupBasket_to_L23PyrRS   =1.2e-3,
       gGABA_SupBasket_to_L23PyrFRB_varInit  =1.2e-3,
       gGABA_SupBasket_to_SupBasket    =0.2e-3,
       gGABA_SupBasket_to_SupAxAx    =0.2e-3,
       gGABA_SupBasket_to_SupLTSInter     =0.5e-3,
#       gGABA_SupBasket_to_L4SpinyStellate  =0.7e-3
       gGABA_SupBasket_to_L4SpinyStellate  =0.1e-3,      # if main inhib. to L4SpinyStellate from deep int.

    ####################################
    #                                  #
    #   sup axoaxonic FS internerons   #
    #              (GABA)              #
    #                                  #
    ####################################

       gGABA_SupAxAx_to_L23PyrRS   =1.2e-3,
       gGABA_SupAxAx_to_L23PyrFRB_varInit  =1.2e-3,
#       gGABA_SupAxAx_to_L4SpinyStellate  =1.0e-3
       gGABA_SupAxAx_to_L4SpinyStellate  =0.1e-3,      # if main inhib. to L4SpinyStellate from deep int.
       gGABA_SupAxAx_to_L5TuftedPyrIB     =1.0e-3,
       gGABA_SupAxAx_to_L5TuftedPyrRS     =1.0e-3,
       gGABA_SupAxAx_to_L6NonTuftedPyrRS  =1.0e-3,

    ####################################
    #                                  #
    #  superficial LTS interneurons    #
    #             (GABA)               #
    #                                  #
    ####################################

       gGABA_SupLTSInter_to_L23PyrRS    =.01e-3,
       gGABA_SupLTSInter_to_L23PyrFRB_varInit   =.01e-3,
       gGABA_SupLTSInter_to_SupBasket     =.01e-3,
       gGABA_SupLTSInter_to_SupAxAx     =.01e-3,
       gGABA_SupLTSInter_to_SupLTSInter      =.05e-3,
       gGABA_SupLTSInter_to_L4SpinyStellate   =.01e-3,
       gGABA_SupLTSInter_to_L5TuftedPyrIB      =.02e-3,
       gGABA_SupLTSInter_to_L5TuftedPyrRS      =.02e-3,
       gGABA_SupLTSInter_to_DeepBasket    =.01e-3,
       gGABA_SupLTSInter_to_DeepAxAx    =.01e-3,
       gGABA_SupLTSInter_to_DeepLTSInter     =.05e-3,
       gGABA_SupLTSInter_to_L6NonTuftedPyrRS   =.01e-3,

    ####################################
    #                                  #
    #      spiny stellate cells        #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

       gAMPA_L4SpinyStellate_to_L23PyrRS  =1.0e-3,
       gNMDA_L4SpinyStellate_to_L23PyrRS  =0.1e-3,
       gAMPA_L4SpinyStellate_to_L23PyrFRB_varInit = 1.0e-3,
       gNMDA_L4SpinyStellate_to_L23PyrFRB_varInit =0.1e-3,
       gAMPA_L4SpinyStellate_to_SupBasket   =1.0e-3,
       gNMDA_L4SpinyStellate_to_SupBasket   =.15e-3,
       gAMPA_L4SpinyStellate_to_SupAxAx   =1.0e-3,
       gNMDA_L4SpinyStellate_to_SupAxAx   =.15e-3,
       gAMPA_L4SpinyStellate_to_SupLTSInter    =1.0e-3,
       gNMDA_L4SpinyStellate_to_SupLTSInter    =.15e-3,
       gAMPA_L4SpinyStellate_to_L4SpinyStellate =1.0e-3,
       gNMDA_L4SpinyStellate_to_L4SpinyStellate =0.1e-3,
       gAMPA_L4SpinyStellate_to_L5TuftedPyrIB    =1.0e-3,
       gNMDA_L4SpinyStellate_to_L5TuftedPyrIB    =0.1e-3,
       gAMPA_L4SpinyStellate_to_L5TuftedPyrRS    =1.0e-3,
       gNMDA_L4SpinyStellate_to_L5TuftedPyrRS    =0.1e-3,
       gAMPA_L4SpinyStellate_to_DeepBasket  =1.0e-3,
       gNMDA_L4SpinyStellate_to_DeepBasket  =.15e-3,
       gAMPA_L4SpinyStellate_to_DeepAxAx  =1.0e-3,
       gNMDA_L4SpinyStellate_to_DeepAxAx  =.15e-3,
       gAMPA_L4SpinyStellate_to_DeepLTSInter   =1.0e-3,
       gNMDA_L4SpinyStellate_to_DeepLTSInter   =.15e-3,
       gAMPA_L4SpinyStellate_to_L6NonTuftedPyrRS =1.0e-3,
       gNMDA_L4SpinyStellate_to_L6NonTuftedPyrRS =0.1e-3,

    ####################################
    #                                  #
    #      deep tufted IB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

       gAMPA_L5TuftedPyrIB_to_L23PyrRS    =0.5e-3,
       gNMDA_L5TuftedPyrIB_to_L23PyrRS    =0.05e-3,
       gAMPA_L5TuftedPyrIB_to_L23PyrFRB_varInit   =0.5e-3,
       gNMDA_L5TuftedPyrIB_to_L23PyrFRB_varInit   =0.05e-3,
       gAMPA_L5TuftedPyrIB_to_SupBasket     =1.0e-3,
       gNMDA_L5TuftedPyrIB_to_SupBasket     =0.15e-3,
       gAMPA_L5TuftedPyrIB_to_SupAxAx     =1.0e-3,
       gNMDA_L5TuftedPyrIB_to_SupAxAx     =0.15e-3,
       gAMPA_L5TuftedPyrIB_to_SupLTSInter      =1.0e-3,
       gNMDA_L5TuftedPyrIB_to_SupLTSInter      =0.15e-3,
       gAMPA_L5TuftedPyrIB_to_L4SpinyStellate   =0.5e-3,
       gNMDA_L5TuftedPyrIB_to_L4SpinyStellate   =0.05e-3,
       gAMPA_L5TuftedPyrIB_to_L5TuftedPyrIB      =2.0e-3,
       gNMDA_L5TuftedPyrIB_to_L5TuftedPyrIB      =0.20e-3,
       gAMPA_L5TuftedPyrIB_to_L5TuftedPyrRS      =2.0e-3,
       gNMDA_L5TuftedPyrIB_to_L5TuftedPyrRS      =0.20e-3,
       gAMPA_L5TuftedPyrIB_to_DeepBasket    =3.0e-3,
       gNMDA_L5TuftedPyrIB_to_DeepBasket    =0.15e-3,
       gAMPA_L5TuftedPyrIB_to_DeepAxAx    =3.0e-3,
       gNMDA_L5TuftedPyrIB_to_DeepAxAx    =0.15e-3,
       gAMPA_L5TuftedPyrIB_to_DeepLTSInter     =2.0e-3,
       gNMDA_L5TuftedPyrIB_to_DeepLTSInter     =0.15e-3,
       gAMPA_L5TuftedPyrIB_to_L6NonTuftedPyrRS   =2.0e-3,
       gNMDA_L5TuftedPyrIB_to_L6NonTuftedPyrRS   =0.20e-3,

    ####################################
    #                                  #
    #      deep tufted RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

       gAMPA_L5TuftedPyrRS_to_L23PyrRS    =0.5e-3,
       gNMDA_L5TuftedPyrRS_to_L23PyrRS    =0.05e-3,
       gAMPA_L5TuftedPyrRS_to_L23PyrFRB_varInit   =0.5e-3,
       gNMDA_L5TuftedPyrRS_to_L23PyrFRB_varInit   =0.05e-3,
       gAMPA_L5TuftedPyrRS_to_SupBasket     =1.0e-3,
       gNMDA_L5TuftedPyrRS_to_SupBasket     =0.15e-3,
       gAMPA_L5TuftedPyrRS_to_SupAxAx     =1.0e-3,
       gNMDA_L5TuftedPyrRS_to_SupAxAx     =0.15e-3,
       gAMPA_L5TuftedPyrRS_to_SupLTSInter      =1.0e-3,
       gNMDA_L5TuftedPyrRS_to_SupLTSInter      =0.15e-3,
       gAMPA_L5TuftedPyrRS_to_L4SpinyStellate   =0.5e-3,
       gNMDA_L5TuftedPyrRS_to_L4SpinyStellate   =0.05e-3,
       gAMPA_L5TuftedPyrRS_to_L5TuftedPyrIB      =1.0e-3,
       gNMDA_L5TuftedPyrRS_to_L5TuftedPyrIB      =0.10e-3,
       gAMPA_L5TuftedPyrRS_to_L5TuftedPyrRS      =1.0e-3,
       gNMDA_L5TuftedPyrRS_to_L5TuftedPyrRS      =0.10e-3,
       gAMPA_L5TuftedPyrRS_to_DeepBasket    =3.0e-3,
       gNMDA_L5TuftedPyrRS_to_DeepBasket    =0.10e-3,
       gAMPA_L5TuftedPyrRS_to_DeepAxAx    =3.0e-3,
       gNMDA_L5TuftedPyrRS_to_DeepAxAx    =0.10e-3,
       gAMPA_L5TuftedPyrRS_to_DeepLTSInter     =2.0e-3,
       gNMDA_L5TuftedPyrRS_to_DeepLTSInter     =0.10e-3,
       gAMPA_L5TuftedPyrRS_to_L6NonTuftedPyrRS   =1.0e-3,
       gNMDA_L5TuftedPyrRS_to_L6NonTuftedPyrRS   =0.10e-3,

    ####################################
    #                                  #
    #    deep Basketet FS interneurons   #
    #              (GABA)              #
    #                                  #
    ####################################

#       gGABA_DeepBasket_to_L4SpinyStellate =1.0e-3
       gGABA_DeepBasket_to_L4SpinyStellate =1.5e-3, # ? suppress spiny stellate bursts ?
       gGABA_DeepBasket_to_L5TuftedPyrIB    =0.7e-3,
       gGABA_DeepBasket_to_L5TuftedPyrRS    =0.7e-3,
       gGABA_DeepBasket_to_DeepBasket  =0.2e-3,
       gGABA_DeepBasket_to_DeepAxAx  =0.2e-3,
       gGABA_DeepBasket_to_DeepLTSInter   =0.7e-3,
       gGABA_DeepBasket_to_L6NonTuftedPyrRS =0.7e-3,

    ####################################
    #                                  #
    #  deep axoaxonic FS interneurons  #
    #              (GABA)              #
    #                                  #
    ####################################

       gGABA_DeepAxAx_to_L23PyrRS   =1.0e-3,
       gGABA_DeepAxAx_to_L23PyrFRB_varInit  =1.0e-3,
#       gGABA_DeepAxAx_to_L4SpinyStellate  =1.0e-3
       gGABA_DeepAxAx_to_L4SpinyStellate  =1.5e-3, # ? suppress spiny stellate bursts ?
       gGABA_DeepAxAx_to_L5TuftedPyrIB     =1.0e-3,
       gGABA_DeepAxAx_to_L5TuftedPyrRS     =1.0e-3,
       gGABA_DeepAxAx_to_L6NonTuftedPyrRS  =1.0e-3,

    ####################################
    #                                  #
    #       deep LTS interneurons      #
    #              (GABA)              #
    #                                  #
    ####################################

       gGABA_DeepLTSInter_to_L23PyrRS    =.01e-3,
       gGABA_DeepLTSInter_to_L23PyrFRB_varInit   =.01e-3,
       gGABA_DeepLTSInter_to_SupBasket     =.01e-3,
       gGABA_DeepLTSInter_to_SupAxAx     =.01e-3,
       gGABA_DeepLTSInter_to_SupLTSInter      =.05e-3,
       gGABA_DeepLTSInter_to_L4SpinyStellate   =.01e-3,
#       gGABA_DeepLTSInter_to_L5TuftedPyrIB      =.02e-3
       gGABA_DeepLTSInter_to_L5TuftedPyrIB      =.05e-3, # will this help suppress bursting?
       gGABA_DeepLTSInter_to_L5TuftedPyrRS      =.02e-3,
       gGABA_DeepLTSInter_to_DeepBasket    =.01e-3,
       gGABA_DeepLTSInter_to_DeepAxAx    =.01e-3,
       gGABA_DeepLTSInter_to_DeepLTSInter     =.05e-3,
       gGABA_DeepLTSInter_to_L6NonTuftedPyrRS   =.01e-3,

    ####################################
    #                                  #
    #      thalamic relay neurons      #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

       gAMPA_TCR_to_L23PyrRS        =0.5e-3,
       gNMDA_TCR_to_L23PyrRS        =0.05e-3,
       gAMPA_TCR_to_L23PyrFRB_varInit       =0.5e-3,
       gNMDA_TCR_to_L23PyrFRB_varInit       =0.05e-3,
#       gAMPA_TCR_to_SupBasket         =1.0e-3
       gAMPA_TCR_to_SupBasket         =0.1e-3,
#       gNMDA_TCR_to_SupBasket         =.10e-3
       gNMDA_TCR_to_SupBasket         =.01e-3,
#       gAMPA_TCR_to_SupAxAx         =1.0e-3
       gAMPA_TCR_to_SupAxAx         =0.1e-3,
       gNMDA_TCR_to_SupAxAx         =.01e-3,
       gAMPA_TCR_to_L4SpinyStellate       =1.0e-3,
       gNMDA_TCR_to_L4SpinyStellate       =.10e-3,
       gAMPA_TCR_to_L5TuftedPyrIB          =1.5e-3,
       gNMDA_TCR_to_L5TuftedPyrIB          =.15e-3,
       gAMPA_TCR_to_L5TuftedPyrRS          =1.5e-3,
       gNMDA_TCR_to_L5TuftedPyrRS          =.15e-3,
#       gAMPA_TCR_to_DeepBasket        =1.0e-3
       gAMPA_TCR_to_DeepBasket        =1.5e-3,
       gNMDA_TCR_to_DeepBasket        =.10e-3,
       gAMPA_TCR_to_DeepAxAx        =1.0e-3,
       gNMDA_TCR_to_DeepAxAx        =.10e-3,
       gAMPA_TCR_to_nRT             =0.75e-3,
       gNMDA_TCR_to_nRT             =.15e-3,
       gAMPA_TCR_to_L6NonTuftedPyrRS       =1.0e-3,
       gNMDA_TCR_to_L6NonTuftedPyrRS       =.10e-3,

    ####################################
    #                                  #
    #           nRT neurons            #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

#       gGABA_nRT_to_TCR             =1.0e-3
       gGABA_nRT_to_nRT             =0.30e-3,

    ####################################
    #                                  #
    #      deep RS nontuft pyramids    #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

       gAMPA_L6NonTuftedPyrRS_to_L23PyrRS  =0.5e-3,
       gNMDA_L6NonTuftedPyrRS_to_L23PyrRS  =0.05e-3,
       gAMPA_L6NonTuftedPyrRS_to_L23PyrFRB_varInit =0.5e-3,
       gNMDA_L6NonTuftedPyrRS_to_L23PyrFRB_varInit =0.05e-3,
       gAMPA_L6NonTuftedPyrRS_to_SupBasket   =1.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_SupBasket   =0.1e-3,
       gAMPA_L6NonTuftedPyrRS_to_SupAxAx   =1.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_SupAxAx   =0.1e-3,
       gAMPA_L6NonTuftedPyrRS_to_SupLTSInter    =1.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_SupLTSInter    =0.1e-3,
       gAMPA_L6NonTuftedPyrRS_to_L4SpinyStellate =0.5e-3,
       gNMDA_L6NonTuftedPyrRS_to_L4SpinyStellate =0.05e-3,
       gAMPA_L6NonTuftedPyrRS_to_L5TuftedPyrIB    =1.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_L5TuftedPyrIB    =0.1e-3,
       gAMPA_L6NonTuftedPyrRS_to_L5TuftedPyrRS    =1.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_L5TuftedPyrRS    =0.1e-3,
       gAMPA_L6NonTuftedPyrRS_to_DeepBasket  =3.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_DeepBasket  =.10e-3,
       gAMPA_L6NonTuftedPyrRS_to_DeepAxAx  =3.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_DeepAxAx  =.10e-3,
       gAMPA_L6NonTuftedPyrRS_to_DeepLTSInter   =2.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_DeepLTSInter   =.10e-3,
       gAMPA_L6NonTuftedPyrRS_to_TCR       =.75e-3,
       gNMDA_L6NonTuftedPyrRS_to_TCR       =.075e-3,
       gAMPA_L6NonTuftedPyrRS_to_nRT       =0.5e-3,
       gNMDA_L6NonTuftedPyrRS_to_nRT       =0.05e-3,
       gAMPA_L6NonTuftedPyrRS_to_L6NonTuftedPyrRS =1.0e-3,
       gNMDA_L6NonTuftedPyrRS_to_L6NonTuftedPyrRS =0.1e-3,
)


################################################################################
##### SYNAPTIC COMPARTMENT LEVEL NAMING
################################################################################
'''
Redo into just explicit list

(Not finished and not sure how useful)

'''
cellsec_comps = dict(
       L23PyrRS_Axon = 0,
       L23PyrRS_Soma = 1,
       L23PyrRS_PBO_Dends = 2,      # proximal basal and oblique dendrites
       L23PyrRS_MBO_Dends = 3,      # middle basal and oblique dendrites
       L23PyrRS_DBO_Dends = 4,      # distal basal and oblique dendrites
       L23PyrRS_DA_Dends = arange(5,12+1),    # progressively more distal apical dendrites

       L23PyrFRB_varInit_Axon = 0,
       L23PyrFRB_varInit_Soma = 1,
       L23PyrFRB_varInit_PBO_Dends = 2,      # proximal and oblique dendrites
       L23PyrFRB_varInit_MBO_Dends = 3,      # middle basal and oblique dendrites
       L23PyrFRB_varInit_DBO_Dends = 4,      # distal basal and oblique dendrites
       L23PyrFRB_varInit_DA_Dends = arange(5,12+1),    # progressively more distal apical dendrites

       SupBasket_Axon = 0,
       SupBasket_Soma = 1,
       SupBasket_PBO_Dends = 2,
       SupBasket_MBO_Dends = 3,
       SupBasket_DBO_Dends = 4,

       SupAxAx_Axon = 0,
       SupAxAx_Soma = 1,
       SupAxAx_PBO_Dends = 2,
       SupAxAx_MBO_Dends = 3,
       SupAxAx_DBO_Dends = 4,

       SupLTSInter_Axon = 0,
       SupLTSInter_Soma = 1,
       SupLTSInter_PBO_Dends = 2,
       SupLTSInter_MBO_Dends = 3,
       SupLTSInter_DBO_Dends = 4,

       L4SpinyStellate_Axon = 0,
       L4SpinyStellate_Soma = 1,
       L4SpinyStellate_Dends = arange(2,9+1),

       L5TuftedPyrIB_Axon = arange(56,61+1).tolist(),
       L5TuftedPyrIB_Soma = [1],
       L5TuftedPyrIB_PBO_Dends = arange(2,12+1).tolist(),
       L5TuftedPyrIB_MBO_Dends = arange(13,23+1).tolist(),
       L5TuftedPyrIB_DBO_Dends = arange(24,34+1).tolist(),
       L5TuftedPyrIB_DA_Dends = arange(35,47+1).tolist(),
       L5TuftedPyrIB_Tuft = arange(48,55+1).tolist(),            # apical tuft

       L5TuftedPyrRS_Axon = 0,
       L5TuftedPyrRS_Soma = 1,
       L5TuftedPyrRS_PBO_Dends = 2,
       L5TuftedPyrRS_MBO_Dends = 3,
       L5TuftedPyrRS_DBO_Dends = 4,
       L5TuftedPyrRS_DA_Dends = arange(5,17+1),
       L5TuftedPyrRS_Tuft = 18,            # apical tuft

       L6NonTuftedPyrRS_Axon = 0,
       L6NonTuftedPyrRS_Soma = 1,
       L6NonTuftedPyrRS_PBO_Dends = 2,
       L6NonTuftedPyrRS_MBO_Dends = 3,
       L6NonTuftedPyrRS_DBO_Dends = 4,
       L6NonTuftedPyrRS_DA_Dends = arange(5,14+1),

       DeepBasket_Axon = 0,
       DeepBasket_Soma = 1,
       DeepBasket_PBO_Dends = 2,
       DeepBasket_MBO_Dends = 3,
       DeepBasket_DBO_Dends = 4,

       DeepAxAx_Axon = 0,
       DeepAxAx_Soma = 1,
       DeepAxAx_PBO_Dends = 2,
       DeepAxAx_MBO_Dends = 3,
       DeepAxAx_DBO_Dends = 4,

       DeepLTSInter_Axon = 0,
       DeepLTSInter_Soma = 1,
       DeepLTSInter_PBO_Dends = 2,
       DeepLTSInter_MBO_Dends = 3,
       DeepLTSInter_DBO_Dends = 4,

       TCR_Axon = 0,
       TCR_Soma = 1,
       TCR_Dends = arange(2,4+1),

       nRT_Axon = 0,
       nRT_Soma = 1,
       nRT_Dends = arange(2,9+1),

)

################################################################################
##### SYNAPTIC COMPARTMENT TOTALS
################################################################################
ncomp_syn = dict(

    ####################################
    #                                  #
    #      superficial RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_L23PyrRS_to_L23PyrRS  = 36,
         ncompallow_L23PyrRS_to_L23PyrFRB_varInit = 36,
         ncompallow_L23PyrRS_to_SupBasket   = 24,
         ncompallow_L23PyrRS_to_SupAxAx   = 24,
         ncompallow_L23PyrRS_to_SupLTSInter    = 24,
         ncompallow_L23PyrRS_to_L4SpinyStellate = 24,
         ncompallow_L23PyrRS_to_L5TuftedPyrIB    =  8,
         ncompallow_L23PyrRS_to_L5TuftedPyrRS    =  8,
         ncompallow_L23PyrRS_to_DeepBasket  = 24,
         ncompallow_L23PyrRS_to_DeepAxAx  = 24,
         ncompallow_L23PyrRS_to_DeepLTSInter   = 24,
         ncompallow_L23PyrRS_to_L6NonTuftedPyrRS =  7,

    ####################################
    #                                  #
    #     superficial FRB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_L23PyrFRB_varInit_to_L23PyrRS  = 36,
         ncompallow_L23PyrFRB_varInit_to_L23PyrFRB_varInit = 36,
         ncompallow_L23PyrFRB_varInit_to_SupBasket   = 24,
         ncompallow_L23PyrFRB_varInit_to_SupAxAx   = 24,
         ncompallow_L23PyrFRB_varInit_to_SupLTSInter    = 24,
         ncompallow_L23PyrFRB_varInit_to_L4SpinyStellate = 24,
         ncompallow_L23PyrFRB_varInit_to_L5TuftedPyrIB    =  8,
         ncompallow_L23PyrFRB_varInit_to_L5TuftedPyrRS    =  8,
         ncompallow_L23PyrFRB_varInit_to_DeepBasket  = 24,
         ncompallow_L23PyrFRB_varInit_to_DeepAxAx  = 24,
         ncompallow_L23PyrFRB_varInit_to_DeepLTSInter   = 24,
         ncompallow_L23PyrFRB_varInit_to_L6NonTuftedPyrRS =  7,

    ####################################
    #                                  #
    #     sup Basketet FS interneurons   #
    #              (GABA)              #
    #                                  #
    ####################################

         ncompallow_SupBasket_to_L23PyrRS    = 11,
         ncompallow_SupBasket_to_L23PyrFRB_varInit   = 11,
         ncompallow_SupBasket_to_SupBasket     = 24,
         ncompallow_SupBasket_to_SupAxAx     = 24,
         ncompallow_SupBasket_to_SupLTSInter      = 24,
         ncompallow_SupBasket_to_L4SpinyStellate   =  5,

    ####################################
    #                                  #
    #   sup axoaxonic FS internerons   #
    #              (GABA)              #
    #                                  #
    ####################################

         ncompallow_SupAxAx_to_L23PyrRS   = 1,
         ncompallow_SupAxAx_to_L23PyrFRB_varInit  = 1,
         ncompallow_SupAxAx_to_L4SpinyStellate  = 1,
         ncompallow_SupAxAx_to_L5TuftedPyrIB     = 1,
         ncompallow_SupAxAx_to_L5TuftedPyrRS     = 1,
         ncompallow_SupAxAx_to_L6NonTuftedPyrRS  = 1,

    ####################################
    #                                  #
    #  superficial LTS interneurons    #
    #             (GABA)               #
    #                                  #
    ####################################

         ncompallow_SupLTSInter_to_L23PyrRS     = 53,
         ncompallow_SupLTSInter_to_L23PyrFRB_varInit    = 53,
         ncompallow_SupLTSInter_to_SupBasket      = 40,
         ncompallow_SupLTSInter_to_SupAxAx      = 40,
         ncompallow_SupLTSInter_to_SupLTSInter       = 40,
         ncompallow_SupLTSInter_to_L4SpinyStellate    = 40,
         ncompallow_SupLTSInter_to_L5TuftedPyrIB       = 40,
         ncompallow_SupLTSInter_to_L5TuftedPyrRS       = 40,
         ncompallow_SupLTSInter_to_DeepBasket     = 20,
         ncompallow_SupLTSInter_to_DeepAxAx     = 20,
         ncompallow_SupLTSInter_to_DeepLTSInter      = 20,
         ncompallow_SupLTSInter_to_L6NonTuftedPyrRS    = 29,

    ####################################
    #                                  #
    #      spiny stellate cells        #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_L4SpinyStellate_to_L23PyrRS  = 24,
         ncompallow_L4SpinyStellate_to_L23PyrFRB_varInit = 24,
         ncompallow_L4SpinyStellate_to_SupBasket   = 24,
         ncompallow_L4SpinyStellate_to_SupAxAx   = 24,
         ncompallow_L4SpinyStellate_to_SupLTSInter    = 24,
         ncompallow_L4SpinyStellate_to_L4SpinyStellate = 24,
         ncompallow_L4SpinyStellate_to_L5TuftedPyrIB    = 12,
         ncompallow_L4SpinyStellate_to_L5TuftedPyrRS    = 12,
         ncompallow_L4SpinyStellate_to_DeepBasket  = 24,
         ncompallow_L4SpinyStellate_to_DeepAxAx  = 24,
         ncompallow_L4SpinyStellate_to_DeepLTSInter   = 24,
         ncompallow_L4SpinyStellate_to_L6NonTuftedPyrRS =  5,

    ####################################
    #                                  #
    #      deep tufted IB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_L5TuftedPyrIB_to_L23PyrRS     = 13,
         ncompallow_L5TuftedPyrIB_to_L23PyrFRB_varInit    = 13,
         ncompallow_L5TuftedPyrIB_to_SupBasket      = 24,
         ncompallow_L5TuftedPyrIB_to_SupAxAx      = 24,
         ncompallow_L5TuftedPyrIB_to_SupLTSInter       = 24,
         ncompallow_L5TuftedPyrIB_to_L4SpinyStellate    = 24,
         ncompallow_L5TuftedPyrIB_to_L5TuftedPyrIB       = 46,
         ncompallow_L5TuftedPyrIB_to_L5TuftedPyrRS       = 46,
         ncompallow_L5TuftedPyrIB_to_DeepBasket     = 24,
         ncompallow_L5TuftedPyrIB_to_DeepAxAx     = 24,
         ncompallow_L5TuftedPyrIB_to_DeepLTSInter      = 24,
         ncompallow_L5TuftedPyrIB_to_L6NonTuftedPyrRS    = 43,

    ####################################
    #                                  #
    #      deep tufted RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_L5TuftedPyrRS_to_L23PyrRS     = 13,
         ncompallow_L5TuftedPyrRS_to_L23PyrFRB_varInit    = 13,
         ncompallow_L5TuftedPyrRS_to_SupBasket      = 24,
         ncompallow_L5TuftedPyrRS_to_SupAxAx      = 24,
         ncompallow_L5TuftedPyrRS_to_SupLTSInter       = 24,
         ncompallow_L5TuftedPyrRS_to_L4SpinyStellate    = 24,
         ncompallow_L5TuftedPyrRS_to_L5TuftedPyrIB       = 46,
         ncompallow_L5TuftedPyrRS_to_L5TuftedPyrRS       = 46,
         ncompallow_L5TuftedPyrRS_to_DeepBasket     = 24,
         ncompallow_L5TuftedPyrRS_to_DeepAxAx     = 24,
         ncompallow_L5TuftedPyrRS_to_DeepLTSInter      = 24,
         ncompallow_L5TuftedPyrRS_to_L6NonTuftedPyrRS    = 43,

    ####################################
    #                                  #
    #    deep Basketet FS interneurons   #
    #              (GABA)              #
    #                                  #
    ####################################

         ncompallow_DeepBasket_to_L4SpinyStellate  =  5,
         ncompallow_DeepBasket_to_L5TuftedPyrIB     =  8,
         ncompallow_DeepBasket_to_L5TuftedPyrRS     =  8,
         ncompallow_DeepBasket_to_DeepBasket   = 24,
         ncompallow_DeepBasket_to_DeepAxAx   = 24,
         ncompallow_DeepBasket_to_DeepLTSInter    = 24,
         ncompallow_DeepBasket_to_L6NonTuftedPyrRS  =  8,

    ####################################
    #                                  #
    #  deep axoaxonic FS interneurons  #
    #              (GABA)              #
    #                                  #
    ####################################

         ncompallow_DeepAxAx_to_L23PyrRS   = 1,
         ncompallow_DeepAxAx_to_L23PyrFRB_varInit  = 1,
         ncompallow_DeepAxAx_to_L4SpinyStellate  = 1,
         ncompallow_DeepAxAx_to_L5TuftedPyrIB     = 1,
         ncompallow_DeepAxAx_to_L5TuftedPyrRS     = 1,
         ncompallow_DeepAxAx_to_L6NonTuftedPyrRS  = 1,

    ####################################
    #                                  #
    #       deep LTS interneurons      #
    #              (GABA)              #
    #                                  #
    ####################################

         ncompallow_DeepLTSInter_to_L23PyrRS    = 53,
         ncompallow_DeepLTSInter_to_L23PyrFRB_varInit   = 53,
         ncompallow_DeepLTSInter_to_SupBasket     = 20,
         ncompallow_DeepLTSInter_to_SupAxAx     = 20,
         ncompallow_DeepLTSInter_to_SupLTSInter      = 20,
         ncompallow_DeepLTSInter_to_L4SpinyStellate   = 40,
         ncompallow_DeepLTSInter_to_L5TuftedPyrIB      = 40,
         ncompallow_DeepLTSInter_to_L5TuftedPyrRS      = 40,
         ncompallow_DeepLTSInter_to_DeepBasket    = 40,
         ncompallow_DeepLTSInter_to_DeepAxAx    = 40,
         ncompallow_DeepLTSInter_to_DeepLTSInter     = 40,
         ncompallow_DeepLTSInter_to_L6NonTuftedPyrRS   = 29,

    ####################################
    #                                  #
    #      thalamic relay neurons      #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_TCR_to_L23PyrRS     = 24,
         ncompallow_TCR_to_L23PyrFRB_varInit    = 24,
         ncompallow_TCR_to_SupBasket      = 12,
         ncompallow_TCR_to_SupAxAx      = 12,
         ncompallow_TCR_to_L4SpinyStellate    = 52,
         ncompallow_TCR_to_L5TuftedPyrIB       =  9,
         ncompallow_TCR_to_L5TuftedPyrRS       =  9,
         ncompallow_TCR_to_DeepBasket     = 12,
         ncompallow_TCR_to_DeepAxAx     = 12,
         ncompallow_TCR_to_nRT          = 12,
         ncompallow_TCR_to_L6NonTuftedPyrRS    =  5,

    ####################################
    #                                  #
    #           nRT neurons            #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_nRT_to_TCR  = 11,
         ncompallow_nRT_to_nRT = 53,

    ####################################
    #                                  #
    #      deep nontuft RS pyramids    #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

         ncompallow_L6NonTuftedPyrRS_to_L23PyrRS  =  4,
         ncompallow_L6NonTuftedPyrRS_to_L23PyrFRB_varInit =  4,
         ncompallow_L6NonTuftedPyrRS_to_SupBasket   = 24,
         ncompallow_L6NonTuftedPyrRS_to_SupAxAx   = 24,
         ncompallow_L6NonTuftedPyrRS_to_SupLTSInter    = 24,
         ncompallow_L6NonTuftedPyrRS_to_L4SpinyStellate = 24,
         ncompallow_L6NonTuftedPyrRS_to_L5TuftedPyrIB    = 46,
         ncompallow_L6NonTuftedPyrRS_to_L5TuftedPyrRS    = 46,
         ncompallow_L6NonTuftedPyrRS_to_DeepBasket  = 24,
         ncompallow_L6NonTuftedPyrRS_to_DeepAxAx  = 24,
         ncompallow_L6NonTuftedPyrRS_to_DeepLTSInter   = 24,
         ncompallow_L6NonTuftedPyrRS_to_TCR       = 90,
         ncompallow_L6NonTuftedPyrRS_to_nRT       = 12,
         ncompallow_L6NonTuftedPyrRS_to_L6NonTuftedPyrRS = 43
)

################################################################################
##### SYNAPTIC COMPARTMENT PERMISSIONS
################################################################################
comp_syn = dict(

    ####################################
    #                                  #
    #      superficial RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

        compallow_L23PyrRS_to_L23PyrRS = [2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,26,
            27,28,29,30,31,32,33,10,11,12,13,22,23,24,25,34,35,36,37],

        compallow_L23PyrRS_to_L23PyrFRB_varInit = [2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,26,
            27,28,29,30,31,32,33,10,11,12,13,22,23,24,25,34,35,36,37],

        compallow_L23PyrRS_to_SupBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrRS_to_SupAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrRS_to_SupLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrRS_to_L4SpinyStellate = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrRS_to_L5TuftedPyrIB = [39,40,41,42,43,44,45,46],

        compallow_L23PyrRS_to_L5TuftedPyrRS = [39,40,41,42,43,44,45,46],

        compallow_L23PyrRS_to_DeepBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrRS_to_DeepAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrRS_to_DeepLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrRS_to_L6NonTuftedPyrRS = [38,39,40,41,42,43,44],

    ####################################
    #                                  #
    #     superficial FRB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

        compallow_L23PyrFRB_varInit_to_L23PyrRS = [2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,26,
            27,28,29,30,31,32,33,10,11,12,13,22,23,24,25,34,35,36,37],

        compallow_L23PyrFRB_varInit_to_L23PyrFRB_varInit = [2,3,4,5,6,7,8,9,14,15,16,17,18,19,20,21,26,
            27,28,29,30,31,32,33,10,11,12,13,22,23,24,25,34,35,36,37],

        compallow_L23PyrFRB_varInit_to_SupBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrFRB_varInit_to_SupAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrFRB_varInit_to_SupLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrFRB_varInit_to_L4SpinyStellate = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrFRB_varInit_to_L5TuftedPyrIB = [39,40,41,42,43,44,45,46],

        compallow_L23PyrFRB_varInit_to_L5TuftedPyrRS = [0.1, 39,40,41,42,43,44,45,46],

        compallow_L23PyrFRB_varInit_to_DeepBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrFRB_varInit_to_DeepAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrFRB_varInit_to_DeepLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_L23PyrFRB_varInit_to_L6NonTuftedPyrRS = [38,39,40,41,42,43,44],

    ####################################
    #                                  #
    #     sup Basketet FS interneurons   #
    #              (GABA)              #
    #                                  #
    ####################################

        compallow_SupBasket_to_L23PyrRS = [1,2,3,4,5,6,7,8,9,38,39],

        compallow_SupBasket_to_L23PyrFRB_varInit = [1,2,3,4,5,6,7,8,9,38,39],

        compallow_SupBasket_to_SupBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_SupBasket_to_SupAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

        compallow_SupBasket_to_SupLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
            44,45,46,47,48,49],

    ####################################
    #                                  #
    #   sup axoaxonic FS internerons   #
    #              (GABA)              #
    #                                  #
    ####################################

        compallow_SupAxAx_to_L23PyrRS   = [69],

        compallow_SupAxAx_to_L23PyrFRB_varInit  = [69],

        compallow_SupAxAx_to_L4SpinyStellate  = [54],

        compallow_SupAxAx_to_L5TuftedPyrIB     = [56],

        compallow_SupAxAx_to_L5TuftedPyrRS     = [56],

        compallow_SupAxAx_to_L6NonTuftedPyrRS  = [45],

    ####################################
    #                                  #
    #  superficial LTS interneurons    #
    #             (GABA)               #
    #                                  #
    ####################################

        compallow_SupLTSInter_to_L23PyrRS = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,
               37,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],

        compallow_SupLTSInter_to_L23PyrFRB_varInit = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,
               36,37,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],

        compallow_SupLTSInter_to_SupBasket = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_SupLTSInter_to_SupAxAx = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_SupLTSInter_to_SupLTSInter = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_SupLTSInter_to_L4SpinyStellate = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_SupLTSInter_to_L5TuftedPyrIB = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
                29,30,31,32,33,34,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],

        compallow_SupLTSInter_to_L5TuftedPyrRS = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
                29,30,31,32,33,34,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],

        compallow_SupLTSInter_to_DeepBasket = [8,9,10,11,12,21,22,23,24,25,34,35,36,37,38,
        	47,48,49,50,51],

        compallow_SupLTSInter_to_DeepAxAx = [8,9,10,11,12,21,22,23,24,25,34,35,36,37,38,
        	47,48,49,50,51],

        compallow_SupLTSInter_to_DeepLTSInter = [8,9,10,11,12,21,22,23,24,25,34,35,36,37,38,
        	47,48,49,50,51],

        compallow_SupLTSInter_to_L6NonTuftedPyrRS = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
        	29,30,31,32,33,34,38,39,40,41,42,43,44],

    ####################################
    #                                  #
    #      spiny stellate cells        #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

    # NOTE: 3 Mar. 2004: Feldmeyer, ..., Sakmann, J Physiol 2002 assert
    # that in barrel ctx, spiny stellates go to basal dendrites of
    # layer 2/3 pyramids

        # compallow_L4SpinyStellate_to_L23PyrRS = [40,41,42,43,44,45,46,47,48,49,50,51,52]

        # compallow_L4SpinyStellate_to_L23PyrFRB_varInit = [40,41,42,43,44,45,46,47,48,49,50,51,52]

        compallow_L4SpinyStellate_to_L23PyrRS = [2, 3, 4, 5, 6, 7, 8, 9,14,15,16,17,18,19,20,21,
                26,27,28,29,30,31,32,33],

        compallow_L4SpinyStellate_to_L23PyrFRB_varInit = [2, 3, 4, 5, 6, 7, 8, 9,14,15,16,17,18,19,20,21,
                26,27,28,29,30,31,32,33],

        compallow_L4SpinyStellate_to_SupBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L4SpinyStellate_to_SupAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L4SpinyStellate_to_SupLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L4SpinyStellate_to_L4SpinyStellate = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L4SpinyStellate_to_L5TuftedPyrIB = [7,8,9,10,11,12,36,37,38,39,40,41],

        compallow_L4SpinyStellate_to_L5TuftedPyrRS = [7,8,9,10,11,12,36,37,38,39,40,41],

        compallow_L4SpinyStellate_to_DeepBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L4SpinyStellate_to_DeepAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L4SpinyStellate_to_DeepLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L4SpinyStellate_to_L6NonTuftedPyrRS = [37,38,39,40,41],

    ####################################
    #                                  #
    #      deep tufted IB pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

        compallow_L5TuftedPyrIB_to_L23PyrRS = [40,41,42,43,44,45,46,47,48,49,50,51,52],

        compallow_L5TuftedPyrIB_to_L23PyrFRB_varInit = [40,41,42,43,44,45,46,47,48,49,50,51,52],

        compallow_L5TuftedPyrIB_to_SupBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrIB_to_SupAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrIB_to_SupLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrIB_to_L4SpinyStellate = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrIB_to_L5TuftedPyrIB = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],

        compallow_L5TuftedPyrIB_to_L5TuftedPyrRS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],

        compallow_L5TuftedPyrIB_to_DeepBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrIB_to_DeepAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrIB_to_DeepLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrIB_to_L6NonTuftedPyrRS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44],

    ####################################
    #                                  #
    #      deep tufted RS pyramids     #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

        compallow_L5TuftedPyrRS_to_L23PyrRS = [40,41,42,43,44,45,46,47,48,49,50,51,52],


        compallow_L5TuftedPyrRS_to_L23PyrFRB_varInit = [40,41,42,43,44,45,46,47,48,49,50,51,52],


        compallow_L5TuftedPyrRS_to_SupBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrRS_to_SupAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrRS_to_SupLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrRS_to_L4SpinyStellate = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrRS_to_L5TuftedPyrIB = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,8,39,40,41,42,43,44,45,46,47],

        compallow_L5TuftedPyrRS_to_L5TuftedPyrRS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],

        compallow_L5TuftedPyrRS_to_DeepBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrRS_to_DeepAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrRS_to_DeepLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L5TuftedPyrRS_to_L6NonTuftedPyrRS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44],

    ####################################
    #                                  #
    #    deep Basketet FS interneurons   #
    #              (GABA)              #
    #                                  #
    ####################################

        compallow_DeepBasket_to_L4SpinyStellate = [1,2,15,28,41],

        compallow_DeepBasket_to_L5TuftedPyrIB = [1,2,3,4,5,6,35,36],

        compallow_DeepBasket_to_L5TuftedPyrRS = [1,2,3,4,5,6,35,36],

        compallow_DeepBasket_to_DeepBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_DeepBasket_to_DeepAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_DeepBasket_to_DeepLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_DeepBasket_to_L6NonTuftedPyrRS = [1,2,3,4,5,6,35,36],

        compallow_SupBasket_to_L4SpinyStellate = [1,2,15,28,41],

    ####################################
    #                                  #
    #  deep axoaxonic FS interneurons  #
    #              (GABA)              #
    #                                  #
    ####################################

        compallow_DeepAxAx_to_L23PyrRS   = [69],

        compallow_DeepAxAx_to_L23PyrFRB_varInit  = [69],

        compallow_DeepAxAx_to_L4SpinyStellate  = [54],

        compallow_DeepAxAx_to_L5TuftedPyrIB     = [56],

        compallow_DeepAxAx_to_L5TuftedPyrRS     = [56],

        compallow_DeepAxAx_to_L6NonTuftedPyrRS  = [45],

    ####################################
    #                                  #
    #       deep LTS interneurons      #
    #              (GABA)              #
    #                                  #
    ####################################

        compallow_DeepLTSInter_to_L23PyrRS = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
                33,34,35,36,37,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],

        compallow_DeepLTSInter_to_L23PyrFRB_varInit = [14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
               31,32,33,34,35,36,37,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68],

        compallow_DeepLTSInter_to_SupBasket = [8,9,10,11,12,21,22,23,24,25,34,35,36,37,38,
                47,48,49,50,51],

        compallow_DeepLTSInter_to_SupAxAx = [8,9,10,11,12,21,22,23,24,25,34,35,36,37,38,
                47,48,49,50,51],

        compallow_DeepLTSInter_to_SupLTSInter = [8,9,10,11,12,21,22,23,24,25,34,35,36,37,38,
             47,48,49,50,51],

        compallow_DeepLTSInter_to_L4SpinyStellate = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_DeepLTSInter_to_L5TuftedPyrIB = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
                29,30,31,32,33,34,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],

        compallow_DeepLTSInter_to_L5TuftedPyrRS = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
                29,30,31,32,33,34,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55],

        compallow_DeepLTSInter_to_DeepBasket = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_DeepLTSInter_to_DeepAxAx = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_DeepLTSInter_to_DeepLTSInter = [5,6,7,8,9,10,11,12,13,14,18,19,20,21,22,23,24,25,
               26,27,31,32,33,34,35,36,37,38,39,40,44,45,46,47,48,49,50,51,52,53],

        compallow_DeepLTSInter_to_L6NonTuftedPyrRS = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,
               29,30,31,32,33,34,38,39,40,41,42,43,44],

    ####################################
    #                                  #
    #      thalamic relay neurons      #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

        compallow_TCR_to_L23PyrRS = [45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
                61,62,63,64,65,66,67,68],

        compallow_TCR_to_L23PyrFRB_varInit = [45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,
                61,62,63,64,65,66,67,68],

        compallow_TCR_to_SupBasket = [2,3,4,15,16,17,28,29,30,41,42,43],


        compallow_TCR_to_SupAxAx = [2,3,4,15,16,17,28,29,30,41,42,43],


        compallow_TCR_to_L4SpinyStellate = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],

        compallow_TCR_to_L5TuftedPyrIB = [47,48,49,50,51,52,53,54,55],

        compallow_TCR_to_L5TuftedPyrRS = [47,48,49,50,51,52,53,54,55],

        compallow_TCR_to_DeepBasket = [2,3,4,15,16,17,28,29,30,41,42,43],

        compallow_TCR_to_DeepAxAx = [2,3,4,15,16,17,28,29,30,41,42,43],

        compallow_TCR_to_nRT = [2,3,4,15,16,17,28,29,30,41,42,43],

        compallow_TCR_to_L6NonTuftedPyrRS = [40,41,42,43,44],

    ####################################
    #                                  #
    #           nRT neurons            #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

        compallow_nRT_to_TCR = [1,2,15,28,41,54,67,80,93,106,119],

        compallow_nRT_to_nRT = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53],

    ####################################
    #                                  #
    #      deep nontuft RS pyramids    #
    #          (AMPA + NMDA)           #
    #                                  #
    ####################################

        compallow_L6NonTuftedPyrRS_to_L23PyrRS = [41,42,43,44],

        compallow_L6NonTuftedPyrRS_to_L23PyrFRB_varInit = [41,42,43,44],

        compallow_L6NonTuftedPyrRS_to_SupBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L6NonTuftedPyrRS_to_SupAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L6NonTuftedPyrRS_to_SupLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L6NonTuftedPyrRS_to_L4SpinyStellate = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L6NonTuftedPyrRS_to_L5TuftedPyrIB = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],

        compallow_L6NonTuftedPyrRS_to_L5TuftedPyrRS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],

        compallow_L6NonTuftedPyrRS_to_DeepBasket = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L6NonTuftedPyrRS_to_DeepAxAx = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L6NonTuftedPyrRS_to_DeepLTSInter = [5,6,7,8,9,10,18,19,20,21,22,23,31,32,33,34,35,36,
                44,45,46,47,48,49],

        compallow_L6NonTuftedPyrRS_to_TCR = [6,  7,  8,  9, 10, 11, 12, 13, 14,
                                     19, 20, 21, 22, 23, 24, 25, 26, 27,
                                     32, 33, 34, 35, 36, 37, 38, 39, 40,
                                     45, 46, 47, 48, 49, 50, 51, 52, 53,
                                     58, 59, 60, 61, 62, 63, 64, 65, 66,
                                     71, 72, 73, 74, 75, 76, 77, 78, 79,
                                     84, 85, 86, 87, 88, 89, 90, 91, 92,
                                     97, 98, 99,100,101,102,103,104,105,
                                     110,111,112,113,114,115,116,117,118,
                                     123,124,125,126,127,128,129,130,131],

        compallow_L6NonTuftedPyrRS_to_nRT = [2,3,4,15,16,17,28,29,30,41,42,43],

        compallow_L6NonTuftedPyrRS_to_L6NonTuftedPyrRS = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44],
)


################################################################################
##### GAP JUNCTION COMPARTMENT PERMISSIONS
################################################################################
comp_gap = dict(
     compallow_L23PyrRS = [74],
     compallow_L23PyrFRB_varInit = [74],
     compallow_SupBasket = [3,4,16,17,29,30,42,43],
     compallow_SupLTSInter = [3,4,16,17,29,30,42,43],
     compallow_L4SpinyStellate = [59],
     compallow_L5TuftedPyrIB = [61],
     compallow_L5TuftedPyrRS = [61],
     compallow_L6NonTuftedPyrRS = [50],
     compallow_DeepBasket = [3,4,16,17,29,30,42,43],
     compallow_DeepLTSInter = [3,4,16,17,29,30,42,43],
     compallow_TCR = [137],
     compallow_nRT = [3,4,16,17,29,30,42,43],
)

################################################################################
##### GAP JUNCTION UNITARY CONDUCTANCES (in nS)
################################################################################
g_gap = dict(
     gGAP_L23PyrRS = 3,
     gGAP_L23PyrFRB_varInit = 3,
     gGAP_SupBasket = 1,
     gGAP_SupLTSInter = 1,
     gGAP_L4SpinyStellate = 3,
     gGAP_L5TuftedPyrIB = 4,
     gGAP_L5TuftedPyrRS = 4,
     gGAP_DeepBasket = 1,
     gGAP_DeepLTSInter = 1,
     gGAP_L6NonTuftedPyrRS = 4,
)

n_gap = dict(               # Average values:
     nGAP_L23PyrRS = 2,     # 1.44 junctions per RS axon
     nGAP_L23PyrFRB_varInit = 1,    # 0.16 junctions per FRB axon
     nGAP_SupBasket = 4,      # 4.44 junctions per Basket dendrites
     nGAP_SupLTSInter = 4,       # 4.44 junctions per LTS dendrites
     nGAP_L4SpinyStellate = 2,    # 2 juctions per spiny stellate axon
     nGAP_L5TuftedPyrIB = 1,       # 0.875 junctions per L5TuftedPyrIB axon
     nGAP_L5TuftedPyrRS = 4,       # 3.5 junctions per L5TuftedPyrRS axon
     nGAP_DeepBasket = 5,      # 5 junctions per Basket dendrites
     nGAP_DeepLTSInter = 5,       # 5 junctions per LTS dendrites
     nGAP_L6NonTuftedPyrRS = 2     # 2 junctions per L6NonTuftedPyrRS axon
)
