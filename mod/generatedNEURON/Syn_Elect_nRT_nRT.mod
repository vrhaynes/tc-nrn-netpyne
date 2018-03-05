COMMENT

   **************************************************
   File generated by: neuroConstruct v1.7.1 
   **************************************************


ENDCOMMENT


?  This is a NEURON mod file generated from a ChannelML file

?  Unit system of original ChannelML file: Physiological Units

COMMENT
    ChannelML file describing a single synaptic mechanism
ENDCOMMENT

? Creating synaptic mechanism for an electrical synapse
    

TITLE Channel: Syn_Elect_nRT_nRT

COMMENT
    Electrical synapse with scaling constant c = 1 nS (translating to conductance of 1e-06 mS).
        Automatically generated by command:  genSyn.py Syn_Elect_nRT_nRT -1 1 -1 
ENDCOMMENT


UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

    
NEURON {
    POINT_PROCESS Syn_Elect_nRT_nRT
    NONSPECIFIC_CURRENT i
    RANGE g, i
    RANGE weight
    
    POINTER vgap  : Using a POINTER as opposed to RANGE for serial mode
        

}

PARAMETER {
    v (millivolt)
    vgap (millivolt)
    g = 0.001 (microsiemens)
    weight = 1

}


ASSIGNED {
    i (nanoamp)
}

BREAKPOINT {
    i = weight * g * (v - vgap)
} 
