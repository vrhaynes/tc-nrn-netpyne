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

? Creating NMDA like synaptic mechanism, based on NEURON source impl of Exp2Syn
    

TITLE Channel: Syn_NMDA_L5RS_SupIN

COMMENT
    Synapse with syn scaling constant c = 0.15 nS (translating to max cond of 1.5e-07 mS), time course: 100 ms and reversal potential: 0 mV.
        Automatically generated by command:  genSyn.py Syn_NMDA_L5RS_SupIN 100 0.15 0 
ENDCOMMENT


UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

    
NEURON {
    POINT_PROCESS Syn_NMDA_L5RS_SupIN
    RANGE tau_rise, tau_decay 
    GLOBAL total
    

    RANGE mg_conc, eta, gamma, gblock
    GLOBAL total


    RANGE i, e, gmax
    NONSPECIFIC_CURRENT i
    RANGE g, factor

}

PARAMETER {
    gmax = 0.00015
    tau_rise = 1 (ms) <1e-9,1e9>
    tau_decay = 100 (ms) <1e-9,1e9>
    e = 0.0  (mV)
mg_conc = 1.5 
              
    eta = 0.280112 
              
    gamma = 0.062
}


ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    factor 
    total (uS)
    gblock
}

STATE {
    A (uS)
    B (uS)
}

INITIAL {
    LOCAL tp
    total = 0
    
    if (tau_rise == 0) {
        tau_rise = 1e-9  : will effectively give a single exponential timecourse synapse
    }
    
    if (tau_rise/tau_decay > .999999) {
        tau_rise = .999999*tau_decay : will result in an "alpha" synapse waveform
    }
    A = 0
    B = 0
    tp = (tau_rise*tau_decay)/(tau_decay - tau_rise) * log(tau_decay/tau_rise)
    factor = -exp(-tp/tau_rise) + exp(-tp/tau_decay)
    factor = 1/factor
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    gblock = 1 / (1+ (mg_conc * eta * exp(-1 * gamma * v)))
    g = gmax * gblock * (B - A)
    i = g*(v - e)
        

}


DERIVATIVE state {
    A' = -A/tau_rise
    B' = -B/tau_decay 
}

NET_RECEIVE(weight (uS)) {
    
    state_discontinuity(A, A + weight*factor
)
    state_discontinuity(B, B + weight*factor
)

    
    
}

