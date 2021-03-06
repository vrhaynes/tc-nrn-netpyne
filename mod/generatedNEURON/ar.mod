COMMENT

   **************************************************
   File generated by: neuroConstruct v1.7.1 
   **************************************************


ENDCOMMENT


?  This is a NEURON mod file generated from a ChannelML file

?  Unit system of original ChannelML file: Physiological Units

COMMENT
    ChannelML file based on Traub et al. 2003
ENDCOMMENT

TITLE Channel: ar

COMMENT
    Anomalous Rectifier conductance, also known as h-conductance (hyperpolarizing). Based on NEURON port of FRB L2/3 model from Traub et al 2003. Same channel used in Traub et al 2005
ENDCOMMENT


UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
    (um) = (micrometer)
    (molar) = (1/liter)
    (mM) = (millimolar)
    (l) = (liter)
}


    
NEURON {
      

    SUFFIX ar
    USEION ar READ ear WRITE iar VALENCE 1  ? reversal potential of ion is read, outgoing current is written
           
        
    RANGE gmax, gion
    
    RANGE minf, mtau
    
    RANGE m0
}

PARAMETER { 
      

    gmax = 0.00025 (S/cm2)  ? default value, should be overwritten when conductance placed on cell
    
    m0 = 0 : Note units of this will be determined by its usage in the generic functions

}



ASSIGNED {
      

    v (mV)
    
    celsius (degC)
          

    ? Reversal potential of ar
    ear (mV)
    ? The outward flow of ion: ar calculated by rate equations...
    iar (mA/cm2)
    
    
    gion (S/cm2)
    minf
    mtau (ms)
    
}

BREAKPOINT { 
                        
    SOLVE states METHOD cnexp
         

    gion = gmax*((m)^1)      

    iar = gion*(v - ear)
            

}



INITIAL {
    
    ear = -35
        
    rates(v)
    m = minf
        
    
}
    
STATE {
    m
    
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m)/mtau
    
}

PROCEDURE rates(v(mV)) {  
    
    ? Note: not all of these may be used, depending on the form of rate equations
    LOCAL  alpha, beta, tau, inf, gamma, zeta, temp_adj_m, A_tau_m, B_tau_m, Vhalf_tau_m, A_inf_m, B_inf_m, Vhalf_inf_m
        
    TABLE minf, mtau
 DEPEND celsius, m0
 FROM -120 TO 60 WITH 741
    
    
    UNITSOFF
    temp_adj_m = 1
    
            
                
           

        
    ?      ***  Adding rate equations for gate: m  ***
         
    ? Found a generic form of the rate equation for tau, using expression: 1 /((exp (-14.6 - (0.086 * v) )) + (exp (-1.87 + (0.07 * v))))
    tau = 1 /((exp (-14.6 - (0.086 * v) )) + (exp (-1.87 + (0.07 * v))))
        
    mtau = tau/temp_adj_m
    
    ? Found a parameterised form of rate equation for inf, using expression: A / (1 + exp((v-Vhalf)/B))
    A_inf_m = 1
    B_inf_m = 5.5
    Vhalf_inf_m = -75 
    inf = A_inf_m / (exp((v - Vhalf_inf_m) / B_inf_m) + 1)
    
    minf = inf
          
       
    
    ?     *** Finished rate equations for gate: m ***
    

         

}


UNITSON


