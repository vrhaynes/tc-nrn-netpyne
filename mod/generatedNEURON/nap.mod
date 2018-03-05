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

TITLE Channel: nap

COMMENT
    Persistent (non inactivating) Sodium channel. Based on NEURON port of FRB L2/3 model from Traub et al 2003. Same channel used in Traub et al 2005
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
      

    SUFFIX nap
    USEION na READ ena WRITE ina VALENCE 1  ? reversal potential of ion is read, outgoing current is written
           
        
    RANGE gmax, gion
    
    RANGE minf, mtau
    
}

PARAMETER { 
      

    gmax = 0.0001 (S/cm2)  ? default value, should be overwritten when conductance placed on cell
    
}



ASSIGNED {
      

    v (mV)
    
    celsius (degC)
          

    ? Reversal potential of na
    ena (mV)
    ? The outward flow of ion: na calculated by rate equations...
    ina (mA/cm2)
    
    
    gion (S/cm2)
    minf
    mtau (ms)
    
}

BREAKPOINT { 
                        
    SOLVE states METHOD cnexp
         

    gion = gmax*((m)^1)      

    ina = gion*(v - ena)
            

}



INITIAL {
    
    ena = 50
        
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
 DEPEND celsius
 FROM -120 TO 60 WITH 741
    
    
    UNITSOFF
    temp_adj_m = 1
    
            
                
           

        
    ?      ***  Adding rate equations for gate: m  ***
         
    ? Found a generic form of the rate equation for tau, using expression: v < -40 ? 0.025 + 0.14 * (exp (( v + 40 )/10)) : 0.02 + 0.145 * (exp (-1 * (v + 40)/ 10))
    
    
    if (v < -40 ) {
        tau =  0.025 + 0.14 * (exp (( v + 40 )/10)) 
    } else {
        tau =  0.02 + 0.145 * (exp (-1 * (v + 40)/ 10))
    }
    mtau = tau/temp_adj_m
    
    ? Found a parameterised form of rate equation for inf, using expression: A / (1 + exp((v-Vhalf)/B))
    A_inf_m = 1
    B_inf_m = -10
    Vhalf_inf_m = -48 
    inf = A_inf_m / (exp((v - Vhalf_inf_m) / B_inf_m) + 1)
    
    minf = inf
          
       
    
    ?     *** Finished rate equations for gate: m ***
    

         

}


UNITSON


