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

TITLE Channel: cal

COMMENT
    High threshold, long lasting Calcium L-type current. Based on NEURON port of FRB L2/3 model from Traub et al 2003. Same channel used in Traub et al 2005
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
      

    SUFFIX cal
    USEION ca WRITE ica VALENCE 2 ?  outgoing current is written
           
        
    RANGE gmax, gion
    
    RANGE minf, mtau
    
}

PARAMETER { 
      

    gmax = 0.0005 (S/cm2)  ? default value, should be overwritten when conductance placed on cell
    
}



ASSIGNED {
      

    v (mV)
    
    celsius (degC)
          

    ? Reversal potential of ca
    eca (mV)
    ? The outward flow of ion: ca calculated by rate equations...
    ica (mA/cm2)
    
    
    gion (S/cm2)
    minf
    mtau (ms)
    
}

BREAKPOINT { 
                        
    SOLVE states METHOD cnexp
         

    gion = gmax*((1*m)^2)      

    ica = gion*(v - eca)
            

}



INITIAL {
    
    eca = 125
        
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
    LOCAL  alpha, beta, tau, inf, gamma, zeta, temp_adj_m, A_alpha_m, B_alpha_m, Vhalf_alpha_m, A_beta_m, B_beta_m, Vhalf_beta_m
        
    TABLE minf, mtau
 DEPEND celsius
 FROM -120 TO 60 WITH 741
    
    
    UNITSOFF
    temp_adj_m = 1
    
            
                
           

        
    ?      ***  Adding rate equations for gate: m  ***
        
    ? Found a parameterised form of rate equation for alpha, using expression: A / (1 + exp((v-Vhalf)/B))
    A_alpha_m = 1.6
    B_alpha_m = -13.888889
    Vhalf_alpha_m = 5 
    alpha = A_alpha_m / (exp((v - Vhalf_alpha_m) / B_alpha_m) + 1)
    
    
    ? Found a parameterised form of rate equation for beta, using expression: A*((v-Vhalf)/B) / (1 - exp(-((v-Vhalf)/B)))
    A_beta_m = 0.1
    B_beta_m = -5
    Vhalf_beta_m = -8.9 
    beta = A_beta_m * vtrap((v - Vhalf_beta_m), B_beta_m)
    
    mtau = 1/(temp_adj_m*(alpha + beta))
    minf = alpha/(alpha + beta)
          
       
    
    ?     *** Finished rate equations for gate: m ***
    

         

}


? Function to assist with parameterised expressions of type linoid/exp_linear

FUNCTION vtrap(VminV0, B) {
    if (fabs(VminV0/B) < 1e-6) {
    vtrap = (1 + VminV0/B/2)
}else{
    vtrap = (VminV0 / B) /(1 - exp((-1 *VminV0)/B))
    }
}

UNITSON


