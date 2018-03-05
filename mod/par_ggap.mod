: ggap.mod
: This is a conductance based gap junction model rather
: than resistance because Traub occasionally likes to
: set g=0 which of course is infinite resistance.
NEURON {
	POINT_PROCESS gGapPar
	RANGE g, i, vpeer
	RANGE weight
	ELECTRODE_CURRENT i
}

PARAMETER {
		v (millivolt)
		vpeer (millivolt)
		g = 1e-10 (1/megohm)
		weight = 1
}

ASSIGNED {
	i (nanoamp)
}
BREAKPOINT { i = weight * g * (vpeer - v) }
