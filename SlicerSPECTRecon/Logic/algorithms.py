from pytomography.algorithms import OSEM, BSREM, OSMAPOSL
    
def selectAlgorithm(algorithm, likelihood, prior):
    if algorithm == "OSEM":
        reconstruction_algorithm = osem(likelihood)
        return reconstruction_algorithm
    elif algorithm == "BSREM":
        reconstruction_algorithm = bsrem(likelihood, prior=prior)
        return reconstruction_algorithm
    elif algorithm == "OSMAPOSL":
        reconstruction_algorithm = osmaposl(likelihood, prior=prior)
        return reconstruction_algorithm

def osem(likelihood):
    return OSEM(likelihood)

def bsrem(likelihood, prior):
    return BSREM(likelihood, prior=prior)

def osmaposl(likelihood, prior):
    return OSMAPOSL(likelihood, prior=prior)
