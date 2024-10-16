from pytomography.algorithms import OSEM, BSREM, OSMAPOSL
    
def selectAlgorithm(algorithm, likelihood, prior):
    if algorithm == "OSEM":
        reconstruction_algorithm = OSEM(likelihood)
        return reconstruction_algorithm
    elif algorithm == "BSREM":
        reconstruction_algorithm = BSREM(likelihood, prior=prior)
        return reconstruction_algorithm
    elif algorithm == "OSMAPOSL":
        reconstruction_algorithm = OSMAPOSL(likelihood, prior=prior)
        return reconstruction_algorithm
