from pytomography.likelihoods import PoissonLogLikelihood

def poissonLogLikelihood(system_matrix, photopeak, scatter):
    return PoissonLogLikelihood(system_matrix, photopeak, scatter)