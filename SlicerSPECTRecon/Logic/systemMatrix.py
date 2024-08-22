from pytomography.projectors.SPECT import SPECTSystemMatrix

def spectSystemMatrix(obj2obj_transforms, object_meta, proj_meta):
    system_matrix = SPECTSystemMatrix(
        obj2obj_transforms = obj2obj_transforms,
        proj2proj_transforms = [],
        object_meta = object_meta,
        proj_meta = proj_meta)
    return system_matrix