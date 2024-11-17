import slicer
from pytomography.callbacks import Callback, DataStorageCallback

class LoadingCallback(Callback):
    def __init__(self, progressDialog, n_iters, n_subsets):
        self.progressDialog = progressDialog
        self.total_subiterations = n_iters*n_subsets
        self.n_subsets = n_subsets

    def run(self, object, n_iter, n_subset):
        self.progressDialog.value = int((n_iter*self.n_subsets+n_subset+1)/self.total_subiterations * 100)
        slicer.app.processEvents()
        return object
    
class DataStorageWithLoadingCallback(DataStorageCallback):
    def __init__(self, likelihood, object_prediction, progressDialog, n_iters, n_subsets):
        super().__init__(likelihood, object_prediction)
        self.progressDialog = progressDialog
        self.total_subiterations = n_iters*n_subsets
        self.n_subsets = n_subsets

    def run(self, object, n_iter, n_subset):
        object = super().run(object, n_iter, n_subset)
        self.progressDialog.value = int((n_iter*self.n_subsets+n_subset+1)/self.total_subiterations * 100)
        slicer.app.processEvents()
        return object