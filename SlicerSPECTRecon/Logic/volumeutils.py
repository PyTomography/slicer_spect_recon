import slicer
from DICOMLib import DICOMUtils
import tempfile
import shutil
import vtk

def createTempDir():
    return tempfile.mkdtemp()

def saveFilesInBrowser(file_path):
    dicomBrowser = slicer.modules.DICOMWidget.browserWidget.dicomBrowser
    dicomBrowser.importDirectory(file_path, dicomBrowser.ImportDirectoryAddLink)
    dicomBrowser.waitForImportFinished()

def loadFromTempDB(path):
    loadedNodeIDs = []
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(path, db)
        patientUIDs = db.patients()
        for patientUID in patientUIDs:
            loadedNodeIDs.extend(DICOMUtils.loadPatientByUID(patientUID))
    return loadedNodeIDs

def getVolumeNode(NodeIDs):
    if NodeIDs:
        volumeNode = slicer.mrmlScene.GetNodeByID(NodeIDs[0])
        return volumeNode
    
def displayVolumeInViewer(volumeNode, outputVolumeNode):
    outputVolumeNode.SetAndObserveImageData(volumeNode.GetImageData())
    outputVolumeNode.CreateDefaultDisplayNodes() 
    colorTableID = volumeNode.GetDisplayNode().GetColorNodeID()
    outputVolumeNode.GetDisplayNode().SetAndObserveColorNodeID(colorTableID)
    window = volumeNode.GetDisplayNode().GetWindow()
    level = volumeNode.GetDisplayNode().GetLevel()
    outputVolumeNode.GetDisplayNode().SetWindow(window)
    outputVolumeNode.GetDisplayNode().SetLevel(level)
    volumeMatrix = vtk.vtkMatrix4x4()
    volumeNode.GetRASToIJKMatrix(volumeMatrix)
    outputVolumeNode.SetRASToIJKMatrix(volumeMatrix)

def removeNode(volumeNode, dir):
    slicer.mrmlScene.RemoveNode(volumeNode)
    shutil.rmtree(dir)
