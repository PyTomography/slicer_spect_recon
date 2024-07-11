import slicer
import vtk

def getAllScalarVolumeNodes():
    scene = slicer.mrmlScene
    scalarVolumeNodes = []
    nodes = scene.GetNodesByClass('vtkMRMLScalarVolumeNode')
    nodes.InitTraversal()
    node = nodes.GetNextItemAsObject()
    while node:
        scalarVolumeNodes.append(node)
        node = nodes.GetNextItemAsObject()
    return scalarVolumeNodes

def getDicomModalityFromInstanceUIDs(node):
    if node.GetStorageNode():
        dicomDatabase = slicer.dicomDatabase
        modality = dicomDatabase.fileValue(node.GetStorageNode().GetFileName(), "0008,0060")
        return modality
    else:
        instanceUIDs = node.GetAttribute("DICOM.instanceUIDs")
        if instanceUIDs:
            uidList = instanceUIDs.split()
            if uidList:
                dicomDatabase = slicer.dicomDatabase
                modality = dicomDatabase.instanceValue(uidList[0], "0008,0060")
                return modality
    return None

def filterNodesByModality(nodes, modality):
    filteredNodes = []
    for node in nodes:
        dicomModality = getDicomModalityFromInstanceUIDs(node)
        if dicomModality == modality:
            filteredNodes.append(node)
    return filteredNodes


