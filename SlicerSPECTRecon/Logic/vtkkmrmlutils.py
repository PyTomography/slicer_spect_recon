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

def pathFromNode(node):
    #TODO: Review this function to handle the case where the data was dragged and dropped
    if node is not None:
        storageNode = node.GetStorageNode()
        if storageNode is not None: # loaded via drag-drop
            filepath = storageNode.GetFullNameFromFileName()
        else: # Loaded via DICOM browser
            instanceUIDs = node.GetAttribute("DICOM.instanceUIDs").split()
            filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])
    return filepath

def filesFromNode(node):
    #TODO: Review this function to handle the case where the data was dragged and dropped
    if node is not None:
        storageNode = node.GetStorageNode()
        if storageNode is not None: # loaded via drag-drop
            filepaths = storageNode.GetFullNameFromFileName()
        else: # Loaded via DICOM browser
            instanceUIDs = node.GetAttribute("DICOM.instanceUIDs").split()
            filepaths = [slicer.dicomDatabase.fileForInstance(instanceUID) for instanceUID in instanceUIDs]
        return filepaths
    else:
        return None
    
def get_filesNM_from_NMNodes(NM_nodes):
    files_NM = []
    for NM_node in NM_nodes:
        path = pathFromNode(NM_node)
        files_NM.append(path)
    return files_NM


