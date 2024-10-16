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
    if node is not None:
        storageNode = node.GetStorageNode()
        if storageNode is not None: # loaded via drag-drop
            filepath = storageNode.GetFullNameFromFileName()
        else: # Loaded via DICOM browser
            instanceUIDs = node.GetAttribute("DICOM.instanceUIDs").split()
            filepath = slicer.dicomDatabase.fileForInstance(instanceUIDs[0])
    return filepath

def filesFromNode(node):
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

def createTable(column_names, column_types):
    table = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
    for column_name, column_type in zip(column_names, column_types):
        if column_type == 'string':
            col = table.AddColumn()
        elif column_type == 'float':
            col = table.AddColumn(vtk.vtkDoubleArray())
        col.SetName(column_name)
        table.SetColumnTitle(column_name, column_name)
    return table

def displayTable(table):
    currentLayout = slicer.app.layoutManager().layout
    layoutWithTable = slicer.modules.tables.logic().GetLayoutWithTable(currentLayout)
    slicer.app.layoutManager().setLayout(layoutWithTable)
    slicer.app.applicationLogic().GetSelectionNode().SetActiveTableID(table.GetID())
    table.SetUseColumnTitleAsColumnHeader(True)
    slicer.app.applicationLogic().PropagateTableSelection()
    


