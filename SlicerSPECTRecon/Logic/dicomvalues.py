class DicomValues:
    """Instance UIDs for projection test files
        NM1, NM2 -> Spect projection data for test patient
        CT -> CT attenuation file for test patient
        NM3 -> Spect projection for test simind data
        CT1 -> CT attenuation file for test simind data
    """
    def __init__(self):        
        self.NM_studyInstanceUID = '1.2.840.113619.2.280.2.1.13122020104714148.2070446548'
        self.NM1_seriesInstanceUID = '1.2.840.113619.2.280.2.1.14122020074609632.9516459'
        self.NM2_seriesInstanceUID = '1.2.840.113619.2.280.2.1.14122020080313849.910125375'
        self.NM_modality = 'NM'
        self.CT_studyInstanceUID = '1.2.840.113619.2.280.2.1.13122020104714148.2070446548'
        self.CT_seriesInstanceUID = '1.2.840.113619.2.55.3.168428554.503.1605786858.351'
        self.CT_modality = 'CT'
        self.NM3_studyInstanceUID = '1.2.826.0.1.3680043.8.498.61667537236486865579842022382500976504'
        self.NM3_seriesInstanceUID = '1.2.826.0.1.3680043.8.498.44527147084865889252670288028979437911'
        self.CT1_studyInstanceUID = '1.2.826.0.1.3680043.8.498.11431749630562073871553761496090682777'
        self.CT1_studyInstanceUID = '1.2.826.0.1.3680043.8.498.65581312818946477906207696227990322736'