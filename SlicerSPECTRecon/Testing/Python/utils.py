import os
import requests
import slicer
from pathlib import Path
import logging
import qt
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
DATABASE_INITIALIZED = False

def openDICOMDatabase(databaseName='SlicerSpectReconDB'):
    settings = qt.QSettings()
    dbDirectoryKey = f'{databaseName}Directory'
    if not settings.contains(dbDirectoryKey):
        DB_Directory = Path.home() / databaseName
        qt.QSettings().setValue(dbDirectoryKey, str(DB_Directory))
    else:
        DB_Directory = Path(settings.value(dbDirectoryKey))
    currentDatabasePath = Path(slicer.dicomDatabase.databaseFilename).parent if slicer.dicomDatabase.isOpen else None
    if currentDatabasePath == DB_Directory:
        logger.info('The correct database is already opened')
        return
    if slicer.dicomDatabase.isOpen:
        slicer.dicomDatabase.closeDatabase()
        logger.info('Closed existing database')
    try:
        DB_Directory.mkdir(exist_ok=True)
    except FileNotFoundError as e:
        logger.error("The database directory cannot be created \n{}".format(e))
    try:
        databaseFilePath = DB_Directory / 'ctkDICOM.sql'
        databaseFilePath.touch(exist_ok=True)
    except Exception as e:
        logger.error("The database directory cannot be created \n{}: {}".format(type(e), e))

    logger.info(f"Database created in {databaseFilePath}")
    if not (DB_Directory.is_dir() and databaseFilePath.is_file()):
        logger.error(f"The database file path '{databaseFilePath}' cannot be opened, please adjust the read/write privileges")
        raise OSError('Database has invalid read/write privileges')
    slicer.dicomDatabase.openDatabase(str(databaseFilePath))
    if not slicer.dicomDatabase.isOpen:
        logger.error('Unable to reconnect database')
        raise Exception('Unable to connect to database')
    logger.info(f"Database opened at {databaseFilePath}")

def initDICOMDatabase():
    global DATABASE_INITIALIZED
    try:
        openDICOMDatabase()
    except Exception as e:
        logger.error(f"Error opening DICOM database: {e}")
        raise
    if DATABASE_INITIALIZED:
        logger.info("Database is already initialized")
        return
    DATABASE_INITIALIZED = True

def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response.text)
    if token:
        confirm_url, confirm_params = get_confirm_form_params(response.text, token)
        if confirm_url:
            response = session.get(confirm_url, params=confirm_params, stream=True)
            if 'text/html' in response.headers.get("Content-Type"):
                return 0
            else:
                save_response_content(response, destination)
                return 1
    else:
        params = {'id': file_id, 'export': 'download'}
        response = session.get(URL, params=params, stream=True)
        if 'text/html' in response.headers.get("Content-Type"):
            return 0
        else:
            save_response_content(response, destination)
            return 1

def get_confirm_token(response_text):
    soup = BeautifulSoup(response_text, 'html.parser')
    for input_tag in soup.find_all('input'):
        if input_tag.get('name') == 'confirm':
            return input_tag.get('value')
    return None

def get_confirm_form_params(response_text, token):
    soup = BeautifulSoup(response_text, 'html.parser')
    download_form = soup.find('form', {'id': 'download-form'})
    if download_form:
        action_url = download_form.get('action')
        confirm_params = {}
        for input_tag in download_form.find_all('input'):
            if 'name' in input_tag.attrs and 'value' in input_tag.attrs:
                confirm_params[input_tag['name']] = input_tag['value']
        confirm_params['confirm'] = token
        if action_url:
            if not action_url.startswith('http'):
                action_url = 'https://drive.google.com' + action_url
            return action_url, confirm_params
    return None, None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                
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

