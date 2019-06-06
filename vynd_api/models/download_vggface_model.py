
import gdown
import os
import patoolib
import rarfile

# from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path

cur_dir = os.path.dirname(__file__)
pb_vgg_path = Path(cur_dir, './vggface2.pb')

def download_vggface_model():
    if(vgg_exist() == False):
        id_vgg_pb = '13Q3N7hxGP91hBzkLlU-7aEhemZQ-wRma'
        pb_file_link = 'https://drive.google.com/uc?id=' + id_vgg_pb
        gdown.download(pb_file_link, str(pb_vgg_path), False) # download rar file from google drive
    else:
        print("VGGFace2 .pb file is already installed")
        
def vgg_exist():
    if(os.path.exists(str(Path(cur_dir, './vggface2.pb'))) == False):
        return False
    return True