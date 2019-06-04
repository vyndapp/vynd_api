
import gdown
import os
import patoolib
from rarfile import RarFile

# from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path

cur_dir = os.path.dirname(__file__)
rar_vgg_path = Path(cur_dir, './vggface.rar')
vgg_path = Path(cur_dir, './')

def download_vggface_model():
    if(vgg_exist() == False):
        id_vgg = '1eqd-NRBc6JR_gUtIXt-ZO7gargrvwsjn'
        # GDOWN LIB
        # id_test = '1XBDOcqsnN9uk6EQY2OgQR97QOLc7Mpvj'
        pb_file_link = 'https://drive.google.com/uc?id=' + id_vgg
        gdown.download(pb_file_link, str(rar_vgg_path), False) # download rar file from google drive
        rar_file = RarFile(str(rar_vgg_path))
        rar_file.extractall()
        # print(rar_file.namelist())
        # patoolib
        # patoolib.extract_archive(str(rar_vgg_path), outdir=str(vgg_path)) # extract the file
        
        os.remove(rar_vgg_path) # remove the rar file
    else:
        print("VGGFace2 .pb file is already installed")
        
def vgg_exist():
    if(os.path.exists(str(Path(cur_dir, './vggface2'))) == False):
        return False
    return True