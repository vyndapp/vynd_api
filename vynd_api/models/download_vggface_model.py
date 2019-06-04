
# import gdown
import os
# import patoolib

from google_drive_downloader import GoogleDriveDownloader as gdd
from pathlib import Path

cur_dir = os.path.dirname(__file__)
rar_vgg_path = Path(cur_dir, './vggface.rar')
vgg_path = Path(cur_dir, './')

def download_vggface_model():
    if(os.path.exists(str(Path(cur_dir, './vggface2'))) == False):
        id_vgg = '1eqd-NRBc6JR_gUtIXt-ZO7gargrvwsjn'
        # GDOWN LIB
        # id_test = '1XBDOcqsnN9uk6EQY2OgQR97QOLc7Mpvj'
        # pb_file_link = 'https://drive.google.com/uc?id=' + id_vgg
        # gdown.download(pb_file_link, str(rar_vgg_path), False) # download rar file from google drive
        # patoolib.extract_archive(str(rar_vgg_path), outdir=str(vgg_path)) # extract the file
        # os.remove(rar_vgg_path) # remove the rar file
        
        # GDD LIB
        gdd.download_file_from_google_drive(file_id=id_vgg,
                                            dest_path=str(rar_vgg_path),
                                            unzip=True,
                                            showsize=True)
    else:
        print("VGGFace2 .pb file is already installed")