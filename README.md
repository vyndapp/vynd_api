## Recommended steps for setup

### Make sure you have Python 3

### Install **virtualenv** using pip3

    pip3 install virtualenv 
    

### Now create a virtual environment in the project's root folder

    virtualenv env 

>you can use any name insted of **env**


### Activate your virtual environment:    
    
    source env/bin/activate
    
### Install dependencies
    pip install -r requirements.txt
    
> Make sure to update **requirements.txt** if you add new dependencies

### Run tests    
    
    python -m unittest -v
   
> All test scripts must be added in the "test/" directory, and must start with the "test_" prefix

### Download face recognition models (VGGFace2)

    python setup.py
   
> This script downloads the compressed file containing the model weights, extracts the weights, places the extracted file in **models** directory.
