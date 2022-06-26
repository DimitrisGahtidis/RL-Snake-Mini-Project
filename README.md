# Conda environment setup
At the root of this repo is spec-file.txt, this file contains all the information needed to set up a conda virtual environment identical to that used during development.

After installing conda and python navigate to the spec-file.txt directory using the anaconda prompt (base). To set up the virtual environment run

`conda create --name myenv --file spec-file.txt`

Check that the virtual The virtual environment is created successfully by running
  
`conda info --envs` 
  
or 
  
`coda env list`
  
If myenv shows on the list it is available for activation. To activate it run

`conda activate myenv`

This conda terminal can now execute the appropriate python files by running

`python fileiwanttoexecute.py`