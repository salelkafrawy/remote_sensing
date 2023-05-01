!/bin/bash
python3.9 -m venv /home/sara246/glc_venv
source /home/sara246/glc_venv/bin/activate
pip install --no-index torch torchvision
pip install pytorch-lightning
pip install wandb
pip install hydra-core --upgrade

pip install lightning-bolts
pip install pandas
pip install numpy
pip install Pillow
pip install tifffile
pip install -U scikit-learn
pip install matplotlib

pip install ipykernel
python -m ipykernel install --user --name=glc_venv