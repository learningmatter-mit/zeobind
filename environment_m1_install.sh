mamba create -n zeobind python=3.8.12 -c conda-forge -y
mamba activate zeobind
pip3 install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 scikit-learn==1.0.2 PyYAML sigopt
mamba install -y -c conda-forge rdkit beautifulsoup4=4.11.1 matplotlib=3.3.2 networkx=3.0 tqdm=4.59.0 xgboost=1.6.2 yaml=0.2.5 