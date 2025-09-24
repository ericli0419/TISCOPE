[![Stars](https://img.shields.io/github/stars/ericli0419/TISCOPE?logo=GitHub&color=yellow)](https://github.com/ericli0419/TISCOPE/stargazers)
<!-- [![PyPI](https://img.shields.io/pypi/v/scalex.svg)](https://pypi.org/project/scalex) -->
<!-- [![Documentation Status](https://readthedocs.org/projects/scalex/badge/?version=latest)](https://scalex.readthedocs.io/en/latest/?badge=stable) -->
<!-- [![Downloads](https://pepy.tech/badge/scalex)](https://pepy.tech/project/scalex) -->
<!-- [![DOI](https://zenodo.org/badge/345941713.svg)](https://zenodo.org/badge/latestdoi/345941713) -->
# [TISCOPE enables integrative and comparative analyses of spatial omics data  to reveal condition-associated tissue modules]()

<p align="center" style="background-color: white;">
    <img src="TISCOPE.png" alt="TISCOPE" style="background-color: white;">
</p>


## News

<!-- ## [Documentation](https://scalex.readthedocs.io/en/latest/index.html)  -->
<!-- ## [Tutorial](https://scalex.readthedocs.io/en/latest/tutorial/index.html)  -->
## Installation  	

1. We recommend creating a virtual environment using Python 3.11:
```
conda create -n tiscope python=3.11
conda activate tiscope
```
2. (Skip if PyTorch is already installed) install PyTorch following the [PyTorch installation guide](https://pytorch.org/get-started/locally/). For example, on a machine with `cuda 12.x`:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
3. (Skip if PyG is already installed) Install PyG following the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), usually:
```
pip install torch_geometric
```
4. Install dependencies:
```
pip install 'scanpy[leiden]' louvain squidpy ipykernel
```
5. Install TISCOPE:

Via Pypi:
```
pip install tiscope
```
or git clone and install
```
git clone git://github.com/ericli0419/TISCOPE.git
cd tiscope
pip install -e .
```
## Getting started

Please refer to the [Documentation]() and [Tutorial]()

    
## Release notes

See the [changelog](https://github.com/ericli0419/TISCOPE/CHANGELOG.md).  


## Citation


    