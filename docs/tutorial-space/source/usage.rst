Usage
----------------

SPACE provide api function used in jupyter notebook   

    
API function
^^^^^^^^^^^^
Use SPACE in jupyter notebook::

    import space as sp
    adata=sp.SPACE(adata,outdir='sp-celltype',GPU=0,patience=50,alpha=0.05,epoch=5000)
    adata=sp.SPACE(adata,outdir='tissuemodule',GPU=0,patience=50,alpha=0.5,epoch=5000)
