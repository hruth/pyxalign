io
====

File Loading/Preparation
-------------------------

PEAR
^^^^^
pyxalign options and functions for loading datasets prepared using the PEAR
branch of pty-chi.

.. member-order doesn't trickle down to the attributes of the dataclasses, 
.. so they must be done explicitly

.. autofunction:: pyxalign.io.loaders.pear.load_data_from_pear_format

.. autoclass:: pyxalign.io.loaders.pear.BaseLoadOptions
   :members:
   :show-inheritance:
   :undoc-members:
   :member-order: bysource

.. autoclass:: pyxalign.io.loaders.pear.LYNXLoadOptions
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: pyxalign.io.loaders.pear
   :members:
   :show-inheritance:
   :undoc-members:
   :inherited-members:
   :exclude-members: BaseLoadOptions, load_data_from_pear_format, LYNXLoadOptions, LoaderType
   :member-order: bysource

.. autoclass:: pyxalign.io.loaders.pear.LoaderType
   :members:
   :member-order: bysource

XRF
^^^
pyxalign options and functions for loading x-ray flourescence (XRF) datasets.

.. autofunction:: pyxalign.io.loaders.xrf.load_data_from_xrf_format

.. autoclass:: pyxalign.io.loaders.xrf.XRFLoadOptions
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

.. automodule:: pyxalign.io.loaders.xrf
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: XRFLoadOptions

Data Structures
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyxalign.io.loaders.StandardData
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: