io
====
The functions and options classes in this module are used to load laminography/tomography datasets into a standardized format.

The following input formats are currently supported:

* Ptychography reconstructions created using the PEAR wrapper for Pty-Chi (beamlines 31-ID-E, 2-ID-E, 2-ID-D)
* XRF datasets (beamline 2-ID-E)

File Loading/Preparation
-------------------------

PEAR
^^^^^
`pyxalign.io.loaders.pear` contains options and functions for loading datasets prepared using the PEAR branch of Pty-Chi.

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
`pyxalign.io.loaders.xrf` contains options and functions for loading x-ray flourescence (XRF) datasets.

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
   :exclude-members: XRFLoadOptions, load_data_from_xrf_format

Data Structures
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyxalign.io.loaders.StandardData
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: