io
====

File Loading/Preparation
-------------------------

The functions and options classes in this module are used to load laminography/tomography datasets into a standardized format.

pyxalign currently supports ptychography and x-ray flourescence (XRF) data.

**How to Load Data**

Find the appropriate loading options class for your beamline:

* Beamline 2-ID-E
    * Ptychography data loading options: :class:`pyxalign.io.loaders.pear.Microprobe2IDELoadOptions`
    * XRF data loading options: :class:`pyxalign.io.loaders.xrf.XRF2IDELoadOptions`
* Beamline 2-ID-D
    * Ptychography data loading options: :class:`pyxalign.io.loaders.pear.BNP2IDDLoadOptions`
* Beamline 31-ID-E
    * Ptychography data loading options: :class:`pyxalign.io.loaders.pear.LYNXLoadOptions`

To load the data, you can either: 1) use the appropriate loading function or 2) use the data loading GUI.

* Loading functions:
    * Ptychography data processed by PEAR/Pty-Chi: :func:`pyxalign.io.loaders.pear.load_data_from_pear_format`
    * XRF Data: :func:`pyxalign.io.loaders.xrf.load_data_from_xrf_format`
* Loading GUI:
    * Any type of data: :func:`pyxalign.gui.launch_data_loader`


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

.. .. autoclass:: pyxalign.io.loaders.xrf.XRFLoadOptions
..    :members:
..    :inherited-members:
..    :undoc-members:
..    :show-inheritance:
..    :member-order: bysource

.. automodule:: pyxalign.io.loaders.xrf
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: load_data_from_xrf_format

Data Structures
^^^^^^^^^^^^^^^^^^^
.. autoclass:: pyxalign.io.loaders.StandardData
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: