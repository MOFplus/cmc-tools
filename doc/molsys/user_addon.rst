.. molsys documentation master file, created by
   sphinx-quickstart on Mon Aug 21 14:29:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Addons
################

Molsys features several addons. In general instances of addon classes can be attached to a mol object **m**
in the following way. 

.. code-block:: python

    >> m.addon("name_of_addon")

Methods and attributes of the corrsponding addon object are then available under m.name_of_addon.

.. toctree::
   :maxdepth: 3

   user_addon_ff

