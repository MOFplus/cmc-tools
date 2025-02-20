.. molsys documentation master file, created by
   sphinx-quickstart on Mon Aug 21 14:29:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


MOLSYS
###### 

*molsys* is a Python library (and a bunch of scripts and other useful stuff) to store, manipulate and analyze molecular systems.
It is developed in the `CMC group <http://www.rochusschmid.de>`_ at the `RUB <http://www.rub.de>`_ and serves as a foundation of a number
of ongoing projects like the Python wrapped force field engines `pydlpoly <https://github.com/MOFplus/pydlpoly>`_ and 
`pylmps <https://github.com/MOFplus/pylmps>`_ or `weaver <https://github.com/MOFplus/weaver>`_ and `ff_gen <https://github.com/MOFplus/ff_gen>`_.

Why MOLSYS?
***********

The best way to explain what *molsys* is, and what it is not, is to explain why it was made ... or better started, since it is still under development and 
constantly evolving. "Why *molsys*?" is also a good question, becasue there are already quite a number of *molsys*'ish libraries and codes out there and 
it is really the question whether one needs another one. However, historically we did not really think about this at all. It started with a script. When
we developed *pydlpoly* we needed a class to store the molecule with its atomtypes, connectivity information and so on. Later we wrote the first version of *weaver*
to implement the reversed topological approach to generate MOF and other network structures. In the end there was another *mol* class in *weaver*, very similar to
the one in *pydlpoly* and it was clear that we needed a common *mol* object system.

The first attempt was to start with the molecule or *atoms* class from *ASE* (`atomic simulation environemnt <https://wiki.fysik.dtu.dk/ase/>`_). *ASE* is really built 
for QM computations and lacks the atom type and connectivty information part, which is necessary for force field calculations (but also for representing e.g. embeddings of
network topologies). So we decided to go and extend *ASE* by these features. However, since *ASE* is a mighty big and powerful library, this turned out to be a tedious operation
if we would not restrict to a superficial on-top implementation. Thus, we ambandoned this plan and decided to extend what we already had in *pydlpoly* and *weaver*. Technically, 
the heart if *molsys* is the *mol* class defined in *mol.py*. A further aspect is that *ASE* intentionally stays away from dependincies and compiled code. However, we wanted
to build on other libraries (e.g. *openbabel*, *graphtool*), which are sometimes not straight forward to install. To this end we developed a system called "addon"s, which is
inspired by the component system of game engines like `unity <https://unity.com/>`_: addons are attached to a *mol* object (if the underlying libraries are available and installed)
to extend te capabilities of the *mol* object. This means that the core features of *molsys* can still be used if these extra libs are not installed. In the end we can now 
generate a *molsys* *mol* object from a *pymatgen* object or export to an *ASE* *atoms* object.

Meanwhile, *molsys* is the basis for all projects dealing with force field type calculations. Over time, with the `MOF+ web platform <https://www.mofplus.org>`_, also our own
specific file formats emerged, which are currently only readable by *molsys*. Thus, *molsys* is needed and no other could replace it (at least we do not know about it). With the addon
mechanism we are able to "borrow" from other systems (the cif-reader from *pymatgen* is so much better than our clumsy code) but of course it also has its limitations. 
This is why we still work on it and the code is always ahead of the documentation. Therefore, you should also read the next section.

How to use this document
************************

The problem with code development (especially in an academic environment) is that time and resources for testing and documentation is alwways barely existing. In other words:
our docu s....! Sorry for that. This means: this document is all we have for the moment and we try to extend it but the code will always be ahead. It might even be, that
the docu here is wrong and the way to use the code has changed. So please beware and always have a look at the sources, Luke! This is also the reason that for a long time *molsys*
and friends was only available for collborators and in a private repository. By reaching a critical size with this document we will make *molsys* available.

The API documentation from the docstrings should, however, be up to date if you have rebuild the docs.



.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: User Documentation

   user_quickstart
   user_installation
   user_addon
   user_util

.. toctree::
   :maxdepth: 1
   :numbered:
   :caption: Technical Stuff

   tech_api


.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: File Formats

   file_io






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
