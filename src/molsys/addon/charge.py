"""
    
                charge addon

    RS 2023 : simple addon to handle charges of molsys objects.

    its primary use is to extract charges from an exisiting ff addon

    additional methods like reading from file or generating by other means could be added 

"""

from molsys.addon import base
from molsys.util import chargemodels
import numpy as np


# the available charge fluctuation models area assigned to their nicknames here for easy access
models = {
    "qeq"    :chargemodels.qeq,      
    "qeq_cv" :chargemodels.qeq_core_valence,
    "acks2"  :chargemodels.acks2,
    "topoqeq":chargemodels.topoqeq
}

class charge(base):

    def __init__(self, mol):
        super(charge,self).__init__(mol)
        self.natoms = self._mol.get_natoms()
        self.q = np.zeros([self.natoms])
        self.method = None
        return
    
    def zero_charges(self):
        # do we need to use charges or is all zero
        return (self.q == 0.0).all()

    def get_from_ff(self):
        """generate charges from ff addon

        this will fail if there is no ff addon loaded

        Returns:
            ndarray: charges
        """
        assert "ff" in self._mol.loaded_addons 
        # set up the delta charges dictionary
        delta_chrg = {}
        for k in self._mol.ff.par["chapr"]:
            if self._mol.ff.par["chapr"][k][0] == "delta":
                delta = self._mol.ff.par["chapr"][k][1][0]
                at1, at2 = k.split("(")[1].split(")")[0].split(",")
                delta_chrg[at1] = (at2, delta)
        # compute charges
        conn = self._mol.get_conn()
        for i in range(self.natoms):
            chat  = self._mol.ff.parind["cha"][i][0]
            chrgpar    = self._mol.ff.par["cha"][chat]
            assert chrgpar[0] in ["gaussian","point"], "Only Gaussian or point charges supported"   # TODO
            self.q[i] += chrgpar[1][0]
            # check if chat in delta_chrg
            chat_red = chat.split("(")[1].split(")")[0]
            if chat_red in delta_chrg:
                at2, delta = delta_chrg[chat_red]
                # chek if any of the bonded atoms is of type at2
                for j in conn[i]:
                    if repr(self._mol.ff.ric.aftypes[j]) == at2:   # Note: aftypes are aftype objects and not strings .. we call repr() to get the string
                        # print ("atom %s connected to atom %s ..apply delta %f" % (chat_red, at2, delta))
                        self.q[i] += delta
                        self.q[j] -= delta
        self.method = "ff"
        return self.q
    
    def get_from_file(self, fname):
        f = open(fname, 'r')
        charges = []
        for i in range(self._mol.get_natoms()):
            line = f.readline()
            charges.append(float(line.split()[-1]))
        self.q = np.array(charges)
        self.method = "file"
        return self.q

    def set_model(self, model, dtype, cutoff, q_tot, *args, **kwargs):
        """
        sets the model to be used for charge generation

        Parameters:
            model (str)   : nickname of the charge fluctuation model to be employed
                            choose from: qeq, qeq_cv, acks2, topoqeq
            dtype (str)   : charge distribution type used to compute the electrostatic interactions
                            choose from: gauss, gauss_wolf, gauss_dsf, slater, point
            cutoff (float): distance cutoff for computation of electrostatic interactions
            q_tot (int)   : total charge of the system
            *args (in order) and **kwargs:
                model = qeq, qeq_cv, acks2 and topoqeq:
                    qeq_parfile (str, optional)  : name of the file containing the qeq parameters,
                                                   qeq, qeq_cv, acks2: if none is given all qeq parameters are initialized as 0.0,
                                                   topoqeq: if none is given they are taken from the internal chargeparam database,
                                                   default is None
                model = acks2:
                    acks2_parfile (str, optional): name of the file containing the non-interacting response parameters,
                                                   if none is given all non-interacting response parameters are initialized as 0.0,
                                                   default is None
                model = topoqeq:
                    rule (int, optional)         : sets level of atom type diversification
                                                   choose from 1 - full atom type, 2 - hybridization type (TODO), 3 - element type (TODO),
                                                   default is 1
                    from_ff (bool, optional)     : (TODO: not yet functional) if set to True, r_0 values are taken from loaded ff addon,
                                                   else they are taken from the internal chargeparam database,
                                                   default is False
                    bl_file (str, optional)      : name of the file containing the topological bond lengths,
                                                   if none is given they are taken from the internal chargeparam database
        """
        self.model = models[model](self._mol, dtype, cutoff, q_tot, *args, **kwargs)
        self.method = model
        return
    
    def get_from_model(self):
        """
        generates charges from the currently set charge fluctuation model

        Returns:
            ndarray: charges
        """
        self.model.calc()
        self.q[:] = self.model.q[:]
        return self.q
    
    def get_from_topoqeq(self, dtype="slater", cutoff=1000.0, q_tot=0, rule=2, from_ff=False, qeq_parfile=None):
        """
        convenience method for charge generation from topological QEq model
        exists for legacy reasons

        Returns:
            ndarray: charges
        """
        self.set_model("topoqeq", dtype, cutoff, q_tot, rule, from_ff, qeq_parfile)
        self.get_from_model()
        return self.q
    
