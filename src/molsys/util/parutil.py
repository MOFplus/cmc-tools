from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import numpy as np

################################################################################
###
### USAGE
###     list_of_objects = parutil.init(list_of_object)
###     ### DO WHAT YOU WANT
###     list_of object = parutil.finish(list_of_objects)
###
################################################################################


def init(iterable):
    iterable = scatter(iterable)
    return iterable

def finish(iterable, barrier=True):
    iterable = bcast(gather(iterable))
    if barrier: comm.Barrier()
    return iterable

def scatter(iterable):
    """Scatter iterable as chunks to the cores.
    N.B.: len(iterable) == size after scattering!"""
    iterable = comm.scatter(chop(iterable))
    return iterable

def gather(iterable, keep_order=True, keep_type=True):
    iterable = comm.gather(iterable)
    if rank == 0:
        itertype = type(iterable)
        iterable = sum(iterable,[])
        if keep_order:
            natural = range(len(iterable))
            mixed = chop(natural)
            try:
                mixed = sum(mixed,[])
            except TypeError:
                ### list elements are generators!
                mixed = [list(i) for i in mixed]
                mixed = sum(mixed,[])
            order = np.argsort(mixed)
            iterable = np.array(iterable)[order]
            if keep_type == True:
                if itertype in [list, tuple, set, frozenset]:
                    iterable = itertype(iterable)
                elif itertype is not np.ndarray:
                    raise NotImplementedError("NOT TESTED")
    return iterable

def bcast(iterable):
    iterable = comm.bcast(iterable)
    return iterable

def chop(iterable):
    """Chop an iterable into (quasi)-equally long chunks.
    Automatically handle non-multiplier! ( len(iterable)%size != 0 )
    Core function for parallelization"""
    chunks = [iterable[i::size] for i in range(size)]
    return chunks
