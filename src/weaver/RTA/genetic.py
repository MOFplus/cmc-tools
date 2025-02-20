#
# genetic.py
# taken from http://code.activestate.com/recipes/199121-a-simple-genetic-algorithm/
# created by Sean Ross
#
# revised version for MOF structure search with weaver:
#    - only integer numbers as alleles
#    - allow for a different range for each chromosome
#    - write and read restart
#    - discard offspring if it is already in the population (to maintain diversity)
#    - keeps a register of already calcualted (we do opt only once)
#
#      R. Schmid RUB 2012
#     

import random
import string
import os
import sys
import numpy
from molsys.molsys_mpi import mpiobject
MAXIMIZE, MINIMIZE = 11, 22
from mpi4py import MPI

class Individual(mpiobject):

    def __init__(self, world, chromosome=None,mpi_comm=None,out=None):
        super(Individual,self).__init__(mpi_comm, out)
        self.world = world
        self.chromosome = chromosome or self._makechromosome()
        self.chromestring = ":".join([str(i) for i in self.chromosome])
        self.score = None  # set during evaluation
        return
    
    def _makechromosome(self):
        "makes a chromosome from randomly selected alleles."
        if self.mpi_rank  == 0:
            randchrom = [random.randint(0,self.world.alleles[gene]-1) for gene in range(self.world.length)]
        else:
            randchrom = None
        randchrom = self.mpi_comm.bcast(randchrom)
        return randchrom

    def evaluate(self, optimum=None):
        "this method MUST be overridden to evaluate individual fitness score."
        if not self.score:
            if self.chromestring in self.world.register:
                # self.pprint("Chromosome %s is known in registry" % self.chromestring)
                known = self.world.register[self.chromestring]
                self.score = known.score
                self.myid  = known.myid
            else:
                self.myid = self.world.speciescounter
                self.world.speciescounter += 1
                self.score = self.world.calc_score(self.chromosome, self.myid)
                # register myself
                self.world.set_register(self)
        return
    
    def crossover(self, other):
        "override this method to use your preferred crossover method."
        m1, m2 = self._twopoint(other)
        m1.invalidate_score()
        m2.invalidate_score()
        return m1, m2
    
    def mutate(self, gene):
        "override this method to use your preferred mutation method."
        self._pick(gene)
        self.invalidate_score()
        return
    
    # sample mutation method
    def _pick(self, gene):
        "chooses a random allele to replace this gene's allele."
        if self.mpi_rank  == 0:
            chrom = random.randint(0,self.world.alleles[gene]-1)
        else:
            chrom = None
        if self.mpi_size > 1: chrom = self.mpi_comm.bcast(chrom)
        self.chromosome[gene] = chrom
    
    # sample crossover method
    def _twopoint(self, other):
        "creates offspring via two-point crossover between mates."
        left, right = self._pickpivots()
        def mate(p0, p1):
            chromosome = p0.chromosome[:]
            chromosome[left:right] = p1.chromosome[left:right]
            child = p0.__class__(p0.world, chromosome)
            #child._repair(p0, p1)
            return child
        return mate(self, other), mate(other, self)

    # some crossover helpers ...
    def _repair(self, parent1, parent2):
        "override this method, if necessary, to fix duplicated genes."
        pass

    def _pickpivots(self):
        if self.mpi_rank == 0:
            left = random.randrange(1, self.world.length-2)
            right = random.randrange(left, self.world.length-1)
        else:
            left,right = None, None
        if self.mpi_size > 1: 
            left = self.mpi_comm.bcast(left)
            right = self.mpi_comm.bcast(right)
        return left, right

    def invalidate_score(self):
        self.score = None
        return    

    #
    # other methods
    #

    def __repr__(self):
        "returns string representation of self"
        return '<"%s" score=%10.5f id=%5d>' % \
               (self.chromestring, self.score, self.myid)

    def __cmp__(self, other):
        if self.world.optimization == "min":
            return cmp(self.score, other.score)
        else: # MAXIMIZE
            return cmp(other.score, self.score)
    
    def copy(self):
        twin = self.__class__(self.world, self.chromosome[:])
        twin.score = self.score
        twin.myid  = self.myid
        return twin


class Environment(mpiobject):
    def __init__(self, name="run", kind=Individual, restart=False, size=100, maxgenerations=100, 
                 crossover_rate=0.90, mutation_rate=0.01, optimum=None, score_func=None, alleles=[2,2,2], optimization="min",
                 mpi_comm=None,out=None):
        super(Environment,self).__init__(mpi_comm, out)
        self.name = name
        # open report
        self.repfile  = open(self.name+".ga_report","a")
        self.statf = open(self.name+"_ga_stat.dat", "w")
        #
        self.alleles = alleles
        self.length  = len(alleles)
        self.optimization = optimization
        self.kind = kind
        self.size = size
        self.optimum = optimum
        self.calc_score = score_func
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.maxgenerations = maxgenerations
        self.generation = 0
        # other settings
        self.tselect_choosebest = 0.90
        self.tselect_size       = 8
        # this register keeps all individuals. if the chromsome was already there we do not reavaluate it
        self.register = {}
        self.speciescounter = 0
        if os.path.isfile(self.name+".ga_reg"):
            self.pprint("Registry exists -> reusing it!!  make sure that this is ok for you!!")
            self.read_register()
        self.regfile = open(self.name+".ga_reg", "a")
        #
        # start up 
        if not restart:    
            self.population = self._makepopulation()
            for individual in self.population:
                individual.evaluate(self.optimum)
        else:
            self.read_restart()
        #sorted_list = sorted(self.population.menteesdd, key=lambda p: p.name)
        self.population.sort(key=lambda x: x.score)
        #self.population.sort_register()
        self.report()
        return

    def set_tselect_params(self, size, choosebest):
        self.tselect_size = size
        self.tselect_choosebest = choosebest
        return

    def _makepopulation(self):
        return [self.kind(self) for individual in range(self.size)]
    
    def run(self):
        while not self._goal():
            self.step()

    def sort_register(self, tofile = True):
        ### return sorted registry
        nind = len(self.register)
        dtype = [('chromosome', 'S10'), ('id', int), ('score', float)]
        values = []
        for i in list(self.register.values()): values.append([i.chromestring, i.myid, i.score])
        values = numpy.array(values, dtype=dtype)
        values = numpy.sort(values, order = "energy")
        if tofile:
            if self.mpi_rank != 0: return values
            with open("%s.ga_sreg" % self.name) as f:
                for v in values: f.write("%s %5d %12.6f\n" % (v[0], v[1], v[2]))
        return values

    def _goal(self):
        return self.generation > self.maxgenerations or \
               self.best.score == self.optimum
    
    def step(self):
        self._crossover()
        self.generation += 1
        self.population.sort(key=lambda x: x.score)
        self.report()
        self.report_file()
        self.write_stat()
        self.write_restart()
        return
    
    def _crossover(self):
        # modifed to make sure that we do not produce all identical offsprings
        #  -> keep a list of ids and discard offspring if id already exists ... do not want all these mutants
        next_population = [self.best.copy()]
        next_idlist     = [self.best.myid]
        while len(next_population) < self.size:
            mate1 = self._select()
            if self.mpi_rank == 0:
                rnd = random.random()
            else: 
                rnd = None
            if self.mpi_size > 1: rnd = self.mpi_comm.bcast(rnd)
            if rnd < self.crossover_rate:
                mate2 = self._select()
                offspring = mate1.crossover(mate2)
            else:
                offspring = [mate1.copy()]
            for individual in offspring:
                self._mutate(individual)
                individual.evaluate(self.optimum)
                # at this point the id of this new individual is known.
                # append to next population only if its id is not known, yet
                if not next_idlist.count(individual.myid):
                    next_population.append(individual)
                    next_idlist.append(individual.myid)
                # DEBUG
                #else:
                #    self.pprint("id %d rejected" % individual.myid)
                # DEBUG
        # truncate population to proper size 
        self.population = next_population[:self.size]

    def _select(self):
        "override this to use your preferred selection method"
        return self._tournament()
    
    def _mutate(self, individual):
        for gene in range(self.length):
            if self.mpi_rank == 0:
                rnd = random.random()
            else: 
                rnd = None
            if self.mpi_size > 1: rnd = self.mpi_comm.bcast(rnd)
            if rnd < self.mutation_rate:
                individual.mutate(gene)

    #
    # sample selection method
    #
    #def _tournament(self):
    #    if self.mpi_rank == 0:
    #        competitors = [random.choice(self.population) for i in range(self.tselect_size)]
    #    else:
    #        competitors = None
    #    if self.mpi_size > 1: competitors = self.mpi_comm.bcast(competitors)
    #    competitors.sort()
    
    def _tournament(self):
        if self.mpi_rank == 0:
            idx_range = list(range(len(self.population)))
            random.shuffle(idx_range) 
            competitors_idx = idx_range[0:self.tselect_size] 
        else:
            competitors_idx = None
        if self.mpi_size > 1: competitors_idx = self.mpi_comm.bcast(competitors_idx,root=0)
        competitors = [self.population[xx] for xx in competitors_idx]
        competitors.sort(key=lambda x: x.score)
        if self.mpi_rank == 0:
            rnd = random.random()
        else: 
            rnd = None
        if self.mpi_size > 1: rnd = self.mpi_comm.bcast(rnd)
        if rnd < self.tselect_choosebest:
            return competitors[0]
        else:
            if self.mpi_rank == 0:
                randchoice = random.randint(1,len(competitors)-1) # beware!!!! the python random module
                                                                  # includes the end point in randint!
            else:
                randchoice = None
            if self.mpi_size > 1: randchoice = self.mpi_comm.bcast(randchoice)
            return competitors[randchoice]
    
    def best():
        doc = "individual with best fitness score in population."
        def fget(self):
            return self.population[0]
        return locals()
    best = property(**best())

    def report_file(self):
        if self.mpi_rank != 0: return
        self.repfile.write("="*70+"\n")
        self.repfile.write("generation: %5d\n" % self.generation)
        for i in self.population: self.repfile.write("%s\n" % i)
        self.repfile.write ("best:       %s\n" % str(self.best))
        return

    def report(self):
        self.pprint("="*70)
        self.pprint("generation: %5d" % self.generation)
        for i in self.population: self.pprint("%s" % i)
        self.pprint("best:       %s" % str(self.best))
        return
        
    def write_stat(self):
        if self.mpi_rank != 0: return
        scores = []
        for p in self.population: scores.append(p.score)
        bestscore  = scores[0]
        worstscore = scores[-1]
        avrgscore  = sum(scores)/len(scores)
        self.pprint("best/worst (avrg)  : %10.5f/%10.5f (%10.5f)" % (bestscore, worstscore, avrgscore))
        self.statf.write("%5d %10.5f %10.5f %10.5f\n" % (self.generation, bestscore, worstscore, avrgscore))
        return
        
    def set_register(self, buddy):
        self.register[buddy.chromestring] = buddy
        if self.mpi_rank != 0: return
        self.regfile.write("%s %5d %12.6f\n" % (buddy.chromestring, buddy.myid, buddy.score))
        self.regfile.flush()
        return
       
    def write_restart(self):
        if self.mpi_rank != 0: return
        rf = open(self.name+".ga_restart","w")
        rf.write("%d\n" % self.generation)
        for i in self.population:
            rf.write("%s %5d %12.6f\n" % (i.chromestring, i.myid, i.score))
        rf.close()
        return

    def read_register(self):
        self.pprint("Reading existing register file %s.ga_reg to avoid recomputing known individuals" % self.name)
        # read register ... so we open it again as another file instance with read
        regf = open(self.name+".ga_reg", "r")
        line = regf.readline().split()
        while len(line)> 0:
            cstr = line[0]
            myid = int(line[1])
            score= float(line[2])
            chromosome = [int(i) for i in cstr.split(":")]
            individual = self.kind(self, chromosome)
            individual.myid = myid
            individual.score= score
            self.register[cstr] = individual
            self.speciescounter += 1
            line = regf.readline().split()
        regf.close()
        return

    def read_restart(self):
        self.pprint("Reading restart %s.ga_restart" % self.name)
        # now we read the actual restart file (only the generation and the chromestrings are actually used because we can take our species
        #                                      from the register)
        resf = open(self.name+".ga_restart", "r")
        line = string.split(resf.readline())
        self.generation = string.atoi(line[0])
        self.population = []
        for i in range(self.size):
            line = string.split(resf.readline())
            cstr = line[0]
            self.population.append(self.register[cstr])
        resf.close()
        return
            
        
    
if __name__ == "__main__":
    def score(chrome, dummy):
        return sum(chrome)
        
    env = Environment(size=50, maxgenerations=100, optimum=0, score_func=score, alleles = 10*[4])
    env.run()

