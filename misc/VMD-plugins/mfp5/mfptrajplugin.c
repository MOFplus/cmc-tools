#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "molfile_plugin.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include "libmfp5.h"


const char default_name[16]="O";
const char default_type[16]="O";
const char default_resname[8]="";
const int default_resid= 1;
const char default_segid[8]= "";
const char default_chain[2]= "";
const char default_altloc[2]= "";
const char default_insertion[2]= "";
const float default_occupancy= 1.0;
const float default_bfactor= 1.0; 
const float default_mass= 1.0;
const float default_charge= 0.0;
const float default_radius= 0.5;
const int default_atomicnumber= 1;

static const char *element_symbols[] = { 
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P" , "S",  "Cl", "Ar", "K",  "Ca", "Sc",
    "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", 
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc",
    "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr",
    "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt",
    "Ds", "Rg"
};

int get_element_index(const char *elem){
    for (int i=0; i<112;i++){
        // printf("elemcmp %s %s %i\n", element_symbols[i],elem,strcmp(element_symbols[i],elem)); fflush(stdout);
        if (strcmp(element_symbols[i],elem) == 0){
            return i;
        }
    }
    return default_atomicnumber;
}

static void *open_mfp5_read(const char *filename, const char *filetype, int *natoms){
	struct mfp5_file* file;
	mfp5_open(&file,filename,-1);
	mfp5_get_natoms(file, natoms);
	return file;
}

void close_file(void* _file){
	struct mfp5_file* file=_file;
	mfp5_close(file);
	//TODO bonds from, to need to be freed, compare line 01050 http://www.ks.uiuc.edu/Research/vmd/plugins/doxygen/psfplugin_8c-source.html
}

// Structure IO
//load whole VMD structure
int read_mfp5_structure(void *_file, int *optflags,molfile_atom_t *atoms) {
    printf("Entering read function\n\n");
    molfile_atom_t *atom;
	*optflags =  MOLFILE_ATOMICNUMBER | MOLFILE_CHARGE; 
	struct mfp5_file* file=_file;
	int natoms;
	mfp5_get_natoms(file, &natoms);

    char **elems, **atypes, **fragtypes;
    int *fragnumbers;
	mfp5_get_elems(file,(void**) &elems);
    printf ("mfp5_get_elems terminated properly\n");fflush(stdout);
    mfp5_get_atypes(file,(void**) &atypes);
    printf ("mfp5_get_atypes terminated properly\n");fflush(stdout);
    mfp5_get_fragtypes(file,(void**) &fragtypes);
    printf ("mfp5_get_fragtypes terminated properly\n");fflush(stdout);
    mfp5_get_fragnumbers(file,(void**) &fragnumbers);
    printf ("mfp5_get_fragnumbers terminated properly\n");fflush(stdout);
    for (int i = 0; i < natoms; i++) {

        atom = atoms + i;
        //printf ("%d\n",i);fflush(stdout);
        strncpy(atom->name,elems[i],16*sizeof(char));
        strncpy(atom->name,elems[i],16*sizeof(char));
        strncpy(atom->type,atypes[i],16*sizeof(char));
        strncpy(atom->resname,fragtypes[i],8*sizeof(char));
        atom->resid=fragnumbers[i];	
        atom->atomicnumber = get_element_index(elems[i]);
        //atom->name            
        //atom->type
        //atom->atomicnumber
        //atom->name
        //atom->mass
        //atom->radius
        //atom->charge
        //atom->segid
        //atom->resid
        //atom->chain
        //atom->resname
    }
    // user input for stage
    // todo: find avaiable stages and display them!
    printf("available stages:\n");
    mfp5_detect_stages(file);
    char stagename[100];
    printf("which stage should be displayed? ");
    scanf("%s", stagename);
    //const char stagename[] = "/produ_pramp";
    mfp5_set_stage(file, stagename);
    printf ("end of read_mfp5_structure reached\n");fflush(stdout);
	return MOLFILE_SUCCESS;
}

/* read the coordinates */
static int read_mfp5_timestep(void *_file, int natoms, molfile_timestep_t *ts) {
	struct mfp5_file* file=_file;
	int status=MOLFILE_SUCCESS;
	int current_time;
	mfp5_get_current_time(file,&current_time);
	int ntime;
	mfp5_get_ntime(file,&ntime);
	if(current_time>=ntime)
		return MOLFILE_ERROR;    
	if (ts != NULL) { //skip reading if ts is NULL pointer (needs modification of the timestep below)
		mfp5_get_natoms(file, &natoms);
		//read boxinformation
		float box_information[6] = {0,0,0,0,0,0};
		int status_box=mfp5_get_box_information(file ,current_time, &box_information);
        /*printf("BOX  %d %d %f \n\n", ntime, current_time, box_information[0]); fflush(stdout);
        printf("BOX  %d %d %f \n\n", ntime, current_time, box_information[1]); fflush(stdout);
        printf("BOX  %d %d %f \n\n", ntime, current_time, box_information[2]); fflush(stdout);
        printf("BOX  %d %d %f \n\n", ntime, current_time, box_information[3]); fflush(stdout);
        printf("BOX  %d %d %f \n\n", ntime, current_time, box_information[4]); fflush(stdout);
        printf("BOX  %d %d %f \n\n", ntime, current_time, box_information[5]); fflush(stdout);		
        */
        ts->A=box_information[0];
        ts->B=box_information[1];
        ts->C=box_information[2];
        ts->alpha=box_information[3];
        ts->beta=box_information[4];
        ts->gamma=box_information[5];
        ts->velocities = NULL;
		ts->physical_time = 0.0;
        // read xyz coordinates
        int status_xyz = mfp5_get_timestep(file,current_time,ts->coords);
	}
	int status_seek=mfp5_seek_timestep(file, current_time+1); //modify timestep in the internal state of the plugin for this file
    if(status_seek!=0){
		status= MOLFILE_SUCCESS;
	}else if(status_seek!=0 ){
		status= MOLFILE_ERROR;
	}
    return status;
}

static int mfp5_get_bonds(void *_file, int *nbonds, int **from, int **to, float **bondorder, int **bondtype,  int *nbondtypes, char ***bondtypename){
	*bondorder = NULL;
	*bondtype=NULL;
	*nbondtypes=0;
	bondtypename=NULL;
    int status;
	struct mfp5_file* file=_file; 
	H5T_class_t type_class_bond_from;
	H5T_class_t type_class_bond_to;

	status=mfp5_get_conn(file, nbonds, (void**) from, (void**) to);

	if(status==0){ 
		printf("read %d bonds \n",*nbonds);fflush(stdout);
  		return MOLFILE_SUCCESS;
	}else{
        printf("read bonds failed\n");fflush(stdout);
		return MOLFILE_ERROR;		
	}
}



/* VMD API REGISTRATION */
static molfile_plugin_t plugin;

VMDPLUGIN_API int VMDPLUGIN_init() {
	memset(&plugin, 0, sizeof(molfile_plugin_t));

	plugin.abiversion = vmdplugin_ABIVERSION;
	plugin.type = MOLFILE_PLUGIN_TYPE;
	plugin.name = "mfp5";
	plugin.prettyname = "MOFplus Trajectory File Format - former mfp5";
	plugin.author = "Julian Keupp";
	plugin.majorv = 1;
	plugin.minorv = 1;
	plugin.is_reentrant = VMDPLUGIN_THREADSAFE;
	plugin.filename_extension = "mfp5";
	plugin.open_file_read = open_mfp5_read; // done, not tested
	plugin.read_structure = read_mfp5_structure;
	plugin.read_next_timestep = read_mfp5_timestep;
	plugin.read_bonds = mfp5_get_bonds;
	plugin.close_file_read = close_file;
	//plugin.open_file_write = ;
	//plugin.write_timestep = ;
	//plugin.close_file_write = ;
	

	return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_register(void *v, vmdplugin_register_cb cb) {
	(*cb)(v, (vmdplugin_t *) &plugin);
	return VMDPLUGIN_SUCCESS;
}

VMDPLUGIN_API int VMDPLUGIN_fini() {
	return VMDPLUGIN_SUCCESS;
}
