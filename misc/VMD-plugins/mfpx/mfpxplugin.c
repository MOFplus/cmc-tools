#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>	
#include "molfile_plugin.h"

//#define DEBUG
#define PI 3.14159265

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
    "x",  "h",  "he", "li", "be", "b",  "c",  "n",  "o",  "f",  "ne",
    "na", "mg", "al", "si", "p" , "s",  "cl", "ar", "k",  "ca", "sc",
    "ti", "v",  "cr", "mn", "fe", "co", "ni", "cu", "zn", "ga", "ge", 
    "as", "se", "br", "kr", "rb", "sr", "y",  "zr", "nb", "mo", "tc",
    "ru", "rh", "pd", "ag", "cd", "in", "sn", "sb", "te", "i",  "xe",
    "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb",
    "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w",  "re", "os",
    "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn", "fr",
    "ra", "ac", "th", "pa", "u",  "np", "pu", "am", "cm", "bk", "cf",
    "es", "fm", "md", "no", "lr", "rf", "db", "sg", "bh", "hs", "mt",
    "ds", "rg"
};
/*static const char *element_symbols[] = { 
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
};*/


int get_element_index(const char *elem){
    for (int i=0; i<112;i++){
        // printf("elemcmp %s %s %i\n", element_symbols[i],elem,strcmp(element_symbols[i],elem)); fflush(stdout);
        if (strcmp(element_symbols[i],elem) == 0){
            return i;
        }
    }
    return default_atomicnumber;
}
/*
Possible Contents of the mfpx file headeR:
# type xyz, topo or bb
# if type is bb
    # bbcenter com or coc or special
    # bbconn connstring
# if type topo
    # body contains different information
# periodicity: 
    # if there is no info, the mfpx is not periodic
    # cell a b c alpha beta gamma
    # cellvect x1 y1 z1 x2 y2 z2 x3 y3 z3
The header lines are all indicated with a hashtag as first chracter
The first line after the header is always the number of atoms
afterwards the body follows
*/


char* copy( const char *s )
{
    char *a = malloc(strlen(s)+1);
    if (a == NULL)
    {
        perror( "malloc failed" );
        return NULL;
    }

    char *c = a;
    while( *s != '\0' )
    {
        *c = *s;
        s++;
        c++;
    }

    *c = '\0';
    return a;
}

float calculate_length_of_vector(float* vector, int dimensions){
	float length=0.0;
	for(int i=0;i<dimensions;i++){
	length+=vector[i]*vector[i];
	}
	length=sqrt(length);
	return length;
}

//returns 0 if "vector" is the zero vector, otherwise nonzero
int is_zero_vector(float* vector, int dimensions){
	float absolute_error=0.00001;
	int number_of_nonzero_entries=0;
	for(int i=0;i<dimensions;i++){
		if(fabs(vector[i]-0)>absolute_error){
				number_of_nonzero_entries+=1;
				break;
		}
	}
	return number_of_nonzero_entries;
}

float calculate_angle_between_vectors(float* vector1, float* vector2, int dimensions){
	if(is_zero_vector(vector1, dimensions)==0 || is_zero_vector(vector2, dimensions)==0){
		printf("ERROR: Cannot calculate angle to the zero vector which has length 0\n");
		return -1.0;
	}
	float angle=0.0;
	for(int i=0;i<dimensions;i++){
		angle+=vector1[i]*vector2[i];
	}
	float len_vector1=calculate_length_of_vector(vector1, dimensions);
	float len_vector2=calculate_length_of_vector(vector2, dimensions);
	angle=acos(angle/(len_vector1*len_vector2))*180/PI;	//*180/PI converts radian to degree
	return angle;
}

struct mfpx_file{
	FILE * file_id;
    char *filename;
    char **header;
	int natoms;
    int mfpxtype;
    int nheader_lines;
    int body_parsed;
    int is_periodic;
    float cellvect[9];
    float cellparams[6];
    float* xyz;
    int* conn_from;
    int* conn_to;
    int nbonds;
};
/*  ********** DESCRIPTION ********** 
* file_id holds the reference to the open file
* header contains the strings of the header for later use
* nheader_lines contains the number of header lines.
* natoms contains the number of atoms
* mfpxtype defines the type of the mfpx file
    0: xyz type
    1: bb type
    2: topo type

*/

int mfpx_open(struct mfpx_file** _file, const char *filename){
    struct mfpx_file *file = malloc(sizeof(struct mfpx_file));
    file->filename = copy(filename);
    file->file_id = fopen(filename,"r"); // open file in read only mode
    *_file = file;
}

int parse_header_line(struct mfpx_file* file, const char *line){
    char *s;
    float aa,bb,cc,al,be,ga;
    // check for type
    #ifdef DEBUG
        printf("parsing header line ... \n");
    #endif
    //char typechar[] = '# type'
    s = strstr(line,"# type");
    if (s != NULL){
        // type can either be xyz, bb or topo
        s = strstr(line,"xyz");
        if (s != NULL){
            #ifdef DEBUG
            printf("xyz type detected ... \n");
            #endif
            file->mfpxtype=0;
        }
        s = strstr(line,"bb");
        if (s != NULL){
            #ifdef DEBUG
                printf("bb type detected ... \n");
            #endif
            file->mfpxtype=1;
        }   
        s = strstr(line,"topo");
        if (s != NULL){
            #ifdef DEBUG
                printf("topo type detected ... \n");
            #endif
            file->mfpxtype=2;
        }
    }  
    // check for cell parameters
    s = strstr(line,"# cellvect");
    if (s != NULL){
        // the character array contains nine 
        #ifdef DEBUG
            printf("cellvect info detected ... \n");
        #endif
        //char *p=line+10;
        char *p=line+9;
        //int numFloats = sscanf(line,"%*s %*s %f %f %f %f %f %f %f %f %f",
        int numFloats = sscanf(p,"%*s %f %f %f %f %f %f %f %f %f",
        &file->cellvect[0],
        &file->cellvect[1],
        &file->cellvect[2],
        &file->cellvect[3],
        &file->cellvect[4],
        &file->cellvect[5],
        &file->cellvect[6],
        &file->cellvect[7],
        &file->cellvect[8]);
        #ifdef DEBUG
            printf("c(0,0) = %f\n",file->cellvect[0]);
            printf("c(1,0) = %f\n",file->cellvect[1]);
            printf("c(2,0) = %f\n",file->cellvect[2]);
            printf("c(0,1) = %f\n",file->cellvect[3]);
            printf("c(1,1) = %f\n",file->cellvect[4]);
            printf("c(2,1) = %f\n",file->cellvect[5]);
            printf("c(0,2) = %f\n",file->cellvect[6]);
            printf("c(1,2) = %f\n",file->cellvect[7]);
            printf("c(2,2) = %f\n",file->cellvect[8]);
        #endif
        // calculate the cellparameters. VMD needs those
        float a[3];
        float b[3];
        float c[3];
        for (int i=0; i<3; i++){
            a[i] = file->cellvect[i];
        }
            for (int i=0; i<3; i++){
            b[i] = file->cellvect[i+3];
        }
            for (int i=0; i<3; i++){
            c[i] = file->cellvect[i+6];
        }
        aa = calculate_length_of_vector(a,3);
        file->cellparams[0] = aa;
        bb = calculate_length_of_vector(b,3);
        file->cellparams[1] = bb;
        cc = calculate_length_of_vector(c,3);
        file->cellparams[2] = cc;
        al = calculate_angle_between_vectors(b,c,3);
        file->cellparams[3] = al;
        be = calculate_angle_between_vectors(a,c,3);
        file->cellparams[4] = be;
        ga = calculate_angle_between_vectors(a,b,3);
        file->cellparams[5] = ga;
        file->is_periodic=1;
        #ifdef DEBUG
            printf("aa = %f\n",aa); fflush(stdout);
            printf("bb = %f\n",bb); fflush(stdout);
            printf("cc = %f\n",cc); fflush(stdout);
            printf("al = %f\n",al); fflush(stdout);
            printf("be = %f\n",be); fflush(stdout);
            printf("ga = %f\n",ga); fflush(stdout);
        #endif
    }
    s = strstr(line,"# cell ");
    if (s != NULL){
        #ifdef DEBUG
            printf("cell info detected ... \n"); fflush(stdout);
        #endif
        char *p=line+5;
        //int numFloats = sscanf(line,"%*s %*s %f %f %f %f %f %f %f %f %f",
        int numFloats = sscanf(p,"%*s %f %f %f %f %f %f",
        &file->cellparams[0],
        &file->cellparams[1],
        &file->cellparams[2],
        &file->cellparams[3],
        &file->cellparams[4],
        &file->cellparams[5]);
        file->is_periodic=1;        
        #ifdef DEBUG
            printf("a  = %f\n",file->cellparams[0]); fflush(stdout);
            printf("b  = %f\n",file->cellparams[1]); fflush(stdout);
            printf("c  = %f\n",file->cellparams[2]); fflush(stdout);
            printf("al = %f\n",file->cellparams[3]); fflush(stdout);
            printf("be = %f\n",file->cellparams[4]); fflush(stdout);
            printf("ga = %f\n",file->cellparams[5]); fflush(stdout);
        #endif
    }
    return 0;
}

int mfpx_get_natoms(struct mfpx_file* file, int* natoms){
    // This guy reads the header and the number of atoms and stores the header as a char ** into the struct
    char * line = NULL;
    size_t len = 0;
    size_t read;
    int natoms_tmp;
    int header_count = 0;
    int done = 0;
    file->is_periodic=0;
    while (done == 0) {
        read = getline(&line, &len, file->file_id);
        if (read == -1){ //terminate if line could not be red
            done = 1;
            continue;
        }
        //printf("%s", line);
        //printf("%c", line[0]);
        if (line[0] == '#'){
            // fill mfpx_file struct with infos
            parse_header_line(file,line);
        }
        else{
            // okay, we are done reading the header. the next line contains the number of atoms
            done = 1;
            natoms_tmp =  atoi(line);
            *natoms = natoms_tmp;
            file->natoms = natoms_tmp;
        }
    }
    file->body_parsed = 0;
}

static void *open_mfpx_read(const char *filename, const char *filetype, int *natoms){
	struct mfpx_file* file;
    int status;
	status = mfpx_open(&file,filename);
	status = mfpx_get_natoms(file, natoms);
	return file;
}

void close_file(void* _file){
	struct mfpx_file* file=_file;
	fclose(file->file_id);
}

// Structure IO
//load whole VMD structure
int read_mfpx_structure(void *_file, int *optflags,molfile_atom_t *atoms) {
    #ifdef DEBUG
        printf("reading mfpx file body ... \n\n");
    #endif
    molfile_atom_t *atom;
	*optflags =  MOLFILE_ATOMICNUMBER | MOLFILE_CHARGE; 
	struct mfpx_file* file=_file;
	int natoms = file->natoms;

    file->xyz       = malloc(natoms*3*sizeof(float));
    file->conn_from = malloc(natoms*24*sizeof(int));
    file->conn_to   = malloc(natoms*24*sizeof(int));
    file->nbonds=0;

    int j=0;
    int jidx;
    
    #ifdef DEBUG
        printf("natoms = %i\n", natoms);
    #endif
    char * line = NULL;
    size_t len = 0;
    size_t read;
    char * pch;
    char *outer,*inner, *pch_inner;

    for (int i = 0; i < natoms; i++) {
        read = getline(&line, &len, file->file_id);
        #ifdef DEBUG
            printf("(%i) %i: %s",i, len, line); fflush(stdout);
        #endif
        
        atom = atoms + i;
        // case mfpxtype = 0 (xyz)
        // idx, elem, x, y, z, atype, fragtype, fragnumber, conn 
        
        //pch = strtok (line," ");
        // since we sometimes have to split the splitted string once more, we have to use 
        // strtok_r here! r=reentrant. reason below
        // https://stackoverflow.com/questions/15961253/c-correct-usage-of-strtok-r
        pch = strtok_r(line," ",&outer);
        j = 0;
        while (pch != NULL){
            if (j == 1){
                // elem
                strncpy(atom->name,pch,16*sizeof(char));
                atom->atomicnumber = get_element_index(pch);
            }
            if (j >= 2 && j <=4){
                file->xyz[3*i+(j-2)] = atof(pch);
            }
            if (j == 5){
                strncpy(atom->type,pch,16*sizeof(char));
                //printf("%f %f %f", xyz[3*i],xyz[3*i+1],xyz[3*i+2]);
            }
            if (file->mfpxtype == 0){           
                if (j == 6){
                    strncpy(atom->resname,pch,8*sizeof(char));
                }
                if (j == 7){
                    atom->resid=atoi(pch);
                }
                if (j > 7){
                    // this is a conn entry!
                    jidx = atoi(pch);
                    // we have to prevent double counting here! 
                    // add a bond to the ctab only if the current atom idx is smaller than the bonded atom idx
                    if (i < jidx){
                        file->conn_from[file->nbonds]=i+1;
                        file->conn_to[file->nbonds]=jidx;
                        file->nbonds++;
                        #ifdef DEBUG
                            printf("bond %i %i \n", file->conn_from[file->nbonds-1],file->conn_to[file->nbonds-1]);
                            fflush(stdout);
                        #endif
                    }
                }
            }
            if (file->mfpxtype == 2){
                if (j > 5){
                    // this is a conn entry! we have to split by slash and remove the pconn info
                    pch_inner = strtok_r(pch,"/",&inner);
                    jidx = atoi(pch_inner);
                    // we have to prevent double counting here! 
                    // add a bond to the ctab only if the current atom idx is smaller than the bonded atom idx
                    if (i < jidx){
                        file->conn_from[file->nbonds]=i+1;
                        file->conn_to[file->nbonds]=jidx;
                        file->nbonds++;
                        #ifdef DEBUG
                            printf("bond %i %i \n", file->conn_from[file->nbonds-1],file->conn_to[file->nbonds-1]);
                            fflush(stdout);
                        #endif
                    }
                }
            }   
            //pch = strtok (NULL, " ");
            pch = strtok_r (NULL, " ",&outer);
            j = j+1; 
        }
    }
    file->conn_from =(int*) realloc(file->conn_from, (file->nbonds) * sizeof(int));
    file->conn_to =(int*) realloc(file->conn_to, (file->nbonds) * sizeof(int));

        //strncpy(atom->name,elems[i],16*sizeof(char));
        //strncpy(atom->name,elems[i],16*sizeof(char));
        //strncpy(atom->type,atypes[i],16*sizeof(char));
        //strncpy(atom->resname,fragtypes[i],8*sizeof(char));
        //atom->resid=fragnumbers[i];	
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
	return MOLFILE_SUCCESS;
}


static int read_mfpx_timestep(void *_file, int natoms, molfile_timestep_t *ts) {
	struct mfpx_file* file=_file;
    if (file->body_parsed != 0){
        return MOLFILE_ERROR;
    }
    // cell parameters if mfpx is periodic
    if (file->is_periodic != 0){
        ts->A=file->cellparams[0];
        ts->B=file->cellparams[1];
        ts->C=file->cellparams[2];
        ts->alpha=file->cellparams[3];
        ts->beta=file->cellparams[4];
        ts->gamma=file->cellparams[5];
    }
    // coordinates
    memcpy(&(ts->coords[0]), file->xyz, sizeof(float)*3*file->natoms);
    free(file->xyz);
    file->body_parsed=1;
    return MOLFILE_SUCCESS;
}

static int mfpx_get_bonds(void *_file, int *nbonds, int **from, int **to, float **bondorder, int **bondtype,  int *nbondtypes, char ***bondtypename){
	*bondorder = NULL;
	*bondtype=NULL;
	*nbondtypes=0;
	bondtypename=NULL;
	struct mfpx_file* file=_file; 

    *nbonds=file->nbonds;

    /*for (int i=0; i<file->nbonds; i++){
        printf('bond %i %i \n' % (file->conn_from[i],file->conn_to[i]));
    }*/

    if (file->nbonds == 0){
        #ifdef DEBUG
            printf("no bonds provided in the mfpx file");
        #endif
        return MOLFILE_ERROR;

    }else{
        *(from)=file->conn_from;
        *(to)=file->conn_to;
        return MOLFILE_SUCCESS;		
    }
}

/* VMD API REGISTRATION */
static molfile_plugin_t plugin;

VMDPLUGIN_API int VMDPLUGIN_init() {
	memset(&plugin, 0, sizeof(molfile_plugin_t));
	plugin.abiversion = vmdplugin_ABIVERSION;
	plugin.type = MOLFILE_PLUGIN_TYPE;
	plugin.name = "mfpx";
	plugin.prettyname = "MOFplus Structure File Format";
	plugin.author = "Julian Keupp";
	plugin.majorv = 1;
	plugin.minorv = 1;
	plugin.is_reentrant = VMDPLUGIN_THREADSAFE;
	plugin.filename_extension = "mfpx";
	plugin.open_file_read = open_mfpx_read; 
	plugin.read_structure = read_mfpx_structure;
	plugin.read_next_timestep = read_mfpx_timestep;
	plugin.read_bonds = mfpx_get_bonds;
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
