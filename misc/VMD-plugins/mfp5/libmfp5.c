#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "hdf5.h"
#include "hdf5_hl.h"
#include "libmfp5.h"

#include <libgen.h>	//for use of basename
#include <unistd.h>	//for use of getlogin_r
#include <math.h>	//for use of acos

#define TRUE	1
#define FALSE	0
#define PI 3.14159265

#define DEBUG 0

struct mfp5_file{
	hid_t file_id;
	hid_t current_stage;
	hid_t current_stage_xyz;
    char current_stage_name;
	int natoms;
    int ntime;
    int current_time;
	char *last_error_message;
};



//declaration of boring helper functions
char* concatenate_strings(const char* string1,const char* string2);
int max(int a, int b);
float calculate_length_of_vector(float* vector, int dimensions);
float calculate_angle_between_vectors(float* vector1, float* vector2, int dimensions);



int mfp5_open(struct mfp5_file** _file, const char *filename, int can_write){
    struct mfp5_file *file = malloc(sizeof(struct mfp5_file));
	file->file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	initialize_mfp5_struct(file);
    // set mfp5 natoms attribute from length of elems Dset (RS 2024)
    hid_t dset  = H5Dopen(file->file_id, "/system/elems",H5P_DEFAULT); 
    hid_t space = H5Dget_space (dset);
    int ndims = H5Sget_simple_extent_ndims (space);
    printf("ndims is %d \n\n", ndims);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(space, dims, NULL);
    file->natoms = dims[0];
    printf("%d Natoms detected. is that correct? \n\n", file->natoms);
    H5Dclose (dset);
    H5Sclose (space);
    // error if file could not be opened?!!?
	*_file = file;
	if(file->file_id <0){
		return -1;
	}else{
		return 0;
	}
}

void mfp5_detect_stages(struct mfp5_file* file){
    herr_t   status,err;
	hsize_t nobj;
    hid_t    gid;
    ssize_t  len;
    char     group_name[1024],memb_name[1024];
    char** group_names;

    gid = H5Gopen(file->file_id,"/",H5P_DEFAULT);
    len = H5Iget_name(gid, group_name, 1024  );
	err = H5Gget_num_objs(gid, &nobj);

	for (int i = 0; i < nobj; i++) {
                /*
                 *  For each object in the group, get the name and
                 *   what type of object it is.
                 */
		len = H5Gget_objname_by_idx(gid, (hsize_t)i,memb_name, (size_t)1024 );
        if (strcmp(memb_name,"default") == 0){
            continue;
        }
        if (strcmp(memb_name,"system") == 0){
            continue;
        }
		printf("   %s \n",memb_name);fflush(stdout);
    }
    H5Gclose(gid);
    return;
}

int mfp5_set_stage(struct mfp5_file* file, char *stagename){
    if (*stagename == file-> current_stage_name){
        return 0;
    }
    char* stage_charptr = concatenate_strings(stagename ,"/traj/xyz");	
    hid_t xyz_dset = H5Dopen(file->file_id, stage_charptr ,H5P_DEFAULT); //get the 
    hid_t stage_dset = H5Gopen(file->file_id,stagename, H5P_DEFAULT);
    file->current_stage_xyz = xyz_dset;
    file->current_stage = stage_dset;
    file->current_stage_name = *stagename;
    file->current_time = 0;
    hid_t dspace = H5Dget_space(xyz_dset);
    const int ndims = H5Sget_simple_extent_ndims(dspace);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    file->ntime = dims[0];
    free(stage_charptr);
    H5Dclose(xyz_dset);
	if(file->file_id <0){
		return -1;
	}else{
		return 0;
	}
}
// internally reads the box information of a given group into the memory
//int mfp5_get_box_information(struct mfp5_file* file, int time_i, float *cellparams_out[6]){
int mfp5_get_box_information(struct mfp5_file* file, int time_i, float* cellparams_out){
    htri_t exists;
    int status;
    //float *data_box = malloc(9*sizeof(float));
    float data_box[9]; 
    float aa,bb,cc,al,be,ga;
    //float cellparams_out[6];
    // if /stagename/traj/cell exists, we use it as box
    exists = H5Lexists(file->current_stage,"traj/cell",H5P_DEFAULT);
    //printf("stagehandle %d \n", file->current_stage); fflush(stdout);
    if (exists){
        // per timeframe cell information
        hid_t cell_dset = H5Dopen(file->current_stage,"traj/cell",H5P_DEFAULT);
        //hid_t cell_dset = H5Dopen(file->file_id,"/produ_pramp/traj/cell",H5P_DEFAULT);
        hid_t dspace = H5Dget_space(cell_dset);
        hsize_t start[3]; //= {file->current_time, 0, 0}
        start[0] = file->current_time;
        start[1] = 0;
        start[2] = 0;
        hsize_t count[3] = {1,3,3};
        //printf("ctime %d time_i %d\n", file->current_time, time_i); fflush(stdout);
        H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);
        // create memory dataspace
        int rank=1;
        hsize_t dimsmem[rank];
        dimsmem[0] = 9;
        hid_t memspace = H5Screate_simple(rank, dimsmem, NULL);  
        hsize_t offset_out[rank];
        offset_out[0] = 0;
        hsize_t count_out[rank];
        count_out[0] = 9;
        status = H5Sselect_hyperslab (memspace, H5S_SELECT_SET, offset_out, NULL, 
                                  count_out, NULL);
        // read
        H5Dread(cell_dset, H5T_IEEE_F32LE, memspace, dspace, H5P_DEFAULT, &data_box);
        //for (int i=0; i<9; i++){
        //    printf("%d  %f\n", i,data_box[i]);fflush(stdout);
        //}
        status = H5Dclose(cell_dset);
        //printf("status %d\n", status);fflush(stdout);
        status = H5Sclose(dspace);
        //printf("status %d\n", status);fflush(stdout);
    // if not, we use /stage/restart/cell as box
    }else{
        // constant cell information throughout stage
        //printf("getting cell information from restart dataset\n"); fflush(stdout);
        hid_t cell_dset = H5Dopen(file->current_stage,"restart/cell",H5P_DEFAULT);
        hid_t dspace = H5Dget_space(cell_dset);
        //printf("attempting read\n"); fflush(stdout);
        H5Dread(cell_dset, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data_box);
        status = H5Dclose(cell_dset);
        status = H5Sclose(dspace);
    }
    // data_box should in each case be the cell tensor
	float a[3];
	float b[3];
	float c[3];
    for (int i=0; i<3; i++){
        a[i] = data_box[i];
    }
        for (int i=0; i<3; i++){
        b[i] = data_box[i+3];
    }
        for (int i=0; i<3; i++){
        c[i] = data_box[i+6];
    }
    //printf("a =  %f %f %f\n", a[0], a[1], a[2]); fflush(stdout);
    //printf("b =  %f %f %f\n", b[0], b[1], b[2]); fflush(stdout);
    //printf("c =  %f %f %f\n", c[0], c[1], c[2]); fflush(stdout);
    
    aa = calculate_length_of_vector(a,3);
    //printf("aa = %f\n",&aa); fflush(stdout);
    cellparams_out[0] = aa;
    bb = calculate_length_of_vector(b,3);
    cellparams_out[1] = bb;
    //printf("bb = %f\n",bb); fflush(stdout);
    cc = calculate_length_of_vector(c,3);
    cellparams_out[2] = cc;
    //printf("cc = %f\n",cc); fflush(stdout);
    al = calculate_angle_between_vectors(b,c,3);
    cellparams_out[3] = al;
    //printf("al = %f\n",al); fflush(stdout);
    be = calculate_angle_between_vectors(a,c,3);
    cellparams_out[4] = be;
    //printf("be = %f\n",be); fflush(stdout);
    ga = calculate_angle_between_vectors(a,b,3);
    cellparams_out[5] = ga;
    //printf("ga = %f\n",ga); fflush(stdout);
    //free(data_box);
	return 0;
}

int mfp5_get_timestep(struct mfp5_file* file, int time_i, float *coords){
    int status;

    float xyz[file->natoms*3];

    hid_t dset = H5Dopen(file->current_stage,"traj/xyz",H5P_DEFAULT);
    hid_t dspace = H5Dget_space(dset);

    hsize_t start[3] = {file->current_time, 0, 0};
    //hsize_t stride[3]= {1,1,1};
    hsize_t count[3] = {1,file->natoms,3};
    H5Sselect_hyperslab(dspace, H5S_SELECT_SET, start, NULL, count, NULL);

    // memspace 
    int rank=1;
    hsize_t dimsmem[rank];
    dimsmem[0] = file->natoms * 3;
    hid_t memspace = H5Screate_simple(rank, dimsmem, NULL);   
    hsize_t offset_out[rank];
    offset_out[0] = 0;
    hsize_t count_out[rank];
    count_out[0] = file->natoms * 3;
    status = H5Sselect_hyperslab (memspace, H5S_SELECT_SET, offset_out, NULL, 
                                  count_out, NULL);

    H5Dread(dset, H5T_IEEE_F32LE, memspace, dspace, H5P_DEFAULT, xyz);
    /*for (int t=0;t<file->natoms;t++){
        for (int i=0; i<3; i++){
            //printf("%d %d %f\n", t, i, xyz[i+3*t]);fflush(stdout);
        }
    }*/
    status = H5Dclose(dset);
    status = H5Sclose(dspace);
    memcpy(&(coords[0]), xyz, sizeof(float)*3*file->natoms);
    return 0;
}

int mfp5_seek_timestep(struct mfp5_file* file, int i){
	int ntime;
	mfp5_get_ntime(file,&ntime);
    if(i>=0 && i<ntime){
		file->current_time=i;
		return 0;
	}else{
		file->current_time=i;
		return -1;
	}
}

int mfp5_get_conn(struct mfp5_file* file, int *nbonds, int **from, int **to){
    hid_t dset = H5Dopen(file->file_id, "/system/cnc_table" ,H5P_DEFAULT); //get the 
    hid_t dspace = H5Dget_space(dset);
	size_t size_datatype  = H5Tget_size(H5T_STD_I32LE);
    // get the number of bonds from the length of the dataset
    const int ndims = H5Sget_simple_extent_ndims(dspace);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace, dims, NULL);
    *nbonds = dims[0];
    // read the bonds
    hid_t wanted_memory_datatype = H5T_NATIVE_INT;
    //determine needed size
    int conn[dims[0]][dims[1]];
    int* cfrom=(int*) malloc(sizeof(size_datatype)*dims[0]);
    int* cto=(int*) malloc(sizeof(size_datatype)*dims[0]);

    int status=H5Dread(dset, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &conn);
    for (int i=0; i<dims[0]; i++){
        cfrom[i] = conn[i][0]+1;
        cto[i] = conn[i][1]+1;
    }
    *(from)=cfrom;
    *(to)=cto;
	return 0;
}

// close file, datasets and frees the internal structure
int mfp5_close(struct mfp5_file* file){
	if(file!=NULL){
		H5Fflush(file->file_id,H5F_SCOPE_GLOBAL);
		H5Fclose(file->file_id);
		free(file);	
		return 0;
    }
	else{
		return -1;
	}
}



int mfp5_get_ntime(struct mfp5_file* file,int* ntime){
	(*ntime)=file->ntime;
	return 0;
}

int mfp5_get_natoms(struct mfp5_file* file, int* natoms){
	*natoms=file->natoms;
	return 0;
}

//get current time
int mfp5_get_current_time(struct mfp5_file* file, int* current_time){
	(*current_time)=file->current_time;
	return 0;
}



void mfp5_get_elems(struct mfp5_file* file, void** _data_out){
    printf("Entering get_elems function\n"); fflush(stdout);
    
    hid_t       filetype, memtype, space;   /* Handles */                     
    herr_t      status;
    char        **rdata;                    /* Read buffer */
    int         ndims, i;

    char elems_path[] =  "system/elems"; // should always exist in file!
    char* elems_pathptr = &elems_path; // prepare pointer
    //hsize_t dims[1] = {file->natoms};
    hsize_t     dims[1] = {file->natoms};
    hid_t dset = H5Dopen(file->file_id, elems_pathptr,H5P_DEFAULT); //get the 
    // * Get the datatype.
    filetype = H5Dget_type (dset);
    // * Get dataspace and allocate memory for read buffer.
    space = H5Dget_space (dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);
    rdata = (char **) malloc (dims[0] * sizeof (char *));
    // * Create the memory datatype.
    memtype = H5Tcopy (H5T_C_S1);
    status = H5Tset_cset(memtype, H5T_CSET_UTF8); // this shitty line took me a couple of  hours ... 
    status = H5Tset_size (memtype, H5T_VARIABLE);
    // * Read the data.
    status = H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    #if defined DEBUG
    /*for (i=0; i<dims[0]; i++)
        printf ("%s[%d]: %s\n", elems_path, i, rdata[i]);
    printf("dims: %d x %d\n", dims[0],dims[1]); fflush(stdout);
    printf("charlen: %d %d \n", sizeof(char), sizeof(char*)); fflush(stdout); */
    #endif
    for (i=0; i<dims[0]; i++){
        rdata[i][0] = toupper(rdata[i][0]);
    }
    *(_data_out)=rdata;
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Tclose (filetype);
    status = H5Tclose (memtype);
}

void mfp5_get_atypes(struct mfp5_file* file, void** _data_out){
    printf("Entering get_elems function\n"); fflush(stdout);
    
    hid_t       filetype, memtype, space;   /* Handles */                        
    herr_t      status;
    char        **rdata;                    /* Read buffer */
    int         ndims, i;

    char atypes_path[] =  "system/atypes"; // should always exist in file!
    char* atypes_pathptr = &atypes_path; // prepare pointer
    //hsize_t dims[1] = {file->natoms};
    hsize_t     dims[1] = {file->natoms};
    hid_t dset = H5Dopen(file->file_id, atypes_pathptr,H5P_DEFAULT); //get the 
    // * Get the datatype.
    filetype = H5Dget_type (dset);
    // * Get dataspace and allocate memory for read buffer.
    space = H5Dget_space (dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);
    rdata = (char **) malloc (dims[0] * sizeof (char *));
    // * Create the memory datatype.
    memtype = H5Tcopy (H5T_C_S1);
    status = H5Tset_cset(memtype, H5T_CSET_UTF8); // this shitty line took me a couple of  hours ... 
    status = H5Tset_size (memtype, H5T_VARIABLE);
    // * Read the data.
    status = H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    #if defined DEBUG /*
    for (i=0; i<dims[0]; i++)
        printf ("%s[%d]: %s\n", atypes_path, i, rdata[i]);
    printf("dims: %d x %d\n", dims[0],dims[1]); fflush(stdout);
    printf("charlen: %d %d \n", sizeof(char), sizeof(char*)); fflush(stdout);*/
    #endif
    *(_data_out)=rdata;
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Tclose (filetype);
    status = H5Tclose (memtype);
}

void mfp5_get_fragtypes(struct mfp5_file* file, void** _data_out){
    printf("Entering get_elems function\n"); fflush(stdout);
    
    hid_t       filetype, memtype, space;   /* Handles */                     
    herr_t      status;
    char        **rdata;                    /* Read buffer */
    int         ndims, i;

    char fragnames_path[] =  "system/fragtypes"; // should always exist in file!
    char* fragnames_pathptr = &fragnames_path; // prepare pointer
    //hsize_t dims[1] = {file->natoms};
    hsize_t     dims[1] = {file->natoms};
    hid_t dset = H5Dopen(file->file_id, fragnames_pathptr,H5P_DEFAULT); //get the 
    // * Get the datatype.
    filetype = H5Dget_type (dset);
    // * Get dataspace and allocate memory for read buffer.
    space = H5Dget_space (dset);
    ndims = H5Sget_simple_extent_dims (space, dims, NULL);
    rdata = (char **) malloc (dims[0] * sizeof (char *));
    // * Create the memory datatype.
    memtype = H5Tcopy (H5T_C_S1);
    status = H5Tset_cset(memtype, H5T_CSET_UTF8); // this shitty line took me a couple of  hours ... 
    status = H5Tset_size (memtype, H5T_VARIABLE);
    // * Read the data.
    status = H5Dread (dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);
    #if defined DEBUG /*
    for (i=0; i<dims[0]; i++)
        printf ("%s[%d]: %s\n", fragnames_path, i, rdata[i]);
    printf("dims: %d x %d\n", dims[0],dims[1]); fflush(stdout);
    printf("charlen: %d %d \n", sizeof(char), sizeof(char*)); fflush(stdout);*/
    #endif
    *(_data_out)=rdata;
    status = H5Dclose (dset);
    status = H5Sclose (space);
    status = H5Tclose (filetype);
    status = H5Tclose (memtype);
}

void mfp5_get_fragnumbers(struct mfp5_file* file, void** _data_out){
    printf("Entering get_fragnumbers function\n"); fflush(stdout);
                    
    herr_t      status;
    int         ndims, i;

    char fragnumbers_path[] =  "system/fragnumbers"; // should always exist in file!
    char* fragnumbers_pathptr = &fragnumbers_path; // prepare pointer
    hsize_t     dims[1] = {file->natoms};
    hid_t dset = H5Dopen(file->file_id, fragnumbers_path, H5P_DEFAULT); //get the 
    // * Get dataspace and allocate memory for read buffer.
    hid_t datatype  = H5Dget_type(dset);
    size_t size_datatype  = H5Tget_size(datatype);
    hid_t space = H5Dget_space (dset);
    int rank_dataset = H5Sget_simple_extent_ndims (space);
    unsigned long long int dims_dataset[rank_dataset];
    H5Sget_simple_extent_dims(space, dims_dataset, NULL);
    // case code starts here
    int needed_size=dims_dataset[0];
    int len_dims_dataset=sizeof(dims_dataset)/sizeof(dims_dataset[0]);
    for(int i=1; i<len_dims_dataset; i++){
        needed_size*=dims_dataset[i];
    }
    int* rdata = (int*) malloc(sizeof(size_datatype)*needed_size);
    // * Read the data.
    printf("Entering read function\n"); fflush(stdout);
    status = H5Dread (dset, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, rdata);/*
    for (i=0; i<dims[0]; i++)
        printf ("%s[%d]: %d\n", fragnumbers_path, i, rdata[i]);*/
    fflush(stdout);
    *(_data_out)=rdata;
    status = H5Dclose (dset);
    status = H5Sclose (space);
}


int initialize_mfp5_struct(struct mfp5_file* file){
	//file->last_error_message="";
	file->natoms=0;
    file->ntime=0;
    file->current_time=0;
	return 0;
}

int max(int a, int b){
	if(a>b)
		return a;
	else
		return b;
}

char* concatenate_strings(const char* string1, const char* string2){
	char* concat_string=malloc(strlen(string1)+strlen(string2)+1);	//+1 for 0 termination
	concat_string=strcpy(concat_string, string1);
	concat_string=strcat(concat_string, string2);
	return concat_string;
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
