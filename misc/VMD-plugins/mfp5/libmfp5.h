
struct mfp5_file;

/*read operations*/

// opens the file, iff it exists and creates the internal structure and goes to the first timestep
int mfp5_open(struct mfp5_file** _file, const char *filename, int can_write);

// close the file and frees the internal structure
int mfp5_close(struct mfp5_file* file);

//mfp5 register
void mfp5_get_elems(struct mfp5_file* file, void** _data_out);

int mfp5_set_stage(struct mfp5_file* file, char *stagename);

int mfp5_get_box_information(struct mfp5_file* file, int time_i, float* cellparams_out);

int mfp5_get_timestep(struct mfp5_file* file, int time_i, float *coords);

int mfp5_seek_timestep(struct mfp5_file* file, int i);

int mfp5_get_conn(struct mfp5_file* file, int *nbonds, int **from, int **to);

void mfp5_get_fragtypes(struct mfp5_file* file, void** _data_out);

void mfp5_get_fragnumbers(struct mfp5_file* file, void** _data_out);

int initialize_mfp5_struct(struct mfp5_file* file);

void mfp5_get_atypes(struct mfp5_file* file, void** _data_out);

int mfp5_get_ntime(struct mfp5_file* file,int* ntime);

int mfp5_get_current_time(struct mfp5_file* file, int* current_time);

