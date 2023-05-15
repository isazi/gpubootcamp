#include <math.h>

const double pi = 3.14159265358979323846264338327;   //Pi
const double grav = 9.8;                             //Gravitational acceleration (m / s^2)
const double cp = 1004.;                             //Specific heat of dry air at constant pressure
const double rd = 287.;                              //Dry air constant for equation of state (P=rho*rd*T)
const double p0 = 1.e5;                              //Standard pressure at the surface in Pascals
const double C0 = 27.5629410929725921310572974482;   //Constant to translate potential temperature into pressure (P=C0*(rho*theta)**gamma)
const double gamm = 1.40027894002789400278940027894; //gamma=cp/Rd , have to call this gamm because "gamma" is taken (I hate C so much)
//Define domain and stability-related constants
const double xlen = 2.e4;     //Length of the domain in the x-direction (meters)
const double zlen = 1.e4;     //Length of the domain in the z-direction (meters)
const double hv_beta = 0.25;  //How strong to diffuse the solution: hv_beta \in [0:1]
const double cfl = 1.50;      //"Courant, Friedrichs, Lewy" number (for numerical stability)
const double max_speed = 450; //Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const int hs = 2;             //"Halo" size: number of cells needed for a full "stencil" of information for reconstruction
const int sten_size = 4;      //Size of the stencil used for interpolation

//Parameters for indexing and flags
const int NUM_VARS = 4; //Number of fluid state variables
const int ID_DENS = 0;  //index for density ("rho")
const int ID_UMOM = 1;  //index for momentum in the x-direction ("rho * u")
const int ID_WMOM = 2;  //index for momentum in the z-direction ("rho * w")
const int ID_RHOT = 3;  //index for density * potential temperature ("rho * theta")
const int DIR_X = 1;    //Integer constant to express that this operation is in the x-direction
const int DIR_Z = 2;    //Integer constant to express that this operation is in the z-direction

const int nqpoints = 3;

double qpoints[] = {0.112701665379258311482073460022E0, 0.500000000000000000000000000000E0, 0.887298334620741688517926539980E0};
double qweights[] = {0.277777777777777777777777777779E0, 0.444444444444444444444444444444E0, 0.277777777777777777777777777779E0};

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are initialized but remain static over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double sim_time;            //total simulation time in seconds
double output_freq;         //frequency to perform output in seconds
double dt;                  //Model time step (seconds)
int nx, nz;                 //Number of local grid cells in the x- and z- dimensions
double dx, dz;              //Grid space length in x- and z-dimension (meters)
int nx_glob, nz_glob;       //Number of total grid cells in the x- and z- dimensions
int i_beg, k_beg;           //beginning index in the x- and z-directions
int nranks, myrank;         //my rank id
int left_rank, right_rank;  //Rank IDs that exist to my left and right in the global domain
double *hy_dens_cell;       //hydrostatic density (vert cell avgs).   Dimensions: (1-hs:nz+hs)
double *hy_dens_theta_cell; //hydrostatic rho*t (vert cell avgs).     Dimensions: (1-hs:nz+hs)
double *hy_dens_int;        //hydrostatic density (vert cell interf). Dimensions: (1:nz+1)
double *hy_dens_theta_int;  //hydrostatic rho*t (vert cell interf).   Dimensions: (1:nz+1)
double *hy_pressure_int;    //hydrostatic press (vert cell interf).   Dimensions: (1:nz+1)

///////////////////////////////////////////////////////////////////////////////////////
// Variables that are dynamics over the course of the simulation
///////////////////////////////////////////////////////////////////////////////////////
double etime;          //Elapsed model time
double output_counter; //Helps determine when it's time to do output
//Runtime variable arrays
double *state;     //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
double *state_tmp; //Fluid state.             Dimensions: (1-hs:nx+hs,1-hs:nz+hs,NUM_VARS)
double *flux;      //Cell interface fluxes.   Dimensions: (nx+1,nz+1,NUM_VARS)
double *tend;      //Fluid state tendencies.  Dimensions: (nx,nz,NUM_VARS)
int num_out = 0;   //The number of outputs performed so far
int direction_switch = 1;

//Declaring the functions defined after "main"
void init();
void finalize();
void injection(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht);
void hydro_const_theta(double z, double &r, double &t);
void output(double *state, double etime);
void ncwrap(int ierr, int line);
void perform_timestep(double *state, double *state_tmp, double *flux, double *tend, double dt);
void semi_discrete_step(double *state_init, double *state_forcing, double *state_out, double dt, int dir, double *flux, double *tend);
void compute_tendencies_x(double *state, double *flux, double *tend);
void compute_tendencies_z(double *state, double *flux, double *tend);
void set_halo_values_x(double *state);
void set_halo_values_z(double *state);

//This test case is initially balanced but injects fast, cold air from the left boundary near the model top
//x and z are input coordinates at which to sample
//r,u,w,t are output density, u-wind, w-wind, and potential temperature at that location
//hr and ht are output background hydrostatic density and potential temperature at that location
void injection(double x, double z, double &r, double &u, double &w, double &t, double &hr, double &ht)
{
  hydro_const_theta(z, hr, ht);
  r = 0.;
  t = 0.;
  u = 0.;
  w = 0.;
}

//Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
//z is the input coordinate
//r and t are the output background hydrostatic density and potential temperature
void hydro_const_theta(double z, double &r, double &t)
{
  const double theta0 = 300.; //Background potential temperature
  const double exner0 = 1.;   //Surface-level Exner pressure
  double p, exner, rt;
  //Establish hydrostatic balance first using Exner pressure
  t = theta0;                                //Potential Temperature at z
  exner = exner0 - grav * z / (cp * theta0); //Exner pressure at z
  p = p0 * pow(exner, (cp / rd));            //Pressure at z
  rt = pow((p / C0), (1. / gamm));           //rho*theta at z
  r = rt / t;                                //Density at z
}

//How is this not in the standard?!
double dmin(double a, double b)
{
  if (a < b)
  {
    return a;
  }
  else
  {
    return b;
  }
};