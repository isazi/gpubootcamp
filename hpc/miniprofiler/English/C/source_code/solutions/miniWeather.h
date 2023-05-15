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

void init()
{
  int i, k, ii, kk, ll, inds, i_end;
  double x, z, r, u, w, t, hr, ht, nper;

  nx_glob = 40;      //Number of total cells in the x-dirction
  nz_glob = 20;      //Number of total cells in the z-dirction
  sim_time = 1000;   //How many seconds to run the simulation
  output_freq = 100; //How frequently to output data to file (in seconds)

  //Set the cell grid size
  dx = xlen / nx_glob;
  dz = zlen / nz_glob;

  nranks = 1;
  myrank = 0;

  // For simpler version, replace i_beg = 0, nx = nx_glob, left_rank = 0, right_rank = 0;

  nper = ((double)nx_glob) / nranks;
  i_beg = round(nper * (myrank));
  i_end = round(nper * ((myrank) + 1)) - 1;
  nx = i_end - i_beg + 1;
  left_rank = myrank - 1;
  if (left_rank == -1)
    left_rank = nranks - 1;
  right_rank = myrank + 1;
  if (right_rank == nranks)
    right_rank = 0;

  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////
  // YOU DON'T NEED TO ALTER ANYTHING BELOW THIS POINT IN THE CODE
  ////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////

  k_beg = 0;
  nz = nz_glob;

  //Allocate the model data
  state = (double *)malloc((nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(double));
  state_tmp = (double *)malloc((nx + 2 * hs) * (nz + 2 * hs) * NUM_VARS * sizeof(double));
  flux = (double *)malloc((nx + 1) * (nz + 1) * NUM_VARS * sizeof(double));
  tend = (double *)malloc(nx * nz * NUM_VARS * sizeof(double));
  hy_dens_cell = (double *)malloc((nz + 2 * hs) * sizeof(double));
  hy_dens_theta_cell = (double *)malloc((nz + 2 * hs) * sizeof(double));
  hy_dens_int = (double *)malloc((nz + 1) * sizeof(double));
  hy_dens_theta_int = (double *)malloc((nz + 1) * sizeof(double));
  hy_pressure_int = (double *)malloc((nz + 1) * sizeof(double));

  //Define the maximum stable time step based on an assumed maximum wind speed
  dt = dmin(dx, dz) / max_speed * cfl;
  //Set initial elapsed model time and output_counter to zero
  etime = 0.;
  output_counter = 0.;

  // Display grid information

#ifndef kernel_tuner
  printf("nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob);
  printf("dx,dz: %lf %lf\n", dx, dz);
  printf("dt: %lf\n", dt);
#endif

  //////////////////////////////////////////////////////////////////////////
  // Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
  //////////////////////////////////////////////////////////////////////////
  for (k = 0; k < nz + 2 * hs; k++)
  {
    for (i = 0; i < nx + 2 * hs; i++)
    {
      //Initialize the state to zero
      for (ll = 0; ll < NUM_VARS; ll++)
      {
        inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
        state[inds] = 0.;
      }
      //Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
      for (kk = 0; kk < nqpoints; kk++)
      {
        for (ii = 0; ii < nqpoints; ii++)
        {
          //Compute the x,z location within the global domain based on cell and quadrature index
          x = (i_beg + i - hs + 0.5) * dx + (qpoints[ii] - 0.5) * dx;
          z = (k_beg + k - hs + 0.5) * dz + (qpoints[kk] - 0.5) * dz;

          //Set the fluid state based on the user's specification (default is injection in this example)
          injection(x, z, r, u, w, t, hr, ht);

          //Store into the fluid state array
          inds = ID_DENS * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
          state[inds] = state[inds] + r * qweights[ii] * qweights[kk];
          inds = ID_UMOM * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
          state[inds] = state[inds] + (r + hr) * u * qweights[ii] * qweights[kk];
          inds = ID_WMOM * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
          state[inds] = state[inds] + (r + hr) * w * qweights[ii] * qweights[kk];
          inds = ID_RHOT * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
          state[inds] = state[inds] + ((r + hr) * (t + ht) - hr * ht) * qweights[ii] * qweights[kk];
        }
      }
      for (ll = 0; ll < NUM_VARS; ll++)
      {
        inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + k * (nx + 2 * hs) + i;
        state_tmp[inds] = state[inds];
      }
    }
  }
  //Compute the hydrostatic background state over vertical cell averages
  for (k = 0; k < nz + 2 * hs; k++)
  {
    hy_dens_cell[k] = 0.;
    hy_dens_theta_cell[k] = 0.;
    for (kk = 0; kk < nqpoints; kk++)
    {
      z = (k_beg + k - hs + 0.5) * dz;

      //Set the fluid state based on the user's specification (default is injection in this example)
      injection(0., z, r, u, w, t, hr, ht);

      hy_dens_cell[k] = hy_dens_cell[k] + hr * qweights[kk];
      hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr * ht * qweights[kk];
    }
  }
  //Compute the hydrostatic background state at vertical cell interfaces
  for (k = 0; k < nz + 1; k++)
  {
    z = (k_beg + k) * dz;

    //Set the fluid state based on the user's specification (default is injection in this example)
    injection(0., z, r, u, w, t, hr, ht);

    hy_dens_int[k] = hr;
    hy_dens_theta_int[k] = hr * ht;
    hy_pressure_int[k] = C0 * pow((hr * ht), gamm);
  }
}

void finalize()
{
  free(state);
  free(state_tmp);
  free(flux);
  free(tend);
  free(hy_dens_cell);
  free(hy_dens_theta_cell);
  free(hy_dens_int);
  free(hy_dens_theta_int);
  free(hy_pressure_int);
}
