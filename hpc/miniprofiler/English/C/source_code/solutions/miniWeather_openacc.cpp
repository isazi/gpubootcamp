//////////////////////////////////////////////////////////////////////////////////////////
// miniWeather
// Author: Matt Norman <normanmr@ornl.gov>  , Oak Ridge National Laboratory
// This code simulates dry, stratified, compressible, non-hydrostatic fluid flows
// For documentation, please see the attached documentation in the "documentation" folder
//////////////////////////////////////////////////////////////////////////////////////////

/*
** Copyright (c) 2018, National Center for Computational Sciences, Oak Ridge National Laboratory.  All rights reserved.
**
** Portions Copyright (c) 2020, NVIDIA Corporation.  All rights reserved.
*/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <nvtx3/nvToolsExt.h>
#include "miniWeather.h"


///////////////////////////////////////////////////////////////////////////////////////
// THE MAIN PROGRAM STARTS HERE
///////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
   if (argc == 4)
  {
    printf("The arguments supplied are %s %s %s\n", argv[1], argv[2], argv[3]);
    nx_glob = atoi(argv[1]);
    nz_glob = atoi(argv[2]);
    sim_time = atoi(argv[3]);
  }
  else
  {
    printf("Using default values ...\n");
  }

  nvtxRangePushA("Total");
  init();

#pragma acc data copyin(state_tmp[(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS], hy_dens_cell[nz + 2 * hs], hy_dens_theta_cell[nz + 2 * hs], hy_dens_int[nz + 1], hy_dens_theta_int[nz + 1], hy_pressure_int[nz + 1]) \
    create(flux[(nz + 1) * (nx + 1) * NUM_VARS], tend[nz * nx * NUM_VARS])                                                                                                                                        \
        copy(state [0:(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS])
  {
    //Output the initial state
    //output(state, etime);

    ////////////////////////////////////////////////////
    // MAIN TIME STEP LOOP
    ////////////////////////////////////////////////////

    nvtxRangePushA("while");
    while (etime < sim_time)
    {
      //If the time step leads to exceeding the simulation time, shorten it for the last step
      if (etime + dt > sim_time)
      {
        dt = sim_time - etime;
      }

      //Perform a single time step
      nvtxRangePushA("perform_timestep");
      perform_timestep(state, state_tmp, flux, tend, dt);
      nvtxRangePop();

      //Inform the user

      printf("Elapsed Time: %lf / %lf\n", etime, sim_time);

      //Update the elapsed time and output counter
      etime = etime + dt;
      output_counter = output_counter + dt;
      //If it's time for output, reset the counter, and do output

      if (output_counter >= output_freq)
      {
        output_counter = output_counter - output_freq;
#pragma acc update host(state[(nz + 2 * hs) * (nx + 2 * hs) * NUM_VARS])
        //output(state, etime);
      }
    }
    nvtxRangePop();
  }
  finalize();
  nvtxRangePop();
}

//Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
//The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
//order of directions is alternated each time step.
//The Runge-Kutta method used here is defined as follows:
// q*     = q[n] + dt/3 * rhs(q[n])
// q**    = q[n] + dt/2 * rhs(q*  )
// q[n+1] = q[n] + dt/1 * rhs(q** )
void perform_timestep(double *state, double *state_tmp, double *flux, double *tend, double dt)
{
  if (direction_switch)
  {
    //x-direction first
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_X, flux, tend);
    //z-direction second
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_Z, flux, tend);
  }
  else
  {
    //z-direction second
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_Z, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_Z, flux, tend);
    //x-direction first
    semi_discrete_step(state, state, state_tmp, dt / 3, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state_tmp, dt / 2, DIR_X, flux, tend);
    semi_discrete_step(state, state_tmp, state, dt / 1, DIR_X, flux, tend);
  }
  if (direction_switch)
  {
    direction_switch = 0;
  }
  else
  {
    direction_switch = 1;
  }
}

//Perform a single semi-discretized step in time with the form:
//state_out = state_init + dt * rhs(state_forcing)
//Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
void semi_discrete_step(double *state_init, double *state_forcing, double *state_out, double dt, int dir, double *flux, double *tend)
{
  if (dir == DIR_X)
  {
    //Set the halo values  in the x-direction
    set_halo_values_x(state_forcing);
    //Compute the time tendencies for the fluid state in the x-direction
    compute_tendencies_x(state_forcing, flux, tend);
  }
  else if (dir == DIR_Z)
  {
    //Set the halo values  in the z-direction
    set_halo_values_z(state_forcing);
    //Compute the time tendencies for the fluid state in the z-direction
    compute_tendencies_z(state_forcing, flux, tend);
  }

/////////////////////////////////////////////////
// TODO: THREAD ME
/////////////////////////////////////////////////
//Apply the tendencies to the fluid state
#pragma tuner start semi_discrete_step
#ifdef kernel_tuner
  double * state_init = state;
  double * state_out = state_tmp;
#endif
#pragma acc parallel default(present)
#pragma acc loop collapse(3)
  for (int ll = 0; ll < NUM_VARS; ll++)
  {
    for (int k = 0; k < nz; k++)
    {
      for (int i = 0; i < nx; i++)
      {
        int inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
        int indt = ll * nz * nx + k * nx + i;
        state_out[inds] = state_init[inds] + dt * tend[indt];
      }
    }
  }
#pragma tuner stop
}

//Compute the time tendencies of the fluid state using forcing in the x-direction

//First, compute the flux vector at each cell interface in the x-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_x(double *state, double *flux, double *tend)
{
  int i, k, ll, s, inds, indf1, indf2, indt;
  double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dx / (16 * dt);
  /////////////////////////////////////////////////
  // TODO: THREAD ME
  /////////////////////////////////////////////////
  //Compute fluxes in the x-direction for each cell
#pragma tuner start compute_tendencies_x_0
#pragma acc parallel private(ll, s, inds, stencil, vals, d3_vals, r, u, w, t, p) default(present)
#pragma acc loop collapse(2)
  for (k = 0; k < nz; k++)
  {
    for (i = 0; i < nx + 1; i++)
    {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll = 0; ll < NUM_VARS; ll++)
      {
        for (s = 0; s < sten_size; s++)
        {
          inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + s;
          stencil[s] = state[inds];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
        //First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
        d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_cell[k + hs];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = (vals[ID_RHOT] + hy_dens_theta_cell[k + hs]) / r;
      p = C0 * pow((r * t), gamm);

      //Compute the flux vector
      flux[ID_DENS * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u - hv_coef * d3_vals[ID_DENS];
      flux[ID_UMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u * u + p - hv_coef * d3_vals[ID_UMOM];
      flux[ID_WMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u * w - hv_coef * d3_vals[ID_WMOM];
      flux[ID_RHOT * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * u * t - hv_coef * d3_vals[ID_RHOT];
    }
  }
#pragma tuner stop

/////////////////////////////////////////////////
// TODO: THREAD ME
/////////////////////////////////////////////////
//Use the fluxes to compute tendencies for each cell
#pragma tuner start compute_tendencies_x_1
#pragma acc parallel private(indt, indf1, indf2) default(present)
#pragma acc  loop collapse(3)
  for (ll = 0; ll < NUM_VARS; ll++)
  {
    for (k = 0; k < nz; k++)
    {
      for (i = 0; i < nx; i++)
      {
        indt = ll * nz * nx + k * nx + i;
        indf1 = ll * (nz + 1) * (nx + 1) + k * (nx + 1) + i;
        indf2 = ll * (nz + 1) * (nx + 1) + k * (nx + 1) + i + 1;
        tend[indt] = -(flux[indf2] - flux[indf1]) / dx;
      }
    }
  }
#pragma tuner stop
}

//Compute the time tendencies of the fluid state using forcing in the z-direction

//First, compute the flux vector at each cell interface in the z-direction (including hyperviscosity)
//Then, compute the tendencies using those fluxes
void compute_tendencies_z(double *state, double *flux, double *tend)
{
  int i, k, ll, s, inds, indf1, indf2, indt;
  double r, u, w, t, p, stencil[4], d3_vals[NUM_VARS], vals[NUM_VARS], hv_coef;
  //Compute the hyperviscosity coeficient
  hv_coef = -hv_beta * dx / (16 * dt);
/////////////////////////////////////////////////
// TODO: THREAD ME
/////////////////////////////////////////////////
//Compute fluxes in the x-direction for each cell
#pragma tuner start compute_tendencies_z_0
#pragma acc parallel private(ll, s, inds, stencil, vals, d3_vals, r, u, w, t, p) default(present)
#pragma acc loop collapse(2)
  for (k = 0; k < nz + 1; k++)
  {
    for (i = 0; i < nx; i++)
    {
      //Use fourth-order interpolation from four cell averages to compute the value at the interface in question
      for (ll = 0; ll < NUM_VARS; ll++)
      {
        for (s = 0; s < sten_size; s++)
        {
          inds = ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + s) * (nx + 2 * hs) + i + hs;
          stencil[s] = state[inds];
        }
        //Fourth-order-accurate interpolation of the state
        vals[ll] = -stencil[0] / 12 + 7 * stencil[1] / 12 + 7 * stencil[2] / 12 - stencil[3] / 12;
        //First-order-accurate interpolation of the third spatial derivative of the state
        d3_vals[ll] = -stencil[0] + 3 * stencil[1] - 3 * stencil[2] + stencil[3];
      }

      //Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
      r = vals[ID_DENS] + hy_dens_int[k];
      u = vals[ID_UMOM] / r;
      w = vals[ID_WMOM] / r;
      t = (vals[ID_RHOT] + hy_dens_theta_int[k]) / r;
      p = C0 * pow((r * t), gamm) - hy_pressure_int[k];

      //Compute the flux vector with hyperviscosity
      flux[ID_DENS * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w - hv_coef * d3_vals[ID_DENS];
      flux[ID_UMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w * u - hv_coef * d3_vals[ID_UMOM];
      flux[ID_WMOM * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w * w + p - hv_coef * d3_vals[ID_WMOM];
      flux[ID_RHOT * (nz + 1) * (nx + 1) + k * (nx + 1) + i] = r * w * t - hv_coef * d3_vals[ID_RHOT];
    }
  }
#pragma tuner stop

/////////////////////////////////////////////////
// TODO: THREAD ME
/////////////////////////////////////////////////
//Use the fluxes to compute tendencies for each cell
#pragma tuner start compute_tendencies_z_1
#pragma acc parallel private(indt, indf1, indf2) default(present)
#pragma acc loop collapse(3)
  for (ll = 0; ll < NUM_VARS; ll++)
  {
    for (k = 0; k < nz; k++)
    {
      for (i = 0; i < nx; i++)
      {
        indt = ll * nz * nx + k * nx + i;
        indf1 = ll * (nz + 1) * (nx + 1) + (k) * (nx + 1) + i;
        indf2 = ll * (nz + 1) * (nx + 1) + (k + 1) * (nx + 1) + i;
        tend[indt] = -(flux[indf2] - flux[indf1]) / dz;
        if (ll == ID_WMOM)
        {
          inds = ID_DENS * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i + hs;
          tend[indt] = tend[indt] - state[inds] * grav;
        }
      }
    }
  }
#pragma tuner stop
}

void set_halo_values_x(double *state)
{
  int k, ll, ind_r, ind_u, ind_t, i;
  double z;

#pragma tuner start set_halo_values_x
#pragma acc parallel default(present)
#pragma acc loop collapse(2)
  for (ll = 0; ll < NUM_VARS; ll++)
  {
    for (k = 0; k < nz; k++)
    {
      state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + 0] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + nx + hs - 2];
      state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + 1] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + nx + hs - 1];
      state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + nx + hs] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + hs];
      state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + nx + hs + 1] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + hs + 1];
    }
  }
#pragma tuner stop
  ////////////////////////////////////////////////////

  if (myrank == 0)
  {
    for (k = 0; k < nz; k++)
    {
      for (i = 0; i < hs; i++)
      {
        z = (k_beg + k + 0.5) * dz;
        if (abs(z - 3 * zlen / 4) <= zlen / 16)
        {
          ind_r = ID_DENS * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i;
          ind_u = ID_UMOM * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i;
          ind_t = ID_RHOT * (nz + 2 * hs) * (nx + 2 * hs) + (k + hs) * (nx + 2 * hs) + i;
          state[ind_u] = (state[ind_r] + hy_dens_cell[k + hs]) * 50.;
          state[ind_t] = (state[ind_r] + hy_dens_cell[k + hs]) * 298. - hy_dens_theta_cell[k + hs];
        }
      }
    }
  }
}

//Set this task's halo values in the z-direction.
//decomposition in the vertical direction.
void set_halo_values_z(double *state)
{
  int i, ll;
  const double mnt_width = xlen / 8;
  double x, xloc, mnt_deriv;
/////////////////////////////////////////////////
// TODO: THREAD ME
/////////////////////////////////////////////////
#pragma tuner start set_halo_values_z
#pragma acc parallel private(x, xloc, mnt_deriv) default(present)
#pragma acc  loop
  for (ll = 0; ll < NUM_VARS; ll++)
  {
    for (i = 0; i < nx + 2 * hs; i++)
    {
      if (ll == ID_WMOM)
      {
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (0) * (nx + 2 * hs) + i] = 0.;
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (1) * (nx + 2 * hs) + i] = 0.;
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (nz + hs) * (nx + 2 * hs) + i] = 0.;
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (nz + hs + 1) * (nx + 2 * hs) + i] = 0.;
      }
      else
      {
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (0) * (nx + 2 * hs) + i] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (hs) * (nx + 2 * hs) + i];
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (1) * (nx + 2 * hs) + i] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (hs) * (nx + 2 * hs) + i];
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (nz + hs) * (nx + 2 * hs) + i] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (nz + hs - 1) * (nx + 2 * hs) + i];
        state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (nz + hs + 1) * (nx + 2 * hs) + i] = state[ll * (nz + 2 * hs) * (nx + 2 * hs) + (nz + hs - 1) * (nx + 2 * hs) + i];
      }
    }
  }
#pragma tuner stop
}
