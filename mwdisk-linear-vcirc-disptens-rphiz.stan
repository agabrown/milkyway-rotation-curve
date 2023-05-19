/*
 * Stan implementation of a simple Milky Way disk rotation model which is intended
 * to fit observed proper motions of a sample of OBA stars.
 *
 * In this model the rotation curve has a constant gradient as a function of distance 
 * from the galactic centre. That is,it declines (or increases) linearly with distance.
 * The free parameters are:
 *
 *  Vcirc_sun: circular velocity at the location of the sun (positive value by convention, km/s)
 *  dVcirc_dVr: gradient in circular velocity (km/s/kpc)
 *  Vsun_pec_x: peculiar motion of the sun in Cartesian galactocentric X (km/s)
 *  Vsun_pec_y: peculiar motion of the sun in Cartesian galactocentric Y (km/s)
 *  Vsun_pec_z: peculiar motion of the sun in Cartesian galactocentric Z (km/s)
 *  vdispR: Velocity dispersion of the stars around the circular motion in the R 
 *          direction (cylindrical Galactocentric coordinates, km/s)
 *  vdispPhi: Velocity dispersion of the stars around the circular motion in the Phi 
 *          direction (cylindrical Galactocentric coordinates, km/s)
 *  vdispZ: Velocity dispersion of the stars around the circular motion in the Z 
 *          direction (cylindrical Galactocentric coordinates, km/s)
 *
 * Fixed parameters:
 *
 *  Rsun: Distance from sun to Galactic centre (8277 pc, GRAVITY)
 *  Ysun: Position of the sun in Cartesian galactocentric Y (0 pc, by definition)
 *  Zsun: Position of the sun in Cartesian galactocentric Z (20.8 pc, Bennett & Bovy)
 *  Parallaxes are assumed to be error free
 *
 * A right handed coordinate system is used in which (X,Y,Z)_sun = (-Rsun, Ysun, Zsun)
 * and Vphi(sun) = -Vcirc(sun).
 *
 * Anthony Brown May 2023 - May 2023
 * <brown@strw.leidenuniv.nl>
 */

functions{
  array[] vector predicted_proper_motions(vector plx, vector Rstar, vector phi, 
        array[] vector p, array[] vector q, array[] vector r,
        real Av, real Rsun, vector Vsun_pec, real Vcirc_sun, real dVcirc_dVr) {
    /*
     * Using a simple Milky Way disk kinematics model, predict observed proper motions. This version also returns the value of the Galactocentric cylindrical coordinate Phi for each star. This is needed for the velocity dispersion covariance matrix transformation.
     *
     * Parameters
     *  plx: vector of size N
     *   Observed parallax (mas)
     *  Rstar: vector of size N
     *   Value of galactocentric cylindrical coordinate R, inferred from parallax and sky position.
     *  phi: vector of size N
     *   Value of galactocentric cylindrical coordinate phi, inferred from parallax and sky position.
     *  p, q, r: arrays of size N of 3-vectors
     *    The normal triads corresponding to the (l,b) positions of the sources
     *  Av: real
     *    Value of the constant relating velocity and proper motion units (4.74... km*yr/s)
     *  Rsun: real
     *    distance of sun to Galactic centre (pc)
     *  Vsun_pec: vector of size 3
     *    Galactocentric Cartesian peculiar velocity of the sun (km/s)
     *  Vcirc_sun: real
     *    Circular velocity at the position of the sun (km/s, positive)
     *  dVcirc_dVr: real
     *    Gradient of the circular velocity (km/s/kpc, positive)
     *
     * Returns 
     *  array[N] vector[3] of predicted proper motions (mas/yr) and the value of the Galactocentric cylindrical coordinate Phi (radians).
     */
    array[size(plx)] vector[2] predicted_pm;

    vector[3] vsun = [0.0, Vcirc_sun, 0.0]' + Vsun_pec;
    vector[3] vdiff;
    real vphistar;

    for (i in 1:size(plx)) {
      vphistar = -(Vcirc_sun + dVcirc_dVr*(Rstar[i]-Rsun)/1000.0);
      vdiff = [-vphistar*sin(phi[i]), vphistar*cos(phi[i]), 0.0]' - vsun;
      predicted_pm[i][1] = dot_product(p[i], vdiff) * plx[i] / Av;
      predicted_pm[i][2] = dot_product(q[i], vdiff) * plx[i] / Av;
    }

    return predicted_pm;
  }
}

data {
  int<lower=0> N;
  vector[N] galon;
  vector[N] galat;
  vector[N] pml_obs;
  vector[N] pml_obs_unc;
  vector[N] pmb_obs;
  vector[N] pmb_obs_unc;
  vector[N] pml_pmb_corr;
  vector[N] plx_obs;
  real Rsun;              // Distance from Sun to Galactic centre (pc), used as fixed parameter
  real Zsun;              // Sun's height above the Galactic plane (pc), used as fixed parameter
}

transformed data {
  real Ysun = 0.0;    // Sun galactocentric Cartesian y-coordinate (0 by definition)
  
  // Parameters for priors
  real Vcirc_sun_prior_mean = 220.0;
  real Vcirc_sun_prior_sigma = 50.0;
  real dVcirc_dVr_prior_mean = 0.0;
  real dVcirc_dVr_prior_sigma = 10.0;
  real Vsun_pec_x_prior_mean = 11.0;
  real Vsun_pec_y_prior_mean = 12.0;
  real Vsun_pec_z_prior_mean = 7.0;
  real Vsun_pec_x_prior_sigma = 20.0;
  real Vsun_pec_y_prior_sigma = 20.0;
  real Vsun_pec_z_prior_sigma = 20.0;
  real vdisp_prior_alpha = 2.0;
  real vdisp_prior_beta = 0.1;

  real auInMeter = 149597870700.0;
  real julianYearSeconds = 365.25 * 86400.0;
  real auKmYearPerSec = auInMeter/(julianYearSeconds*1000.0);

  array[N] cov_matrix[2] cov_pm;  // Covariance matrix of the proper motions (auxiliary variable only)
  array[N] vector[2] pm_obs;      // Observed proper motions
  array[N] vector[3] pvec;
  array[N] vector[3] qvec;
  array[N] vector[3] rvec;
  vector[N] Rstar;
  vector[N] phistar;
  vector[3] starpos;
  vector[3] sunpos = [-Rsun, Ysun, Zsun]';

  for (n in 1:N) {
    cov_pm[n][1,1] = pml_obs_unc[n]^2;
    cov_pm[n][2,2] = pmb_obs_unc[n]^2;
    cov_pm[n][1,2] = pml_obs_unc[n]*pmb_obs_unc[n]*pml_pmb_corr[n];
    cov_pm[n][2,1] = cov_pm[n][1,2];

    pm_obs[n][1] = pml_obs[n];
    pm_obs[n][2] = pmb_obs[n];

    pvec[n][1] = -sin(galon[n]);
    pvec[n][2] = cos(galon[n]);
    pvec[n][3] = 0.0;
    
    qvec[n][1] = -sin(galat[n])*cos(galon[n]);
    qvec[n][2] = -sin(galat[n])*sin(galon[n]);
    qvec[n][3] = cos(galat[n]);
    
    rvec[n][1] = cos(galat[n])*cos(galon[n]);
    rvec[n][2] = cos(galat[n])*sin(galon[n]);
    rvec[n][3] = sin(galat[n]);

    starpos = (1000.0/plx_obs[n])*rvec[n] + sunpos;
    Rstar[n] = sqrt(starpos[1]^2+starpos[2]^2);
    phistar[n] = atan2(starpos[2], starpos[1]);
  }
}

parameters {
  real Vcirc_sun;          // Circular velocity at the sun's position
  real dVcirc_dVr;         // Gradient of the circular velocity
  real Vsun_pec_x;         // Peculiar velocity of Sun in Galactocentric Cartesian X
  real Vsun_pec_y;         // Peculiar velocity of Sun in Galactocentric Cartesian Y
  real Vsun_pec_z;         // Peculiar velocity of Sun in Galactocentric Cartesian Z
  real<lower=0> vdispR;    // Velocity dispersion around circular motion in R direction
  real<lower=0> vdispPhi;  // Velocity dispersion around circular motion in Phi direction
  real<lower=0> vdispZ;    // Velocity dispersion around circular motion in Z direction
}

transformed parameters {
  array[N] vector[2] model_pm;
  cov_matrix[3] scov;          // Model covariance matrix for velocity dispersions
  array[N] cov_matrix[2] dcov; // Total covariance matrix (observational uncertainties plus velocity dispersion)
  real vdispRSqr, vdispPhiSqr;

  model_pm = predicted_proper_motions(plx_obs, Rstar, phistar, pvec, qvec, rvec, 
        auKmYearPerSec, Rsun, [Vsun_pec_x, Vsun_pec_y, Vsun_pec_z]', Vcirc_sun, dVcirc_dVr);

  scov[1,1] = 1.0;
  scov[2,2] = 1.0;
  scov[3,3] = vdispZ^2;
  scov[1,2] = 0.0;
  scov[1,3] = 0.0;
  scov[2,3] = 0.0;
  scov[2,1] = scov[1,2];
  scov[3,1] = scov[1,3];
  scov[3,2] = scov[2,3];

  vdispRSqr = vdispR^2;
  vdispPhiSqr = vdispPhi^2;
  for (n in 1:N) {
    scov[1,1] = vdispRSqr*cos(phistar[n])^2 + vdispPhiSqr*sin(phistar[n])^2;
    scov[2,2] = vdispRSqr*sin(phistar[n])^2 + vdispPhiSqr*cos(phistar[n])^2;
    scov[1,2] = sin(phistar[n])*cos(phistar[n])*(vdispRSqr-vdispPhiSqr);
    scov[2,1] = scov[1,2];
    dcov[n][1,1] = dot_product(pvec[n], scov*pvec[n]); 
    dcov[n][2,1] = dot_product(qvec[n], scov*pvec[n]); 
    dcov[n][1,2] = dot_product(pvec[n], scov*qvec[n]); 
    dcov[n][2,2] = dot_product(qvec[n], scov*qvec[n]);
    dcov[n] = cov_pm[n] + (plx_obs[n]/auKmYearPerSec)^2 * dcov[n];
  }

}

model {
  Vcirc_sun ~ normal(Vcirc_sun_prior_mean, Vcirc_sun_prior_sigma);
  dVcirc_dVr ~ normal(dVcirc_dVr_prior_mean, dVcirc_dVr_prior_sigma);
  Vsun_pec_x ~ normal(Vsun_pec_x_prior_mean, Vsun_pec_x_prior_sigma);
  Vsun_pec_y ~ normal(Vsun_pec_y_prior_mean, Vsun_pec_y_prior_sigma);
  Vsun_pec_z ~ normal(Vsun_pec_z_prior_mean, Vsun_pec_z_prior_sigma);
  //vdispR ~ gamma(vdisp_prior_alpha, vdisp_prior_beta);
  //vdispPhi ~ gamma(vdisp_prior_alpha, vdisp_prior_beta);
  //vdispZ ~ gamma(vdisp_prior_alpha, vdisp_prior_beta);
  vdispR ~ uniform(0,200);
  vdispPhi ~ uniform(0,200);
  vdispZ ~ uniform(0,200);
  
  for (i in 1:N) {
    pm_obs[i] ~ multi_normal(model_pm[i], dcov[i]);
  }
}

generated quantities {
  vector[N] pred_pml;
  vector[N] pred_pmb;
  vector[2] pred_pm;
  for (i in 1:N) {
    pred_pm = multi_normal_rng(model_pm[i], dcov[i]);
    pred_pml[i] = pred_pm[1];
    pred_pmb[i] = pred_pm[2];
  }
}
