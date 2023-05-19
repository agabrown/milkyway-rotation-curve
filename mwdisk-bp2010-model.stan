/*
 * Stan implementation of a simple Milky Way disk rotation model which is intended
 * to fit observed proper motions of a sample of disk stars.
 *
 * In this model the rotation curve is the one given by equation (7) in Brunetti & Pfenniger (2010).
 *
 *  v0: Velocity scale of the rotation curve (positive, km/s)
 *  h: Scale length of the potential that produces the rotation curve (pc)
 *  p: Shape parameter of rotation curve (dimensionless, must be in [-1,2])
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
 *  Rsun: Distance from sun to Galactic centre (pc)
 *  Ysun: Position of the sun in Cartesian galactocentric Y (0 pc, by definition)
 *  Zsun: Position of the sun in Cartesian galactocentric Z (pc)
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
        real Av, real Rsun, vector Vsun_pec, real Vcirc_sun, real v0, real hscale, real pshape) {
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
     *    distance of sun to Galactic centre (kpc)
     *  Vsun_pec: vector of size 3
     *    Galactocentric Cartesian peculiar velocity of the sun (km/s)
     *  Vcirc_sun : real
     *    Circular velocity at the position of the sun (km/s).
     *  v0: real
     *    Velocity scaling of the rotation curve (km/s, positive)
     *  hscale: real
     *    Scale length of potential (kpc, positive)
     *  pshape: real
     *    Shape parameter rotation curve (dimensionless, in [-1,2])
     *
     * Returns 
     *  array[N] vector[3] of predicted proper motions (mas/yr) and the value of the Galactocentric cylindrical coordinate Phi (radians).
     */
    array[size(plx)] vector[2] predicted_pm;

    //real Vcirc_sun = v0*Rsun/hscale*(1+(Rsun/hscale)^2)^((pshape-2)/4);
    vector[3] vsun = [0.0, Vcirc_sun, 0.0]' + Vsun_pec;
    vector[3] vdiff;
    real vphistar;

    for (i in 1:size(plx)) {
        vphistar = -(v0*Rstar[i]/hscale*(1+(Rstar[i]/hscale)^2)^((pshape-2)/4));
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
  real Rsun;              // Distance from Sun to Galactic centre (kpc), used as fixed parameter
  real Zsun;              // Sun's height above the Galactic plane (kpc), used as fixed parameter
}

transformed data {
  real Ysun = 0.0;    // Sun galactocentric Cartesian y-coordinate (0 by definition)
  
  // Parameters for priors
  real Vcirc_sun_prior_mean = 220.0;
  real Vcirc_sun_prior_sigma = 50.0;
  real hbp_prior_mean = 4.0;
  real hbp_prior_sigma = 1.0;
  real Vsun_pec_x_prior_mean = 11.0;
  real Vsun_pec_y_prior_mean = 12.0;
  real Vsun_pec_z_prior_mean = 7.0;
  real Vsun_pec_x_prior_sigma = 20.0;
  real Vsun_pec_y_prior_sigma = 20.0;
  real Vsun_pec_z_prior_sigma = 20.0;
  real vdisp_prior_low = 0.0;
  real vdisp_prior_max = 200.0;

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
  array[N] matrix[3,3] J;

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

    starpos = (1.0/plx_obs[n])*rvec[n] + sunpos;
    Rstar[n] = sqrt(starpos[1]^2+starpos[2]^2);
    phistar[n] = atan2(starpos[2], starpos[1]);

    J[n][1,3] = 0.0;
    J[n][2,3] = 0.0;
    J[n][3,1] = 0.0;
    J[n][3,2] = 0.0;
    J[n][3,3] = 1.0;
    J[n][1,1] = cos(phistar[n]);
    J[n][2,2] = cos(phistar[n]);
    J[n][1,2] = -sin(phistar[n]);
    J[n][2,1] = sin(phistar[n]);
  }
}

parameters {
  real<lower=0> Vcirc_sun; // Circular velocity at the position of the sun (km/s)
  real<lower=0> hbp;       // Scale length potential (kpc)
  real<lower=-1, upper=2> pbp;          // Rotation curve shape parameter;
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
  cov_matrix[3] scov_cyl; // Covariance matrix in cylindrical coordinates (diagonal)
  array[N] cov_matrix[2] dcov; // Total covariance matrix (observational uncertainties plus velocity dispersion)
  real v0;

  v0 = Vcirc_sun/(Rsun/hbp*(1+(Rsun/hbp)^2)^((pbp-2)/4));
  model_pm = predicted_proper_motions(plx_obs, Rstar, phistar, pvec, qvec, rvec, 
        auKmYearPerSec, Rsun, [Vsun_pec_x, Vsun_pec_y, Vsun_pec_z]', Vcirc_sun, v0, hbp, pbp);

  scov_cyl = diag_matrix([vdispR^2, vdispPhi^2, vdispZ^2]');;

  for (n in 1:N) {
    scov = quad_form_sym(scov_cyl, J[n]');
    dcov[n][1,1] = dot_product(pvec[n], scov*pvec[n]); 
    dcov[n][2,1] = dot_product(qvec[n], scov*pvec[n]); 
    dcov[n][1,2] = dot_product(pvec[n], scov*qvec[n]); 
    dcov[n][2,2] = dot_product(qvec[n], scov*qvec[n]);
    dcov[n] = cov_pm[n] + (plx_obs[n]/auKmYearPerSec)^2 * dcov[n];
  }

}

model {
  Vcirc_sun ~ normal(Vcirc_sun_prior_mean, Vcirc_sun_prior_sigma);
  hbp ~ normal(hbp_prior_mean, hbp_prior_sigma);
  pbp ~ uniform(-1,2);
  Vsun_pec_x ~ normal(Vsun_pec_x_prior_mean, Vsun_pec_x_prior_sigma);
  Vsun_pec_y ~ normal(Vsun_pec_y_prior_mean, Vsun_pec_y_prior_sigma);
  Vsun_pec_z ~ normal(Vsun_pec_z_prior_mean, Vsun_pec_z_prior_sigma);
  vdispR ~ uniform(vdisp_prior_low, vdisp_prior_max);
  vdispPhi ~ uniform(vdisp_prior_low, vdisp_prior_max);
  vdispZ ~ uniform(vdisp_prior_low, vdisp_prior_max);
  
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
