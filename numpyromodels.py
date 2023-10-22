"""
Implementations of the Milky Way disk rotation curve models in NumPyro.

Anthony Brown Sep 2023 - Sep 2023
"""

from jax import numpy as jnp
import numpyro
from numpyro import sample
from numpyro.distributions import (
    Uniform,
    Normal,
    MultivariateNormal,
)

from pygaia.astrometry.constants import au_km_year_per_sec


def predicted_proper_motions(plx, p, q, phi, vphi, vSun):
    """
    For the input value of the azimuthal velocity of the star (which follows from the rotation curve) and the velocity of the Sun, calculate the predicted proper motions in l and b. The parallax and the p and q vectors of the normal triad at (l,b) are used to calculate the proper motions.

    The assumption is thus that the star moves on a purely circular orbit with zero velocity in the R and z directions.

    Parameters
    ----------

    plx: array-like
        List of parallax values [mas] (shape (N,))
    p: array-like
        The component of the normal triad pointing along increasing longitude (shape (plx.size, 3))
    q: array-like
        The component of the normal triad pointing along increasing latitude (shape (plx.size, 3))
    phi: array-like
        Phi coordinate of star in Galactocentric cylindrical coordinate system [radians]
    vphi: array-like
        Azimuthal velocity V_phi [km/s]
    vSun: array-like
        The velocity of the sun in Galactocentric cylindrical coordinates, including the peculiar motion with respect to the circular motion at the sun's position (3-element array) [km/s]

    Returns
    -------
    predicted_pm : array-like
        Proper motions (mu_l*, mu_b) predicted from the input values (shape (plx.size, 2)) [mas/yr]
    """
    vdiff = (
        jnp.stack(
            [-vphi * jnp.sin(phi), vphi * jnp.cos(phi), jnp.zeros(plx.size)], axis=1
        )
        - vSun
    )
    predicted_pm = (
        jnp.stack([jnp.sum(vdiff * p, axis=1), jnp.sum(vdiff * q, axis=1)])
        * plx
        / au_km_year_per_sec
    ).T
    return predicted_pm


def galactocentric_star_position(plx, r, sunposvec):
    """
    Calculate the Galactocentric position of the stars with direction vector r and parallax plx.

    Parameters
    ----------

    plx: array-like
        List of parallax values [mas] (shape (N,))
    r: array-like
        The component of the normal triad pointing along the direction to the star (shape (plx.size, 3))
    sunposvec: array-like
        Position vector of the sun in Galactocentric coordinates (kpc, shape (3,))

    Returns
    -------
    Rstar : array-like
        Galactocentric cylindrical R-coordinate of the stars (kpc, shape (N,))
    phistar: array-like
        Galactocentric cylindrical phi-coordinate of the stars (kpc, shape(N,))
    """
    starpos = jnp.dot(jnp.diag(1.0 / plx), r) + sunposvec
    Rstar = jnp.sqrt(starpos[:, 0] ** 2 + starpos[:, 1] ** 2)
    phistar = jnp.arctan2(starpos[:, 1], starpos[:, 0])
    return Rstar, phistar


def propermotion_covariance_matrix(plx, p, q, vdispR, vdispPhi, vdispZ, phi, obscov):
    """
    Calculate the full covariance matrix of the proper motions as the sum of the observational uncertainty and velocity dispersion terms.

    Parameters
    ----------

    plx: array-like
        List of parallax values [mas] (shape (N,))
    p: array-like
        The component of the normal triad pointing along increasing longitude (shape (plx.size, 3))
    q: array-like
        The component of the normal triad pointing along increasing latitude (shape (plx.size, 3))
    vdispR : float
        Velocity dispersion in R (km/s)
    vdispPhi : float
        Velocity dispersion in phi (km/s)
    vdispZ : float
        Velocity dispersion in z (km/s)
    phi: array-like
        Galactocentric cylindrical phi-coordinate of the stars (radians, shape (N,))
    obscov : array-like
        Covariance matrix of the observed proper motions (shape (N,2,2))

    Returns
    -------
    dcov : array-like
        Covariance matrices (shape (N,2,2))
    """
    nstars = plx.size
    vdispRSqr = vdispR**2
    vdispPhiSqr = vdispPhi**2
    zeros = jnp.zeros(nstars)
    s01 = jnp.sin(phi) * jnp.cos(phi) * (vdispRSqr - vdispPhiSqr)
    scov = jnp.stack(
        [
            vdispRSqr * jnp.cos(phi) ** 2 + vdispPhiSqr * jnp.sin(phi) ** 2,
            s01,
            zeros,
            s01,
            vdispRSqr * jnp.sin(phi) ** 2 + vdispPhiSqr * jnp.cos(phi) ** 2,
            zeros,
            zeros,
            zeros,
            zeros + vdispZ**2,
        ]
    ).T.reshape(nstars, 3, 3)
    dcovtemp = (
        (
            jnp.stack(
                [
                    jnp.einsum("...j,...j", p, jnp.einsum("...jk,...k", scov, p)),
                    jnp.einsum("...j,...j", q, jnp.einsum("...jk,...k", scov, p)),
                    jnp.einsum("...j,...j", p, jnp.einsum("...jk,...k", scov, q)),
                    jnp.einsum("...j,...j", q, jnp.einsum("...jk,...k", scov, q)),
                ]
            )
        )
        * (plx / au_km_year_per_sec) ** 2
    ).T.reshape(nstars, 2, 2)
    return dcovtemp + obscov


def linear_vcirc_disptens_rphiz(
    nstars, p, q, r, plx_obs, Rsun, Zsun, cov_pm, pm_obs=None
):
    """
    NumPyro implementation of a simple Milky Way disk rotation model which is intended to fit observed proper motions of a sample of OBA stars.

    In this model the rotation curve changes with a constant gradient as a function of distance from the galactic centre. That is,it declines (or increases) linearly with distance. The free parameters are:

    Vcirc_sun: circular velocity at the location of the sun (positive value by convention, km/s)
    dVcirc_dR: gradient in circular velocity (km/s/kpc)
    Vsun_pec_x: peculiar motion of the sun in Cartesian galactocentric X (km/s)
    Vsun_pec_y: peculiar motion of the sun in Cartesian galactocentric Y (km/s)
    Vsun_pec_z: peculiar motion of the sun in Cartesian galactocentric Z (km/s)
    vdispR: Velocity dispersion of the stars around the circular motion in the R
               direction (cylindrical Galactocentric coordinates, km/s)
    vdispPhi: Velocity dispersion of the stars around the circular motion in the Phi
               direction (cylindrical Galactocentric coordinates, km/s)
    vdispZ: Velocity dispersion of the stars around the circular motion in the Z
               direction (cylindrical Galactocentric coordinates, km/s)

    Fixed parameters:

    Rsun: Distance from sun to Galactic centre (8277 pc, GRAVITY)
    Ysun: Position of the sun in Cartesian galactocentric Y (0 pc, by definition)
    Zsun: Position of the sun in Cartesian galactocentric Z (20.8 pc, Bennett & Bovy)

    Parallaxes uncertainties are ignored here.

    A right handed coordinate system is used in which (X,Y,Z)_sun = (-Rsun, Ysun, Zsun) and Vphi(sun) = -Vcirc(sun).
    """

    # Parameters for priors
    Vcirc_sun_prior_mean = 220.0
    Vcirc_sun_prior_sigma = 50.0
    dVcirc_dR_prior_mean = 0.0
    dVcirc_dR_prior_sigma = 10.0
    Vsun_pec_x_prior_mean = 11.0
    Vsun_pec_y_prior_mean = 12.0
    Vsun_pec_z_prior_mean = 7.0
    Vsun_pec_prior_sigma = 20.0
    vdisp_prior_max = 200.0

    # Priors
    Vcirc_sun = sample("Vcirc_sun", Normal(Vcirc_sun_prior_mean, Vcirc_sun_prior_sigma))
    dVcirc_dR = sample("dVcirc_dR", Normal(dVcirc_dR_prior_mean, dVcirc_dR_prior_sigma))
    Vsun_pec_x = sample(
        "Vsun_pec_x", Normal(Vsun_pec_x_prior_mean, Vsun_pec_prior_sigma)
    )
    Vsun_pec_y = sample(
        "Vsun_pec_y", Normal(Vsun_pec_y_prior_mean, Vsun_pec_prior_sigma)
    )
    Vsun_pec_z = sample(
        "Vsun_pec_z", Normal(Vsun_pec_z_prior_mean, Vsun_pec_prior_sigma)
    )
    vdispR = sample("vdispR", Uniform(low=0, high=vdisp_prior_max))
    vdispPhi = sample("vdispPhi", Uniform(low=0, high=vdisp_prior_max))
    vdispZ = sample("vdispZ", Uniform(low=0, high=vdisp_prior_max))

    # Calculate star position information
    Ysun = 0.0
    sunpos = jnp.array([-Rsun, Ysun, Zsun])
    Rstar, phistar = galactocentric_star_position(plx_obs, r, sunpos)

    # Rotation curve model
    vphistar = -(Vcirc_sun + dVcirc_dR * (Rstar - Rsun))

    # Predicted proper motions
    vSun = jnp.array([0, Vcirc_sun, 0]) + jnp.array(
        [Vsun_pec_x, Vsun_pec_y, Vsun_pec_z]
    )
    model_pm = predicted_proper_motions(plx_obs, p, q, phistar, vphistar, vSun)

    # Calculation of proper motions covariance matrix (uncertainties plus velocity dispersion)
    dcov = propermotion_covariance_matrix(
        plx_obs, p, q, vdispR, vdispPhi, vdispZ, phistar, cov_pm
    )

    # Likelihood
    with numpyro.plate("data", nstars):
        numpyro.sample("obs", MultivariateNormal(model_pm, dcov), obs=pm_obs)
