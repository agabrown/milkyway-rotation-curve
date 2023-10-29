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
    LogNormal,
    MultivariateNormal,
)

from abc import ABC, abstractmethod

from pygaia.astrometry.constants import au_km_year_per_sec

from diskkinematicmodels import (
    BrunettiPfennigerRotationCurve,
    DiskKinematicModel,
    SlopedRotationCurve,
)

_generic_param_labels = {
    "Vcirc_sun": r"$v_{\mathrm{circ},\odot}$",
    "Vsun_pec_x": r"$v_{x,\odot}(\mathrm{pec})$",
    "Vsun_pec_y": r"$v_{y,\odot}(\mathrm{pec})$",
    "Vsun_pec_z": r"$v_{z,\odot}(\mathrm{pec})$",
    "vdispPhi": r"$\sigma_{v,\phi}$",
    "vdispR": r"$\sigma_{v,R}$",
    "vdispZ": r"$\sigma_{v,z}$",
}


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


class RotationCurve(ABC):
    """
    Abstract class to be subclassed by classes implementing a specific rotation curve and the priors on its parameters. The rotation curve is assumed to be of the form f_theta(R; v_circsun, Rsun), where theta the model parameters and the calculation of v_circ depends only on R, Rsun, and v_circsun.
    """

    @abstractmethod
    def get_priors(self):
        """
        Return a list of numpyro sample primitives each of which corresponds to a rotation curve model parameter and its prior.

        Returns
        -------
        priors : list
            List of numpyro sample primitives
        """
        pass

    @abstractmethod
    def get_vcirc(self, theta, vcircsun, Rsun, R):
        """
        Return the circular velocity at R.

        Parameters
        ----------
        theta : list
            List of model parameters (in same order as they are returned by get_priors())
        vcirsun : float
            Circular velocity of the sun (km/s)
        Rsun : float
            Distance from the sun to the Galactic centre (kpc)
        R : array-like
            R-coordinates of the stars (kpc)

        Returns
        -------
        vcirc : array-like
            Array of circular velocity values (km/s)
        """

    @abstractmethod
    def get_spec_param_label_map(self):
        """
        Get the mapping from parameter names to plot labels for the parameters theta specific to this model.

        Returns
        -------
        map : dict
            Dictionary with key-value pairs representing the parameter name to label mapping.
        """

    def get_all_param_label_map(self):
        """
        Get the mapping from parameter names to plot labels for all model parameters, including the generic ones (solar velocities and velocity dispersions).

        Returns
        -------
        map : dict
            Dictionary with key-value pairs representing the parameter name to label mapping.
        """
        return {**_generic_param_labels, **self.get_spec_param_label_map()}

    @abstractmethod
    def get_oort_constants(self, theta, vcircsun, Rsun, R):
        """
        Calculate the Oort constants A and B at R for the input rotation curve parameters.

        Parameters
        ----------
        theta : list
            List of model parameters (in same order as they are returned by get_priors())
        vcirsun : float
            Circular velocity of the sun (km/s)
        Rsun : float
            Distance from the sun to the Galactic centre (kpc)
        R : array-like
            R-coordinates of the stars (kpc)

        Returns
        -------

        oortA, oortB  : tuple
            Arrays with values of Oort A and B parameters in km/s/kpc
        """

    @abstractmethod
    def get_disk_kinematic_model(self, vcircsun, Rsun, theta, sunpos, vsunpeculiar):
        """
        Return an instance of diskkinematicmodel.DiskKinematicModel corresponding to this rotation curve model.

        Parameters
        ----------
        vcircsun : float
            Sun's circular velocity in km/s
        Rsun : float
            Distance from the sun to the Galactic centre (kpc)
        theta : list
            List of rotation curve specific model parameters (in same order as they are returned by get_priors())
        sunpos : array
            Sun's position as a (3,) array (in kpc).
        vsunpeculiar : array
            Sun's peculiar motion as a (3,) array (in km/s)
        """


class ConstantSlopeRotationCurve(RotationCurve):
    """
    Implements the rotation curve model
        f_theta(R) = v_circ,sun + slope*(R-Rsun)
    """

    def __init__(self):
        """
        Class constructor/initializer.
        """
        self.dVcirc_dR_prior_mean = 0.0
        self.dVcirc_dR_prior_sigma = 10.0
        self.slopename = "dVcirc_dR"

    def get_priors(self):
        return [
            sample(
                self.slopename,
                Normal(self.dVcirc_dR_prior_mean, self.dVcirc_dR_prior_sigma),
            )
        ]

    def get_vcirc(self, theta, vcircsun, Rsun, R):
        return vcircsun + theta[0] * (R - Rsun)

    def get_spec_param_label_map(self):
        return {self.slopename: r"$dV_\mathrm{circ}/dV_R$"}

    def get_oort_constants(self, theta, vcircsun, Rsun, R):
        oortA = 0.5 * (self.get_vcirc(theta, vcircsun, Rsun, R) / R - theta[0])
        oortB = -0.5 * (self.get_vcirc(theta, vcircsun, Rsun, R) / R + theta[0])
        return oortA, oortB

    def get_disk_kinematic_model(self, vcircsun, Rsun, theta, sunpos, vsunpeculiar):
        return DiskKinematicModel(
            SlopedRotationCurve(vcircsun, Rsun, theta[0]), sunpos, vsunpeculiar
        )


class BP2010RotationCurve(RotationCurve):
    r"""
    Implements the rotation curve model from Brunetti & Pfenniger (2010).

    :math:`v_0\frac{R}{h}\left[1+\left(\frac{R}{h}\right)^2\right]^{\frac{p-2}{4}}`

    where

    :math:`v_0 = v_{\mathrm{circ},\odot}\left(\frac{R_\odot}{h}\left[1+\left(\frac{R_\odot}{h}\right)^2\right]^{\frac{p-2}{4}}\right)^{-1}`
    """

    def __init__(self):
        """
        Class constructor/initializer.
        """
        self.log_hbp_prior_mean = jnp.log(4.0)
        self.log_hbp_prior_sigma = 0.5 * (jnp.log(4.0 + 1.0) - jnp.log(4.0 - 1.0))
        self.pbp_prior_min = -1.0
        self.pbp_prior_max = 2.0
        self.h_name = "h"
        self.p_name = "p"

    def get_priors(self):
        return [
            sample(
                self.h_name,
                LogNormal(self.log_hbp_prior_mean, self.log_hbp_prior_sigma),
            ),
            sample(self.p_name, Uniform(self.pbp_prior_min, self.pbp_prior_max)),
        ]

    def get_vcirc(self, theta, vcircsun, Rsun, R):
        v0 = vcircsun / (
            Rsun
            / theta[0]
            * jnp.power(1 + jnp.power(Rsun / theta[0], 2), ((theta[1] - 2) / 4))
        )
        return (
            v0
            * R
            / theta[0]
            * jnp.power(1 + jnp.power(R / theta[0], 2), ((theta[1] - 2) / 4))
        )

    def get_spec_param_label_map(self):
        return {self.h_name: r"$h$", self.p_name: r"$p$"}

    def get_oort_constants(self, theta, vcircsun, Rsun, R):
        rhsqr = R * R / (theta[0] * theta[0])
        rhterm = 1.0 + rhsqr
        v0 = vcircsun / (
            Rsun
            / theta[0]
            * jnp.power(1 + jnp.power(Rsun / theta[0], 2), ((theta[1] - 2) / 4))
        )
        dvdr = (
            v0
            / theta[0]
            * jnp.power(rhterm, (theta[1] - 2) / 4)
            * (1.0 + (theta[1] - 2) / 2 * rhsqr * jnp.power(rhterm, -1))
        )
        oortA = 0.5 * (self.get_vcirc(theta, vcircsun, Rsun, R) / R - dvdr)
        oortB = -0.5 * (self.get_vcirc(theta, vcircsun, Rsun, R) / R + dvdr)
        return oortA, oortB

    def get_disk_kinematic_model(self, vcircsun, Rsun, theta, sunpos, vsunpeculiar):
        return DiskKinematicModel(
            BrunettiPfennigerRotationCurve(vcircsun, Rsun, theta[0], theta[1]),
            sunpos,
            vsunpeculiar,
        )


def rotcurve_bayesian_model(
    rotcurve, plx_obs, p, q, r, Rsun, Zsun, cov_pm, pm_obs=None
):
    """
    NumPyro implementation of a simple Milky Way disk rotation model which is intended to fit observed proper motions of a sample of stars located close to the disk plane.

    Vcirc_sun: circular velocity at the location of the sun (positive value by convention, km/s)
    theta: parameters of the rotation curve
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

    Rsun: Distance from sun to Galactic centre
    Ysun: Position of the sun in Cartesian galactocentric Y (0 pc, by definition)
    Zsun: Position of the sun in Cartesian galactocentric Z
    Position (l,b) (or equivalently, the normal triad [p,q,r]) and parallax of the stars: parallax uncertainties are ignored

    A right handed coordinate system is used in which (X,Y,Z)_sun = (-Rsun, Ysun, Zsun) and Vphi(sun) = -Vcirc(sun).

    Parameters
    ----------

    rotcurve : RotationCurve
        Instance of RotationCurve
    plx_obs : array-like
        List of parallax values [mas] (shape (N,))
    p, q, r : array-like
        Vector of the normal triad (shape (N,3))
    Rsun : float
        Distance from Sun to Galactic centre (kpc)
    Zsun : float
        Height of Sun above Galactic plane (kpc)
    cov_pm : array-like
        Covariance matrix of the observed proper motions in Galactic coordinates (shape (N,2,2))
    pm_obs : array-like
        Observed proper motions in Galactic coordinates (shape (N,2))

    Returns
    -------

    Nothing
    """

    # Parameters for priors
    log_Vcirc_sun_prior_mean = jnp.log(220.0)
    log_Vcirc_sun_prior_sigma = 0.5 * (jnp.log(220 + 50) - jnp.log(220 - 50))
    Vsun_pec_x_prior_mean = 11.0
    Vsun_pec_y_prior_mean = 12.0
    Vsun_pec_z_prior_mean = 7.0
    Vsun_pec_prior_sigma = 20.0
    log_vdisp_prior_mean = jnp.log(10)
    log_vdisp_prior_sigma = 0.5 * (jnp.log(10 + 5) - jnp.log(10 - 5))
    theta = rotcurve.get_priors()

    # Priors
    Vcirc_sun = sample(
        "Vcirc_sun", LogNormal(log_Vcirc_sun_prior_mean, log_Vcirc_sun_prior_sigma)
    )
    Vsun_pec_x = sample(
        "Vsun_pec_x", Normal(Vsun_pec_x_prior_mean, Vsun_pec_prior_sigma)
    )
    Vsun_pec_y = sample(
        "Vsun_pec_y", Normal(Vsun_pec_y_prior_mean, Vsun_pec_prior_sigma)
    )
    Vsun_pec_z = sample(
        "Vsun_pec_z", Normal(Vsun_pec_z_prior_mean, Vsun_pec_prior_sigma)
    )
    vdispR = sample("vdispR", LogNormal(log_vdisp_prior_mean, log_vdisp_prior_sigma))
    vdispPhi = sample(
        "vdispPhi", LogNormal(log_vdisp_prior_mean, log_vdisp_prior_sigma)
    )
    vdispZ = sample("vdispZ", LogNormal(log_vdisp_prior_mean, log_vdisp_prior_sigma))

    # Calculate star position information
    Ysun = 0.0
    sunpos = jnp.array([-Rsun, Ysun, Zsun])
    Rstar, phistar = galactocentric_star_position(plx_obs, r, sunpos)

    # Rotation curve model
    vphistar = -rotcurve.get_vcirc(theta, Vcirc_sun, Rsun, Rstar)

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
    with numpyro.plate("data", plx_obs.size):
        numpyro.sample("obs", MultivariateNormal(model_pm, dcov), obs=pm_obs)
