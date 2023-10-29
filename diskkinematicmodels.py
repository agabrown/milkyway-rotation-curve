"""
Classes and functions that implement a very simple kinematic model for the Milky Way disk. This assumes stars describe strictly circular orbits around the vertical axis of the Milky Way disk plane. Thus in galactocentric cylindrical coordinates the velocity vectors of the stars are (V_R, V_phi, V_z) = (0, V_phi(R), 0), where V_phi(R) is the rotation curve of the disk. The velocity field does not change with z.

Anthony Brown Feb 2022 - May 2023
"""

import numpy as np
from abc import ABC, abstractmethod

import astropy.units as u
import astropy.constants as c
from astropy.coordinates import cartesian_to_spherical

from pygaia.astrometry.vectorastrometry import normal_triad

_au_km_year_per_sec = (c.au / (1 * u.yr).to(u.s)).to(u.km / u.s).value


class RotationCurve(ABC):
    """
    Abstract class to be sub-classed by classes implementing a Milky Way disk rotation curve. These classes are expected to implement a method that returns the value of the circular velocity as a function the 3D (Cartesian) position of the star.

    .. note::
        This class is meat to represent a very simple kinematic model of the Milky Way disk, valid strictly speaking only in the mid-plane of the disk. No proper potential with its rotation curve is implied.
    """

    @abstractmethod
    def circular_velocity(self, q):
        """
        Provide the circular velocity at the input position.

        Parameters
        ----------
        q : array_like shape (3,N), Quantity
            3D Cartesian position(s) in Galactic coordinates. Units must be included.

        Returns
        -------
        vcirc : array_like, Quantity
            Circular velocity at the given position(s) (km/s).
        """
        pass

    @abstractmethod
    def oort_ab(self, q):
        """
        Calculate the Oort A and B constants at position q.

        Parameters
        ----------
        q : array_like shape (3,N), Quantity
            3D Cartesian position(s) in Galactic coordinates. Units must be included.

        Returns
        -------
        oortA, oortB : Quantity
            Oort A and B constants (km/s/kpc).
        """
        pass

    def getinfo(self):
        """
        Returns
        -------
        info : str
            String with information about the rotation curve.
        """
        return self.addinfo()

    @abstractmethod
    def addinfo(self):
        """
        Returns
        -------
        info : str
            String with specific information about the rotation curve.
        """
        pass


class FlatRotationCurve(RotationCurve):
    """
    Implements a very simple kinematic model of the disk in which the circular velocity is constant everywhere.
    """

    def __init__(self, vcirc):
        """
        Class constructor/initializer.

        Parameters
        ----------

        vcirc: float
            Circular velocity in km/s.
        """
        self.vcirc = vcirc * u.km / u.s

    def circular_velocity(self, q):
        return np.tile(self.vcirc, q.shape[1:])

    def oort_ab(self, q):
        rq = np.sqrt(q[0] ** 2 + q[1] ** 2).to(u.kpc)
        oortA = 0.5 * self.circular_velocity(q) / rq
        oortB = -oortA
        return oortA, oortB

    def addinfo(self):
        return f"Flat rotation curve\n vcirc={self.vcirc}"


class SlopedRotationCurve(RotationCurve):
    """
    Implements a very simple kinematic model of the disk in which the circular velocity is given by a linear relation Vc(R) = Vc(Rsun) + slope*(R-Rsun).
    """

    def __init__(self, vcircsun, rsun, slope):
        """
        Class constructor/initializer.

        Parameters
        ----------

        vcircsun: float
            Circular velocity at the position of the sun in km/s.
        rsun: float
            Galactocentric distance of the Sun in kpc.
        slope: float
            Value of dVc(R)/dR in km/s/kpc
        """
        self.vcircsun = vcircsun * u.km / u.s
        self.rsun = rsun * u.kpc
        self.slope = slope * u.km / u.s / u.kpc

    def circular_velocity(self, q):
        rq = np.sqrt(q[0] ** 2 + q[1] ** 2).to(u.kpc)
        return self.vcircsun + ((rq - self.rsun) * self.slope).to(u.km / u.s)

    def oort_ab(self, q):
        rq = np.sqrt(q[0] ** 2 + q[1] ** 2).to(u.kpc)
        oortA = 0.5 * (self.circular_velocity(q) / rq - self.slope)
        oortB = 0.5 * (-self.circular_velocity(q) / rq - self.slope)
        return oortA, oortB

    def addinfo(self):
        return f"Sloped rotation curve\n vcircsun={self.vcircsun}, slope={self.slope}, Rsun={self.rsun}"


class SolidBodyRotationCurve(RotationCurve):
    """
    Implements a very simple kinematic model of the disk in which the circular velocity follows a solid body rotation curve. That is, the angular velocity at all radii is the same and V_phi(R) increases linearly with R.
    """

    def __init__(self, vcircsun, rsun):
        """
        Class constructor/initializer.

        Parameters
        ----------

        vcircsun: float
            Circular velocity at the position of the Sun in km/s.
        rsun: float
            Galactocentric distance of the Sun in kpc.
        """
        self.vcircsun = vcircsun * u.km / u.s
        self.rsun = rsun * u.kpc

    def circular_velocity(self, q):
        rq = np.sqrt(q[0] ** 2 + q[1] ** 2).to(u.kpc)
        return self.vcircsun * (rq / self.rsun).value

    def oort_ab(self, q):
        return 0.0 * u.km / u.s / u.kpc, -self.vcircsun / self.rsun

    def addinfo(self):
        return f"Solid body rotation curve\n vcircsun={self.vcircsun}, Rsun={self.rsun}"


class BrunettiPfennigerRotationCurve(RotationCurve):
    """
    Implements a very simple kinematic model of the disk in which the circular velocity follows the rotation curve from Brunetti & Pfenniger, 2010, https://ui.adsabs.harvard.edu/abs/2010A%26A...510A..34B/abstract
    """

    def __init__(self, vcircsun, rsun, h, p):
        """
        Class constructor/initializer.

        Parameters
        ----------

        vcircsun: float
            Circular velocity at the position of the Sun in km/s.
        rsun: float
            Galactocentric distance of the Sun in kpc.
        h : float
            Potential scale length in kpc
        p : float
            Exponent p in rotation curve equation
        """
        self.vcircsun = vcircsun * u.km / u.s
        self.rsun = rsun * u.kpc
        self.h = h * u.kpc
        self.p = p
        self.v0 = (
            self.vcircsun
            / (
                self.rsun
                / self.h
                * np.power(1 + (self.rsun / self.h) ** 2, (self.p - 2) / 4)
            )
        ).to(u.km / u.s)

    def circular_velocity(self, q):
        rq = np.sqrt(q[0] ** 2 + q[1] ** 2).to(u.kpc)
        return self.v0 * (
            rq / self.h * np.power(1 + (rq / self.h).value ** 2, (self.p - 2) / 4)
        )

    def oort_ab(self, q):
        rq = np.sqrt(q[0] ** 2 + q[1] ** 2).to(u.kpc)
        rhsqr = rq * rq / (self.h * self.h)
        rhterm = 1.0 + rhsqr
        dvdr = (
            self.v0
            / self.h
            * np.power(rhterm, (self.p - 2) / 4)
            * (1.0 + (self.p - 2) / 2 * rhsqr * np.power(rhterm, -1))
        )
        oortA = 0.5 * (self.circular_velocity(q) / rq - dvdr)
        oortB = 0.5 * (-self.circular_velocity(q) / rq - dvdr)
        return oortA, oortB

    @staticmethod
    def oort_ab_static(rcyl, vc, hbp, pbp):
        """
        Calculate the Oort A and B constants for the specific input parameters.

        Parameters
        ----------
        rcyl : float array-like
            Distance R from Galactic centre in cylindrical coordinates (kpc).
        vc : float array-like
            Circular velocity at the position of the sun (km/s).
        hbp : float, array-like
            Scale length of potential (kpc).
        pbp: float, array-like
            Shape parameter for potential.
        """
        rhsqr = rcyl * rcyl / (hbp * hbp)
        rhterm = 1.0 + rhsqr
        vv0 = vc / (rcyl / hbp * np.power(1 + rhterm, (pbp - 2) / 4))
        dvdr = (
            vv0
            / hbp
            * np.power(rhterm, (pbp - 2) / 4)
            * (1.0 + (pbp - 2) / 2 * rhsqr * np.power(rhterm, -1))
        )
        oortA = 0.5 * (vc / rcyl - dvdr)
        oortB = 0.5 * (-vc / rcyl - dvdr)
        return oortA, oortB

    def addinfo(self):
        return f"Brunetti & Pfenniger rotation curve\n vcircsun={self.vcircsun}, Rsun={self.rsun}, h={self.h}, p={self.p}"


class DiskKinematicModel:
    """
    Implements a very simple kinematic model for the Milky Way disk. This assumes stars describe strictly circular orbits
    around the vertical axis of the Milky Way disk plane. Thus in galactocentric cylindrical coordinates the velocity
    vectors of the stars are (V_R, V_phi, V_z) = (0, V_phi(R), 0), where V_phi(R) is the rotation curve of the disk. The
    velocity field does not change with z.
    """

    def __init__(self, rotcurve_instance, sunpos, vsunpeculiar):
        """
        Class constructor/initializer.

        Parameters
        ----------

        rotcurve_instance: RotationCurve instance
            The RotationCurve instance from which the rotation curve will be extracted for the disk kinematic model.
        sunpos: astropy Quantity, 3-element array
            3-vector with the sun's position in the Milky Way in galactocentric Cartesian coordinates. Default units are kpc.
        vsunpeculiar: astropy Quantity, 3-element array
            Sun's peculiar velocity in galactocentric Cartesian coordinates. Default units are km/s.
        """
        self.rotcurve = rotcurve_instance
        self.sunpos = sunpos
        self.vsunpec = vsunpeculiar
        self.vphisun = -self.rotcurve.circular_velocity(self.sunpos)
        self.vsun = np.array([0, -self.vphisun.value, 0]) * u.km / u.s + self.vsunpec

    def get_circular_velocity(self, pos):
        """
        Get the circular velocity at the input positions.

        Parameters
        ----------

        pos: float array of shape (3,N)
            Array of positions for which to retrieve the circular velocity, units of kpc.

        Return
        ------

        Circular velocity as array of shape (N). Units of km/s.
        """
        return self.rotcurve.circular_velocity(pos)

    def get_oort_ab(self, q):
        """
        Get the Oort A and B parameters at the position q.

        Parameters
        ----------

        q : gala PhaseSpacePosition, Quantity, array_like
            The position for which to obtain A and B

        Returns
        -------

        Oort A and B parameters in km/s/kpc
        """
        return self.rotcurve.oort_ab(q)

    def getinfo(self):
        """
        Returns
        -------

        info: str
            String with information on the kinematic model.
        """
        return self.rotcurve.getinfo()

    def observables(self, distance, l, b, vsunpec=np.nan, sunpos=np.nan):
        """
        Calculate the proper motions and radial velocities for stars at a given distance and galactic coordinate (l,b).

        Parameters
        ----------

        distance: astropy length Quantity, float array
            The distances to the stars. Default unit is kpc.
        l: astropy angle-like Quantity, float array
            The Galactic longitude of the stars. Default unit is radians.
        b: astropy angle-like Quantity, float array
            The Galactic latitude of the stars. Default unit is radians.

        Keyword arguments
        -----------------

        vsunpec: astropy quantity, float 3-array
            Custom value for the sun's peculiar velocity, by default same as value used for model initialization
        sunpos: astropy quantity, float 3-array
            Custom value for the sun's position, by default same as value used for model initialization

        Returns
        -------

        Proper motions in l and b, and the radial velocities. Units are mas/yr and km/s.
        pml, pmb, vrad = observables(distance, l, b).
        """
        p, q, r = normal_triad(l, b)
        if np.any(np.isnan(vsunpec)):
            vsunpec = self.vsunpec
        if np.any(np.isnan(sunpos)):
            sunpos = self.sunpos
        vsun = np.array([0, -self.vphisun.value, 0]) * u.km / u.s + vsunpec

        starpos = ((distance * r).T + sunpos).T
        vphistar = -self.rotcurve.circular_velocity(starpos)
        phi = np.arctan2(starpos[1, :], starpos[0, :])

        vstar = np.vstack(
            (-vphistar * np.sin(phi), vphistar * np.cos(phi), np.zeros_like(vphistar))
        )
        vdiff = (vstar.T - vsun).T

        vrad = np.zeros(distance.size) * u.km / u.s
        pml = np.zeros(distance.shape) * u.mas / u.yr
        pmb = np.zeros(distance.shape) * u.mas / u.yr
        for i in range(distance.size):
            vrad[i] = np.dot(r[:, i], vdiff[:, i])
            pml[i] = (
                (
                    np.dot(p[:, i], vdiff[:, i]).to(u.km / u.s)
                    / (distance[i].to(u.kpc) * _au_km_year_per_sec)
                ).value
                * u.mas
                / u.yr
            )
            pmb[i] = (
                (
                    np.dot(q[:, i], vdiff[:, i]).to(u.km / u.s)
                    / (distance[i].to(u.kpc) * _au_km_year_per_sec)
                ).value
                * u.mas
                / u.yr
            )

        return pml, pmb, vrad

    def differential_velocity_field(self, xgrid, ygrid, z):
        """
        Calculate the differntial velocity field for a grid in galactocentric Cartesian (x,y) and fixed z.

        Parameters
        ----------

        xgrid: astropy Quantity, float array, shape (N,N)
            Values of galactocentric x-coordinates over grid (as generated with np.mgrid for example). Default units are kpc.
        ygrid: astropy Quantity, float array, shape (N,N)
            Values of galactocentric y-coordinates over grid (as generated with np.mgrid for example). Default units are kpc.
        z: astropy Quantity, float
            Value of the fixed galactocentric z-coordinate. Default unit is kpc.

        Return
        ------

        The differential velocity at each (x, y, z) as proper motions, radial velocities, and tangential velocities, all with respect to the solar system barycentre. In addition return the normal triad vectors for each (x,y,z).

        pml, pmb, vrad, vtan, p, q, r = differential_velocity_field(xgrid, ygrid, z). Units mas/yr, mas/yr, km/s, km/s
        """

        zgrid = np.zeros_like(xgrid) + z
        phi = np.arctan2(ygrid, xgrid)

        vphistar = -self.rotcurve.circular_velocity(np.stack([xgrid, ygrid, zgrid]))
        print(vphistar.shape)
        vstar = np.stack(
            (-vphistar * np.sin(phi), vphistar * np.cos(phi), np.zeros_like(vphistar))
        )
        vdiff = (vstar.T - self.vsun).T

        dist, b, l = cartesian_to_spherical(
            xgrid - self.sunpos[0], ygrid - self.sunpos[1], zgrid - self.sunpos[2]
        )
        p, q, r = normal_triad(l, b)

        vrad = np.zeros(xgrid.shape) * u.km / u.s
        pml = np.zeros(xgrid.shape) * u.mas / u.yr
        pmb = np.zeros(xgrid.shape) * u.mas / u.yr
        vtan = np.zeros(xgrid.shape) * u.km / u.s
        for i in range(xgrid.shape[0]):
            for j in range(xgrid.shape[1]):
                vrad[i, j] = np.dot(r[:, i, j], vdiff[:, i, j])
                pml[i, j] = (
                    (
                        np.dot(p[:, i, j], vdiff[:, i, j]).to(u.km / u.s)
                        / (dist[i, j].to(u.kpc) * _au_km_year_per_sec)
                    ).value
                    * u.mas
                    / u.yr
                )
                pmb[i, j] = (
                    (
                        np.dot(q[:, i, j], vdiff[:, i, j]).to(u.km / u.s)
                        / (dist[i, j].to(u.kpc) * _au_km_year_per_sec)
                    ).value
                    * u.mas
                    / u.yr
                )
        vtan = np.sqrt(
            vdiff[0, :, :] ** 2 + vdiff[1, :, :] ** 2 + vdiff[2, :, :] ** 2 - vrad**2
        )

        return pml, pmb, vrad, vtan, p, q, r
