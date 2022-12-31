# Folder for input data from Gaia DR3

This folder is expected to contain the input data from Gaia DR3 needed for the rotation curve modelling. The data are not stored in this Github repository. The Gaia DR3 archive queries for obtaining the necessary data are:

### OBA Golden sample from [Gaia Collaboration Creevey, et al., 2022, A&A](https://doi.org/10.1051/0004-6361/202243800)

```sql
select gaia.source_id, gaia.ra, gaia.dec, gaia.parallax, gaia.parallax_error, gaia.pmra, gaia.pmra_error,
gaia.pmdec, gaia.pmdec_error, gaia.parallax_pmra_corr, gaia.parallax_pmdec_corr, gaia.pmra_pmdec_corr, 
gaia.radial_velocity, gaia.radial_velocity_error, gaia.rv_template_teff, gaia.grvs_mag, 
gaia.phot_g_mean_mag, gaia.bp_rp, gaia.g_rp, gaia.bp_g, 
gaia.ag_gspphot, gaia.ebpminrp_gspphot, gaia.teff_gspphot, gaia.logg_gspphot,
aps.teff_esphs, aps.logg_esphs, aps.ag_esphs, aps.ebpminrp_esphs, aps.spectraltype_esphs, oba.vtan_flag
from gaiadr3.gold_sample_oba_stars as oba
join gaiadr3.astrophysical_parameters as aps
using (source_id)
join gaiadr3.gaia_source as gaia
using (source_id)
```

### FGKM Golden sample from [Gaia Collaboration Creevey, et al., 2022, A&A](https://doi.org/10.1051/0004-6361/202243800)

```sql
select fgkm.*, gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec, 
gaia.pmra_error, gaia.pmdec_error, gaia.parallax_error,
gaia.pmra_pmdec_corr, gaia.parallax_pmra_corr, gaia.parallax_pmdec_corr,
gaia.radial_velocity, gaia.radial_velocity_error, gaia.rv_template_teff, gaia.grvs_mag,
gaia.phot_g_mean_mag, gaia.bp_rp, gaia.bp_g, gaia.g_rp, ap.abp_gspphot, ap.arp_gspphot
from gaiadr3.gaia_source as gaia
join gaiadr3.gold_sample_fgkm_stars as fgkm
using (source_id)
join gaiadr3.astrophysical_parameters as ap
using (source_id)
```