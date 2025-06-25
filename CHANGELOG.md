# Changelog

## v1.1

### Added
- parallelization of the `stratage.Geochron._time_increment_pdfs()` to speed up numerical computation of time increment distributions
- improve gridding and efficiency of numerical computation of time increment distributions

## v1.0

### Features
- `Geochron` class for creating time-increment distributions
- `AgeModel` class for combining priors, stratigraphy, and a Geochron object to sample sedimentation rates and hiatus durations via `PyMC`
    - parallelized anchoring of floating age models to create probabilistic absolute age models
