> ## **⚠️ This repository has moved to Codeberg at [codeberg.org/sarttiso/stratage](https://codeberg.org/sarttiso/stratage).⚠️**
> 
> All future development will occur there. This GitHub repository is archived and read-only.

<img src='https://github.com/sarttiso/stratage/raw/main/graphics/logo.png' alt="Logo for StratAge, featuring a stylized geological stratigraphy on the left with stacked layers in shades of brown and green, and a probabilistic age-depth envelope on the right, with probability distributions corresponding to geochronology overlain." width="200px">

# StratAge

StratAge presents a novel approach to age-depth modeling of geologic stratigraphy. By modeling the distribution of time *within* beds as sedimentation rates and *between* beds as hiatuses, StratAge can leverage the structure of stratigraphy to inform the distribution of time within it.

StratAge uses Markov Chain Monte Carlo sampling to estimate posterior distributions of sedimentation rates and hiatus durations within a user-provided stratigraphy, given probabilistic geochronologic constraints. These model parameters can then be anchored in absolute time to yield a Bayesian age-depth model.

## Installation

StratAge can be installed from PyPI:

```bash
pip install stratage
```

## Usage

The best way to get a feel for StratAge is to read through the [Toy Example](https://adrian.tasistro-hart.com/stratage/toy-example.html), which presents the basic functionality of StratAge on a synthetic stratigraphy.

Full documentation can be found [here](https://adrian.tasistro-hart.com/stratage/), which also links to the [package API](https://adrian.tasistro-hart.com/stratage/api/modules.html).

## Cite

The methodology for the age modeling approach implemented by `StratAge` is presented in the following reference:

> Tasistro-Hart, A.R., Ganti, V., and Macdonald, F.A., 2026, Completeness and compensational sedimentation of ANDRILL AND-2A through the Miocene Climate Optimum: Geological Society of America Bulletin, doi:10.1130/B38710.1.
