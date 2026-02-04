# ALK-electrolyzer-model_UNIFI
Main repository for a physics-based, parametric model of an alkaline (ALK) water electrolyzer cell developed in Python by the University of Florence (Italy).
This repository provides a lightweight Python implementation to generate ALK cell performance curves and efficiency trends as a function of operating conditions. The model combines standard electrochemical relationships (reversible voltage, activation losses, ohmic losses) with literature-based correlations and a temperature-dependent Faraday efficiency fit. References are embedded directly in the source code as DOI links in the comments.

---

## Citation

If the model is used in scientific work, please cite:

Francesco Superchi, Francesco Papi, Andrea Mannelli, Francesco Balduzzi, Francesco Maria Ferro, Alessandro Bianchini,
"Development of a reliable simulation framework for techno-economic analyses on green hydrogen production from wind farms using alkaline electrolyzers"  
Renewable Energy,Volume 207,2023,Pages 731-742,ISSN 0960-1481,https://doi.org/10.1016/j.renene.2023.03.077.

---

## What is included

- **Physics-based ALK cell model (ElectroCellALK)**

	- Polarization curve generation (current density range up to rated value)

	- Reversible voltage in standard conditions plus Nernst correction (pressure and vapor terms)

	- Activation overpotentials via Tafel formulation (anode and cathode)

	- Ohmic losses including electrode contributions and an empirical resistance term

	- Faraday efficiency correlation with optional temperature-dependent coefficient interpolation

	- Thermal model



## Design updates and contributions

Design updates are welcome via Pull Requests or by contacting the authors. When contributing, it is recommended to:

- document new correlations and assumptions in code comments (with DOI/reference),

- include a minimal test or example that reproduces expected trends.


