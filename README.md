# Integrated Energy Grids – DTU 46770 Course Project

This repository contains materials related to the MSc course 46770 Integrated Energy Grids at the Technical University of Denmark (DTU). The course focuses on the modeling, simulation, and optimization of integrated energy systems, including electricity, heat, gas, and hydrogen networks.

## Project Overview

The course project consists of building and analyzing a simplified national-scale energy system model using PyPSA (Python for Power System Analysis). The objective is to explore how different energy carriers and infrastructure interact under future decarbonization targets.

For this project, Germany is selected as the base country. The analysis is broken down into multiple tasks, each implemented in a dedicated notebook:
	•	Part_A_analysis.ipynb to Part_H_analysis.ipynb: Cover progressive modeling steps (e.g., generation, storage, CO₂ constraints, grid bottlenecks).
	•	base.py: Contains the base PyPSA model setup used across tasks.
	•	visualization.py: Utility functions for plotting and result interpretation.

Full project description: [IEG Course Project Assignment](https://martavp.github.io/integrated-energy-grids/Problems/IEG_course_project.html)

## Environment Setup

To run the notebooks, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/ieg-germany-model.git
cd ieg-germany-model

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

This environment is based on the one provided for the course. You may also consider using a pre-configured Conda environment if available.

## File Structure

```bash
├── Part_A_analysis.ipynb       # Initial setup & basic modeling
├── Part_B_analysis.ipynb       # Expanding components
├── ...
├── Part_H_analysis.ipynb       # Final analysis on grid bottlenecks
├── base.py                     # Base model definition
├── visualization.py            # Plotting and result utilities
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## License and Solver

This project uses Gurobi as the optimization solver under an academic license.

