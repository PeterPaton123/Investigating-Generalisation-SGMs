# Investigating Generalisation In Score-Based Generative Models

This repository contains the code and libraries created and used in supporting experiments for my masters's project on `Investigating Generalisation In Score-Based Generative Models'. 
Many experiments require building the diffusion jax library which is included as a git submodule.

## Requirements
1. Install Python3.10+

## Setup
1. Check out the repo and the submodules: `git clone --recurse-submodules git@github.com:PeterPaton123/High-Performance-Numerical-Differential-Equation-Solver.git`
2. Build the DiffisionJax submodule as detailed in the submodule README.md 
3. Set up a virtual environment for python libraries: `python3 -m venv .`
4. Install those python libraries: `pip install -r requirements.txt`
5. Open the virtual environment: `. venv/bin/activate`

## Run

To run a given experiment:
1. Open a terminal in the same directiory as the associated python file
2. Ensure any absolute imports are correctly pointing to the associated imports
3. Run the program: 'python3 Example.py'
