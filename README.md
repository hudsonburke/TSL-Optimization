# TSL-Optimization

Supporting materials for the ASB 2025 poster: "Optimizing Tendon Slack Length in Musculoskeletal Models Using Movement-Specific Kinematics"

![ASB_Poster](./ASB_Poster.pdf)

## Overview

This repository contains the code, data, and analysis for a study on optimizing tendon slack length (TSL) parameters in musculoskeletal models. The work implements optimization algorithms to find optimal tendon slack lengths for muscles using the Manal 2004 muscle model and evaluates these parameters against different movement conditions (walking, cutting, and squatting).

## Repository Contents

### Main Analysis
- **`asb_2025_poster.ipynb`** - Complete analysis workflow including data loading, TSL optimization, results visualization, and statistical comparisons across movement types

### Source Code (`src/`)
- **`tsl_optimization.py`** - Core optimization functions implementing SSD and SSDP objective functions for tendon slack length estimation
- **`muscle_params.py`** - Muscle parameter calculations including pennation angles, tendon forces, and slack length computations drawn from Manal 2004. 
- **`curve_wrapper.py`** - Unified interface for OpenSim force-length curves and numpy array interpolation with caching for computational efficiency
- **`manal_curves.py`** - Implementation of Manal 2004 active, passive, and tendon force-length relationships
- **`osim_graph.py`** - Graph-based representation of OpenSim models for analyzing muscle attachments and joint crossings

### Data (`data/`)
- **`Walking_IK.mot`** - Inverse kinematics results for normal gait cycle
- **`Cutting_IK.mot`** - Kinematics for sports cutting maneuver
- **`Squatting_IK.mot`** - Deep squat movement kinematics

### Model (`models/`)
- **`RajagopalLaiUhlrich2023.osim`**

## Getting Started

All analysis and results are contained in the Jupyter notebook `asb_2025_poster.ipynb`. This notebook demonstrates the complete workflow from data loading through optimization and visualization of results.

## Contact

Hudson Burke - [@hudsonburke](https://github.com/hudsonburke)

*Presented at the American Society of Biomechanics Annual Meeting 2025*
