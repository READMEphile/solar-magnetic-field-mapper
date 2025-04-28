# Solar Magnetic Field Mapper with Machine Learning

**Author:** Moulik Mishra
**Institution:**  Shiv Nadar University
**Project Type:** Computational Astrophysics and Machine Learning  
**Programming Language:** MATLAB  
**Project Status:** Completed


## Overview
This project involves the simulation, visualisation, and analysis of solar magnetogram data to study solar magnetic field evolution and predict solar flare events. It integrates principles from computational physics, image processing, feature engineering, and machine learning.

The system generates realistic solar magnetogram simulations, extracts physically significant features related to magnetic complexity, and trains machine learning models to forecast flare activities based on the extracted features.

## Key Components
- **Simulated Solar Magnetogram Data**  
  Includes active regions with realistic characteristics such as limb darkening, magnetic polarity structures, and noise effects.
  
- **Feature Extraction Techniques**  
  - Gradient-based magnetic complexity
  - Total unsigned and net magnetic flux
  - Polarity inversion line lengths
  - Fractal dimension estimation
  - Flux asymmetry
  
- **Sunspot Analysis**  
  - Detection and tracking of sunspots based on magnetic field strength thresholds
  - Temporal analysis of sunspot counts and areas
  
- **Machine Learning for Flare Prediction**  
  - Model training using Decision Trees, Support Vector Machines (SVM), k-Nearest Neighbours (KNN), and Ensemble Methods
  - Feature importance evaluation
  - Confusion matrix and performance metrics
  
- Visualisation and Reporting
  - Magnetogram plots with magnetic field streamlines
  - Statistical visualisations of magnetic field evolution
  - Principal Component Analysis (PCA) visualisation of feature spaces
  - Forecast time-series for flare probabilities

## Technical Requirements
- MATLAB (R2021a or later recommended)
- MATLAB Toolboxes (optional but recommended):
  - Symbolic Math Toolbox
  - Image Processing Toolbox
  - Statistics and Machine Learning Toolbox
  - Optimization Toolbox (optional)

## Project Structure
```
/solar_magnetic_field_mapper
├── cod.txt (source code file)
├── solar_data/ (directory for simulated magnetogram images)
├── flare_prediction_model.mat (saved machine learning model)
├── README.md (project documentation)
```

## Execution Instructions
1. Download or clone this repository.
2. Open `cod.txt` in MATLAB.
3. Run the script. A console-based menu will guide you through the following actions:
   - Download (simulated) magnetogram data
   - Load existing data
   - Visualise magnetic field evolution
   - Analyse sunspot activity
   - Train a machine learning model for flare prediction
   - Predict flare probabilities using the trained model
4. The program will generate and display all necessary plots, forecasts, and reports.

## Representative Outputs
- Solar magnetogram visualisations with magnetic field streamlines
- Sunspot detection maps
- Time series of total magnetic flux and sunspot activity
- Machine learning model comparisons and performance reports
- Probabilistic forecasts of flare activity

## Physical Motivation
Magnetic activity on the solar surface, particularly in complex sunspot regions, is a key driver of solar flare events.  
This project aims to quantify the complexity of the solar magnetic field using physical and statistical features and utilise these indicators for predictive modelling of solar flares, an important aspect of space weather forecasting.

# Possible Future Enhancements
- Integration of real Solar Dynamics Observatory (SDO) HMI data via NASA public data repositories.
- Development of deep learning-based prediction models utilising raw magnetogram images.
- The system will be deployed as a MATLAB App Designer application for improved user interaction.
- Expansion to longer-term (multi-day) flare prediction frameworks.

# Acknowledgments
- Data inspiration from the Solar Dynamics Observatory (SDO) mission.
- Techniques motivated by current research in solar physics and space weather prediction.

# Contact
For any queries, collaborations, or further discussions regarding this project, please contact mm748@snu.edu.in.


# Notes
- This project demonstrates proficiency in scientific computing, image analysis, feature engineering, and supervised machine learning workflows.
- The work is designed to be extendable for future research or professional applications in computational astrophysics and space weather modeling.
