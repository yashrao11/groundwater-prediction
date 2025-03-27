# Groundwater Level Prediction System 🌊

A machine learning-based web application to predict groundwater levels for selected stations in India. This system aggregates historical groundwater data, fills in missing data for specific periods, and forecasts future water levels (e.g., for 2025-2030) with interactive visualizations.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model and Methodology](#model-and-methodology)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Introduction

Groundwater is a critical resource for agriculture, drinking water, and industrial processes. However, over-extraction, climate change, and poor water management have led to alarming levels of depletion. This project aims to predict future groundwater levels by leveraging historical data and machine learning models.

The system allows users to:
- **Select a station code** from a dropdown list.
- View **historical trends** (aggregated yearly, including filled data for missing years between 2019 and 2024).
- See a **forecast** of groundwater levels for the next 6 years (2025-2030).
- Understand the process through an interactive flowchart.

---

## Features

- **User-Friendly Interface:**  
  Interactive web application built using Streamlit with attractive UI elements and custom styling.

- **Historical Data Aggregation:**  
  Aggregates daily groundwater data into yearly averages and fills missing data for the period 2019–2024 using a baseline model.

- **Future Forecasting:**  
  Predicts groundwater levels for the years 2025-2030 using a trained Linear Regression model.

- **Interactive Visualizations:**  
  Displays historical and forecasted data in dynamic plots along with an explanatory flowchart.

- **Dropdown Station Selector:**  
  Allows users to select the exact station code from a list to view station-specific data.

---

## Installation

### Prerequisites

- Python 3.7 or above
- Git

### Dependencies

All necessary dependencies are listed in the `requirements.txt` file. They include:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

### Steps to Install

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/groundwater_prediction_system.git
   cd groundwater_prediction_system
