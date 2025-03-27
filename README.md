# Groundwater Level Prediction System 🌊

A machine learning-based web application that predicts groundwater levels for selected stations in India.

## Table of Contents

- [Introduction](#introduction)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model and Methodology](#model-and-methodology)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Introduction

Groundwater is a critical resource for agriculture, drinking water, and industry. This project develops a system that leverages historical data and machine learning to forecast groundwater levels, helping users (e.g., policymakers, farmers) plan more effectively.

## Usage

To run the application locally, follow these steps:
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Place the dataset `gwl_daily.csv` in the project directory.
4. Run the app with `streamlit run app.py`.

## Project Structure


## Model and Methodology

The project uses historical groundwater data to forecast future levels:
- **Data Preprocessing:** Dates are converted, and features such as Year are extracted. Numeric features are normalized.
- **Model Training:** A Linear Regression model is trained on aggregated yearly data. Missing years (2019–2024) are filled, and predictions for 2025–2030 are generated.
- **Visualization:** Historical and forecasted data are plotted together to illustrate trends.

## Deployment

The app is deployed on Streamlit Community Cloud. Simply push the code to GitHub and connect your repository to Streamlit Cloud. The app will automatically install dependencies from `requirements.txt`.

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, commit your changes, and open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

1. Fan, Y., Li, H., & Miguez-Macho, G. (2013). Global patterns of groundwater table depth. *Science, 339(6122)*, 940-943.
2. Zhou, Y., & Li, W. (2011). A review of regional groundwater flow modeling. *Hydrogeology Journal, 19(1)*, 19-34.
3. Mukherjee, A., et al. (2020). Machine Learning Approaches for Groundwater Level Prediction: A Comprehensive Review. *Environmental Science & Technology, 54(6)*, 3276-3292.
4. Tiwari, K. K., et al. (2018). Application of Artificial Neural Networks for Groundwater Level Prediction in India. *Journal of Hydrology, 562*, 697-708.
5. United Nations World Water Report (2021). [https://www.unwater.org/](https://www.unwater.org/)


### Running the App Locally

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/groundwater_prediction_system.git
   cd groundwater_prediction_system/groundwater_app
