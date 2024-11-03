# Mental-Health-and-Screentime
This repository contains the code I contributed to a university coursework, where we were to investigate the link between mental health and screentime in adolescents. We were given a raw data file containing responses to a survey about the amount of time spent on screens at young ages (along with other factors). It also contained metrics used to indicate the mental health of the survey responders, such as 'Depression Score'. Our task was to use this data to investigate if there were any links to be found between mental health and time spent on screens. This repository contains code that was used to preprocess and clean the data, along with different analysis techniques including regression and correlation analysis.

Files:
- data_clean_nn.py - this Python script preprocesses the raw data file for machine learning & analysis by transforming categorical survey responses into numerical formats. For example, 'Yes/No' survey answers are converted to binary indicators, and numerical ranking systems are applied to categorical variables such as education levels. The script also processes time intervals into numerical values imputes and NaNs.
- regressions_st_only_mode_imputed.ipynb - notebook file containing correlation and regression analysis of the preprocessed data.
  - I start by loading the pre-processed CSV as a Pandas Dataframe and cleaning up the data further, removing any variables not related to mental health or screentime.
  - I attempt to find linear and non-linear relationships between the mental health (target) and screentime (feature) variables, using various correlation coefficients, and visualise this using heatmaps.
  - Linear regression models are then fitted for each of the target variables, and the R^2 and error of these regressions are calculated to assess the performance of the models. The model coefficient of each feature is presented in a line graph in an attempt to show which screentime indicators have the biggest effect on mental health.
  - In the analysis of this project, I talked about how although some of the features have relatively high coefficient values, the R^2 score of each model is very low and thus there are likely no linear relationships between any of the mental health and screentime variables. The low correlation coefficients support this.
- NN_regression_depscore.ipynb - 
