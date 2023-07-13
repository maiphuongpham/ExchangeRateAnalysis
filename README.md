# Exchange Rate Analysis
Author: Mai Pham

Columbia University

## Datasets
The dataset used in this project includes daily exchange rates in USD and monthly geopolitical risk (GPR) indices for 10 countries from January 2005 to November 2022. The daily data was resampled to represent monthly data, resulting in 4647 daily data points and 213 monthly data points collected. Additionally, 214 monthly interest rate data points were obtained from OECD to support later sections of the project.

## Project Goals
Foreign exchange analysis and prediction are valuable for trading strategies and gaining insights into economies. Geopolitical risk has been correlated with exchange rate returns, raising intriguing questions about the impact of events like the Russia-Ukraine war or terrorist attacks on currency asymmetries and optimal currency portfolios during conflicts.

For individual exchange rates, we use statistical analysis techniques to examine the distribution of exchange rate returns and identify trends that may indicate currency appreciation or depreciation. These observations also shed light on global forex market issues, such as inflation.

When analyzing pairs of exchange rates, we test for equal means and perform regression analysis to determine the correlation and movement patterns between currencies.

Additionally, we explore the influence of the geopolitical risk (GPR) index on exchange rate returns. The regressions conducted in this project aim to contribute to exchange rate prediction, which remains a challenging task in finance and economics and is subject to debate in the context of the random walk theory.

## Brief Results
This project analyzed exchange rate returns and investigated the impact of geopolitical risk. The findings show that exchange rate returns deviate from a normal distribution, varying across countries. Further research is needed to identify the macro-factors influencing these distributions.

Single exchange rate analysis revealed non-normal distribution, with some statistically significant historical returns. However, complex non-linear algorithms may be required to explain high-volatility exchange rate returns.

Analysis of pairs of exchange rates showed significant differences in means and varying relationships between currencies, suggesting higher correlation among European countries and more independent behavior in certain Asian countries.

Geopolitical risks were found to have negative effects on exchange rate returns. Including additional regressors, such as oil returns and stock market returns, can improve performance. Future research should normalize data, increase observations, or adjust dependent variables to reduce error terms.

The project was conducted using Python on Google Collaboratory, and the project codes can be found here for more detailed information about the methods.