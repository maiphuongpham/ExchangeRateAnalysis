# For data manipulation
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from scipy.stats import norm

# To fetch financial data
import yfinance as yf

# For visualisation
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
%matplotlib inline
from google.colab.data_table import DataTable
DataTable.max_columns = 40
plt.rcParams.update({'font.size': 6})

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Set the ticker as 'EURUSD=X'
ticker_list = ['RUBUSD=X', 'UAHUSD=X', 'GBPUSD=X', 'SARUSD=X', 'ILSUSD=X',
                'CNYUSD=X', 'KRWUSD=X', 'TRYUSD=X', 'JPYUSD=X', 'CADUSD=X',
                'ARSUSD=X', 'AUDUSD=X', 'BRLUSD=X', 'CHFUSD=X', 'CLPUSD=X',
                'COPUSD=X', 'DKKUSD=X', 'EGPUSD=X', 'HKDUSD=X', 'HUFUSD=X',
                'IDRUSD=X', 'INRUSD=X', 'MXNUSD=X', 'NOKUSD=X', 'PENUSD=X',
                'PHPUSD=X', 'PLNUSD=X', 'SEKUSD=X', 'THBUSD=X', 'TNDUSD=X',
                'TWDUSD=X', 'VEFUSD=X', 'ZARUSD=X', 'MYRUSD=X']

forex_data = yf.download(
    ticker_list,
    start='2005-01-01',
    end='2022-11-02')

# Set the index to a datetime object
forex_data.index = pd.to_datetime(forex_data.index)

# Display the last five rows
display(forex_data.tail())

# Save forex data to drive
forex_data.to_csv('data.csv')

# Trim data to keep adj price
n_var = 34
data = forex_data.iloc[:, :n_var]

# Remove level 0 of column names (adj price)
data.columns = data.columns.get_level_values(1)

# Rename columns
column_names = data.columns.tolist()
for col in range(0,len(column_names)):
  column_names[col] = column_names[col][:3]
data.columns = column_names

# Interpolate the null values
data = data.interpolate(
    method='linear',
    limit_direction='forward',
    axis=0)

# DESCRIPTIVE STATISTICS

# Resample by taking the last day of each month
monthly_data = data.asfreq('M', method='pad')

# Calculate monthly log returns
monthly_log_returns = np.log(
    monthly_data / monthly_data.shift(1))

# Drop the first row (NaN value)
monthly_log_returns.drop(
    index=monthly_log_returns.index[0],
    axis=0,
    inplace=True)

# Display the last five rows
display(monthly_log_returns.describe().round(3))

# Calculate skewness and kurtosis
skew_and_kurt = pd.concat([monthly_log_returns.skew(),
                           monthly_log_returns.kurt()], axis=1).T.round(3)
skew_and_kurt.index = ['skew', 'kurt']
display(skew_and_kurt)

def draw_prob_plot(df, variables, n_rows, n_cols):
    """
    Draw histograms for the specified variables in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        variables (list): A list of variable names to plot.
        n_rows (int): Number of rows in the subplot grid.
        n_cols (int): Number of columns in the subplot grid.

    Returns:
        None
    """
    # Create a figure
    fig = plt.figure(figsize=[13, 13], dpi=500)

    # Add subplots
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        # Plot
        st.probplot(monthly_log_returns[var_name], dist="norm", plot=plt)
        # Set title
        ax.set_title(var_name + " Probability Plot", fontsize=10)
        ax.set_xlabel("Theoretical Quantiles", fontsize=9)
        ax.set_ylabel("Ordered Values", fontsize=9)

    # Improve appearance a bit
    fig.tight_layout()
    # Display
    plt.savefig('Probability plots.png')
    plt.show()

    return None

# Display histograms for each exchange rates log returns
draw_prob_plot(
    monthly_log_returns,
    monthly_log_returns.columns,
    7, 5)

# HISTOGRAMS

# Make a function to plot histograms
def draw_histograms(df, variables, n_rows, n_cols):
    """
    Draw histograms for the specified variables in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        variables (list): A list of variable names to plot.
        n_rows (int): Number of rows in the subplot grid.
        n_cols (int): Number of columns in the subplot grid.

    Returns:
        None
    """
    # Create a figure
    fig = plt.figure(figsize=[13, 6], dpi=500)

    # Add subplots
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        # Plot
        df[var_name].hist(bins=30, ax=ax, color='#E18FAD')
        # Set title
        ax.set_title(var_name + " Distribution")

    # Improve appearance a bit
    fig.tight_layout()
    # Display
    plt.show()

    return None

# Display histograms for each exchange rates log returns
draw_histograms(
    monthly_log_returns,
    monthly_log_returns.columns,
    5, 7)

# Fit normal distributions to data
def draw_histograms_and_pdfs(df, variables, n_rows, n_cols):
    """
    Draw histograms and probability density functions (PDFs)
    for the specified variables in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        variables (list): A list of variable names to plot.
        n_rows (int): Number of rows in the subplot grid.
        n_cols (int): Number of columns in the subplot grid.

    Returns:
        None
    """
    fig = plt.figure(figsize=[9, 7], dpi=500)

    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        # Plot
        df[var_name].hist(bins=30, ax=ax, color='#E18FAD')
        # Set title
        ax.set_title(var_name+' Distribution')
        # Plot the PDF
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, norm.fit(df[var_name].dropna())[0],
                    norm.fit(df[var_name].dropna())[1])
        plt.plot(x, p, 'r', linewidth=1)

    # Improve appearance a bit
    fig.tight_layout()

    # Display
    plt.show()
    return None

# Plot the pdf with the histogram
draw_histograms_and_pdfs(monthly_log_returns, monthly_log_returns.columns, 6, 6)
plt.savefig('Distribution with normal pdf.png')

# SINGLE EXCHANGE RATE ANALYSIS

# Make a function to print confidence interval
def interval(df, confidence_level):
    """
    Calculate confidence intervals for the
    mean and variance of columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        confidence_level (float): The desired confidence level
        (between 0 and 1).

    Returns:
        pandas.DataFrame: A DataFrame with confidence intervals
        for the mean and variance of each column.
    """
    alpha = 1 - confidence_level
    t = []
    chisq = []

    # Apply to each columns of dataframe
    for i, var_name in enumerate(df.columns):
        # Calculate sample mean and variance
        mean, std = np.mean(df[var_name]), np.std(
            df[var_name], ddof=1)
        # Construct confidence intervals for mean
        t_interval = st.t.interval(
            alpha=alpha,
            df=len(df.columns)-1,
            loc=mean, scale=std)

        # Construct confidence intervals for variance
        var = np.var(df[var_name], ddof=1)

        chisq_left = st.chi2.ppf(1-alpha/2, df=len(df.columns)-1)
        chisq_right = st.chi2.ppf(alpha/2, df=len(df.columns)-1)

        chisq_interval = ((len(df)-1) * var / chisq_left,
                        (len(df)-1) * var / chisq_right)

        # Display the CI(s)
        t.append(t_interval)
        chisq.append(chisq_interval)

    t_results, chisq_results = np.array([t, chisq])

    columns = ['(L(mean),', 'U(mean))', '(L(var),', 'U(var))']
    results = pd.DataFrame(
        np.hstack([t_results, chisq_results]),
        index=df.columns,
        columns=columns).round(6)

    return results

# Find the confidence intervals on data at 0.05 level of significance
interval(monthly_log_returns, 0.95)

# Make a function to perform the first-order autoregression
def shift(df):
    """
    Shift the columns of a DataFrame by 1 timestep.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        pandas.DataFrame: The original DataFrame with the first row removed,
        and a new DataFrame with shifted columns.
    """

    # Make a dataframe of shifted data by 1 timestep
    shifted_df = pd.DataFrame()
    for i, var_name in enumerate(df.columns):
        shifted_df[var_name+'_t-1'] = df[var_name].shift(1)

    # Trim data to remove the first rows
    df = df.copy()
    df.drop(index=df.index[0], axis=0, inplace=True)
    shifted_df.drop(
        index=shifted_df.index[0],
        axis=0,
        inplace=True)

    return df, shifted_df


def first_order_autoregression(df):
    """
    Perform first-order autoregression analysis on the columns of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        list: A list of results including slope, standard error,
        t-test statistics, p-values, intercept,
        R-squared, and scatter plot figures.
    """
    # Make a dataframe of shifted data by 1 timestep
    df, shifted_df = shift(df)

    # Create some lists to display results
    slope = []
    intercept = []
    std_error_beta_1 = []
    t_beta_1 = []
    p_value_beta_1 = []
    std_error_beta_0 = []
    t_beta_0 = []
    p_value_beta_0 = []
    R2 = []

    # Make a fig
    fig = plt.figure(figsize=(39,9), dpi=500)
    predicted_df = pd.DataFrame()

    # Start the loop
    for i, var_name in enumerate(df.columns):

        # Define dependent var (y) and independent var (x)
        x = df[var_name].dropna().shift(1).dropna()
        y = df[var_name].dropna()
        Z = pd.concat([x, y], axis=1).dropna()
        x = Z.iloc[:,0]
        y = Z.iloc[:,1]

        # HERE I WANT TO CODE THE OLS MYSELF, AND COMPARE THE RESULTS
        # WITH THE ONE OBTAINED BY AVAILABLE LIBRARY

        # library model
        model = sm.OLS(
            endog=y.values,
            exog=x.values,
            hasconst=True)

        results = model.fit()
        print('='*97)
        print(' '*30, 'OLS SUMMARY FOR ', var_name)
        print(results.summary())

        # Estimate OLS parameters again by our formula in class
        cov_matrix = np.cov(x, y, ddof=1)

        # Slopes
        beta_1 = cov_matrix[0,1] / cov_matrix[0,0]
        slope.append(beta_1)

        # Intercepts
        beta_0 = np.mean(y) - slope[i]*np.mean(x)
        intercept.append(beta_0)

        # Predicted values
        yhat = intercept[i]*np.ones(len(x)) + slope[i]*x

        # Sum of squared
        SSR = np.sum((y - yhat)**2)
        Sxx = np.sum((x - np.mean(x))**2)
        SST = np.sum((y - np.mean(y))**2)

        # Standard Error of parameters
        std_error_beta_1.append(
            np.sqrt( (1 / (len(x)-2)) * (SSR / Sxx) ))

        std_error_beta_0.append(
            np.sqrt( ((sum(x**2)/len(x)) / (len(x)-2)) * (SSR / Sxx) ))

        # t test statistics and p-values
        t_beta_1.append(
            beta_1 / std_error_beta_1[i])
        p_value_beta_1.append(
            (1 - st.t.cdf(x=abs(t_beta_1[i]), df=len(x)-2))*2)

        t_beta_0.append(
            beta_0 / std_error_beta_1[i])
        p_value_beta_0.append(
            (1 - st.t.cdf(x=abs(t_beta_0[i]), df=len(x)-2))*2)

        # R-squared
        R2.append(1 - SSR/SST)

        # Scatter plots and linear regression lines
        ax = fig.add_subplot(4, 9, i+1)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        plt.xlabel(var_name+'(t-1)', size=20, fontweight='bold')
        plt.ylabel(var_name+'(t)', size=20, fontweight='bold')
        ax.set_title(var_name, size='large')
        sns.regplot(x=x.values,y=y.values, color='#E18FAD')

    # Display results in tabular form
    results = [slope, std_error_beta_1,
               t_beta_1, p_value_beta_1,
               intercept, std_error_beta_0,
               t_beta_0, p_value_beta_0, R2]

    print(' '*8, 'FIRST-ORDER REGRESSION ON THE WHOLE DATAFRAME:')

    columns = ['B1', 'SE(B1)', 't(B1)',
               'p(B1)', 'B0', 'SE(B0)',
               't(B0)', 'p(B0)', 'R2']
    display(pd.DataFrame(
        data=np.array(results).T.round(3),
        index=df.columns,
        columns=columns))

    # Improve appearance a bit
    fig.tight_layout(pad=0.5)

    # Display the figure
    plt.savefig('Scatter plots.png')

    fig = plt.figure(figsize=(39,9), dpi=500)

    for i, var_name in enumerate(df.columns):
        # Define dependent var (y) and independent var (x)
        x = shifted_df[var_name+'_t-1']
        y = df[var_name]

        # Scatter plots and linear regression lines
        ax = fig.add_subplot(4, 9, i+1)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=15)
        plt.xlabel(var_name+'(t-1)', size=20, fontweight='bold')
        plt.ylabel(var_name+'(t)', size=20, fontweight='bold')
        ax.set_title(var_name+' Scatter Plot and Regression Line', size='small')
        sns.residplot(x=x.values, y=y.values, lowess = False, color='#AF4C70')

    fig.tight_layout(pad=0.5)

    plt.savefig('Residual plots.png')

    return results

# Perform a regression of the log-return on time
autoregression = first_order_autoregression(monthly_log_returns)

# TWO EXCHANGE RATE ANALYSIS

# Test the equality of the two population means
# (use a test for paired data if appropriate)

def paired_test(df):
    """
    Perform paired t-tests between all pairs of columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        pandas.DataFrame, pandas.DataFrame: DataFrames with
        paired test statistics and p-values.
    """
    paired_test_statistics = pd.DataFrame(
        index=df.columns, columns=df.columns)

    paired_test_pvalues = pd.DataFrame(
        index=df.columns, columns=df.columns)

    for i, var_name_row in enumerate(df.columns):
        for j, var_name_col in enumerate(df.columns):
            paired_test_statistics.iloc[i,j] = st.ttest_rel(
                df[var_name_row],
                df[var_name_col],
                nan_policy='omit')[0]

            paired_test_pvalues.iloc[i,j] = st.ttest_rel(
                df[var_name_row],
                df[var_name_col],
                nan_policy='omit')[1]

    return paired_test_statistics, paired_test_pvalues

# Perform the test
paired_test_statistics, paired_test_pvalues = paired_test(monthly_log_returns)

# Display results
print(' '*40,'CORRELATION MATRIX:')
display(monthly_log_returns.corr().round(3))

print('='*130)

print(' '*40, 'PAIRED TEST T STATISTICS:')
display(paired_test_statistics.round(3))

print('='*130)

print(' '*40, 'PAIRED TEST P-VALUES:')
display(paired_test_pvalues.round(3))

# Perform a regression of one log-return on the other (all results)
def regression(endog, exog):
    """
    Perform linear regression on the given endogenous and exogenous variables.

    Args:
        endog (array-like): The endogenous variable.
        exog (array-like): The exogenous variable.

    Returns:
        pandas.DataFrame: A DataFrame with regression results
        including coefficients, t-values, p-values, and R-squared.
    """
    exog_const = sm.add_constant(exog)
    model = sm.OLS(endog=endog,exog=exog_const, hasconst=True, missing='drop')
    results = model.fit()

    # Tabular resuls
    table = pd.DataFrame(
        np.hstack(
            [results.params.values.T,
             results.tvalues.values.round(3).T,
             results.pvalues.values.T,
             results.rsquared]
        )[:,np.newaxis].round(4).T)

    table_columns = ['B1', 'B0', 't_1', 't_0', 'p-value_1', 'p-value_0', 'R2']

    table.columns = table_columns

    # Graphical results (enable to see)
    sns.regplot(x=exog,y=endog, color='#E18FAD')
    plt.title('Diagram of data with the least-squared line')
    plt.show()

    sns.residplot(x=exog, y=endog, lowess = False)
    plt.title('Graphical depiction of residuals')
    plt.show()

    return table

# Perform regression of all index on ARS
def regression_all(df):
    """
    Perform linear regression between all pairs of columns in a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    for i, var_name in enumerate(monthly_log_returns.columns):
        regression_all = pd.DataFrame()
        for j, var_name2 in enumerate(monthly_log_returns.columns):
            regression_all = regression_all.append(
                regression(df[var_name], df[var_name2]))
        regression_all.index = df.columns
        print('REGRESSION ON ', var_name)
        display(regression_all)

    return None

regression_all(monthly_log_returns)

# GEOPOLITICAL RISK AND EXCHANGE RETURN

# Load data:
gpr_data = pd.read_csv(
    'data_gpr_full.csv',
    index_col=0)

gpr_data.index = pd.to_datetime(gpr_data.index)

gpr = gpr_data.loc["01-01-2005":].copy(deep=True).asfreq('M', method='pad')

# Regression of GPR index on exchange rate return (1 factor)
def regression2(endog, exog):
    """
    Perform linear regression on the given endogenous and exogenous variables.

    Args:
        endog (array-like): The endogenous variable.
        exog (array-like): The exogenous variable.

    Returns:
        pandas.DataFrame: A DataFrame with regression results
        including coefficients, t-values, p-values, and R-squared.
    """

    exog_const = sm.add_constant(exog)
    model = sm.OLS(endog=endog,exog=exog_const, hasconst=True, missing='drop')
    results = model.fit()

    # Tabular resuls
    table = pd.DataFrame(
        np.hstack(
            [results.params.T,
            results.tvalues.round(3).T,
            results.pvalues.T,
            results.rsquared]
        )[:,np.newaxis].round(4).T)

    table_columns = ['B1', 'B0', 't_0', 't_1', 'p-value_0', 'p-value_1', 'R2']

    table.columns = table_columns

    # Graphical results
    sns.regplot(x=exog,y=endog, color='#E18FAD')
    plt.title('Diagram of data with the least-squared line')
    plt.show()

    sns.residplot(x=exog, y=endog, lowess = False)
    plt.title('Graphical depiction of residuals')
    plt.show()

    return table

# Perform OLS on all Exchange rate returns
regression_all2 = pd.DataFrame()

for i, var_name in enumerate(monthly_log_returns.columns):
    y = monthly_log_returns[var_name].values
    x = gpr['GPR_'+var_name].values
    regression_all2 = regression_all2.append(regression2(y, x))

regression_all2.index = monthly_log_returns.columns
regression_all2

# Regression of GPR index and Interest rate on exchange rate return (2 factors)
ir_data = pd.read_csv(
    'ir_data.csv',
    index_col=0).iloc[:-1,:]

ir_data.index = pd.to_datetime(ir_data.index)

def regression3(endog, exog):
    """
    Perform linear regression on the given endogenous and exogenous variables.

    Args:
        endog (array-like): The endogenous variable.
        exog (array-like): The exogenous variable.

    Returns:
        pandas.DataFrame: A DataFrame with regression results
        including coefficients, t-values, p-values, and R-squared.
    """
    exog_const = sm.add_constant(exog)
    model = sm.OLS(
        endog=endog,
        exog=exog_const,
        hasconst=True,
        missing='drop')
    results = model.fit()

    # Tabular resuls
    table = pd.DataFrame(
        np.hstack([
            results.params.T,
            results.tvalues.round(3).T,
            results.pvalues.T,
            results.rsquared]
        )[:,np.newaxis].round(4).T)

    table_columns = ['B1', 'B2', 'B0',
                     't_0','t_1', 't_2',
                     'p-value_0','p-value_1', 'p-value_2',
                     'R2']

    table.columns = table_columns
    display(table)

    print(results.summary())
    print(results.params)
    print(results.pvalues.round(4))

    print(exog.shape, endog.shape)

    # Graphical results
    sns.regplot(x=exog[:,0], y=endog, color='#E18FAD')
    sns.regplot(x=exog[:,1],y=endog, color='#E18FAD')
    plt.title('Diagram of data with the least-squared line')
    plt.show()

    sns.residplot(x=exog[:,0], y=endog, lowess = False)
    sns.residplot(x=exog[:,1], y=endog, lowess = False)
    plt.title('Graphical depiction of residuals')
    plt.show()

    return table

regression3_all = pd.DataFrame()
for i, var in enumerate(ir_data.columns):
    print(' '*30 ,'REGRESSION ON', var)

    x_1 = pd.DataFrame(gpr['GPR_' + var].copy()).values[1:]
    x_2 = pd.DataFrame(ir_data[var].copy()).shift(1).values[1:]

    y = monthly_log_returns[var][1:].values

    x = np.hstack((x_1, x_2))

    regression3_all = regression3_all.append(regression3(y, x))

regression3_all.index = ir_data.columns
regression3_all

# p/s: the graphs take long time to run, so I might disable them