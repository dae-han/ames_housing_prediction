# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn import metrics

def high_corr_cols(cols, dataframe, n = 0.5, p = False):
    
    """
    Return columns of feature with absolute value of correlation to the target higher than n.
    
    Parameters:
        - cols: list/ Names of the column from df
        - n: float/ Correlation to the target
        - df: dataframe/ Dataframe with the target name 'SalePrice' and its features.
        - p: boolean/ Print correlation table with features with its correaltion to the target higher than n.
    """
    
    corr_df = dataframe.loc[:,cols].corr()[['SalePrice']]
    high_corrs = corr_df.loc[abs(corr_df['SalePrice']) >= n , :]
    if p == True:
        print(high_corrs)

    high_corr_cols = list(high_corrs.index)
    return high_corr_cols

def score_calc(test_df):
    
    """
    Return dataframe of 'cv_score', 'train_score', 'test_score' 
    from the Linear Regression fit of features(X) and target(y) in 'test_df'
    
        - 5 trials are performed with different random_state values 
            and the mean of each scores from the trials is caculated.
    
    - Parameters:
        - test_df: dataframe /Dataframe with the target name 'SalePrice' and its features
    """
    
    rs_list = [42, 112, 23, 121, 48]

    X = test_df.drop('SalePrice', axis = 1)
    y = test_df.loc[:, 'SalePrice']
    
    emp_dict = {'cv_score': 0, 'train_score': 0, 'test_score':0, 'num_features': 0}
    out_df = pd.DataFrame(data= list(emp_dict.items()), columns = ['score','to_drop'])
    
    for rs in rs_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rs)

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        score_dict = {'cv_score': cross_val_score(lr, X_train, y_train, cv = 5).mean(), 
                      'train_score': lr.score(X_train, y_train),
                      'test_score' :lr.score(X_test, y_test),
                      'num_features': 0}

        score_df = pd.DataFrame(data = list(score_dict.items()), columns = ['score', f'rs={rs}'])
        
        out_df = pd.merge(out_df, score_df, on = 'score')
    
    out_df = out_df.drop('to_drop', axis = 1)
    out_df['mean'] = out_df.mean(axis=1)
    
    return out_df

def opt_n(cols, dataframe , corr_start = 0.1, corr_end = 0.7, corr_step = 0.025):
    
    """
    Return the mean of 'cv_score', 'train_score', 'test_score', 'num_features'
        from 5 trials with different random state values 
        in the Linear Regression fit of features(X) and target(y) in 'test_df' 
        with different correlation threshold in the range of 'corr_start' to 'corr_end'.
    
    Parameters:
        - cols: list/ Name of columns of feature and target in a dataframe 'df'
        - df: dataframe/ Dataframe with features and target (default: df)
        - corr_start: float/ correlation minimum
        - corr_end: float/ correlation max
        - corr_step: flaot/ correlation step size
    
    """
    
    emp_dict = {'cv_score': 0, 'train_score': 0, 'test_score':0, 'num_features': 0}
    out_df = pd.DataFrame(data= list(emp_dict.items()), columns = ['score','to_drop'])
    
    corrs = np.arange(corr_start, corr_end, corr_step)
    for corr in corrs:
        test_df = dataframe[high_corr_cols(cols = cols, dataframe = dataframe, n = corr)].dropna()
        score_df = score_calc(test_df)
        score_df.loc[score_df['score'] == 'num_features', 'mean'] = \
        int(len(high_corr_cols(cols = cols, dataframe = dataframe, n = corr)))
        score_df = score_df.rename({'mean': f'{round(corr,3)}'}, axis=1)
        
        out_df = pd.merge(out_df, score_df.iloc[:,[0,6]], on = 'score')

    return out_df.drop('to_drop', axis=1)


def corr_r2_plot(dataframe, cols ,title_name ,figsize = (18,6), corr_start = 0.1, corr_end = 0.7, corr_step = 0.025):
    
    """
    Create a plot that shows how R2 scores change with features with varying correaltions to the target.
    
    Parameters:
        - df: Dataframe with features and target
        - cols: list/ Names of columns for features
        - title_name: str/ Name of the title
        - figsize: tuple/ (width, height) size the the figure 
        - corr_start: float/ correlation minimum
        - corr_end: float/ correlation max
        - corr_step: flaot/ correlation step size
        
        #Co rrelation Threshold*: absolute value of minimum correlation to the target the included features have.
    """
    
    df_corr = opt_n(cols, dataframe, corr_start = corr_start, corr_end = corr_end, corr_step = corr_step)
    
    plt.figure(figsize=figsize)
    plt.plot(df_corr.iloc[0,1:].values, label = 'CV Score')
    plt.plot(df_corr.iloc[1,1:].values, label = 'Train Score')
    plt.plot(df_corr.iloc[2,1:].values, label = 'Test Score')
    plt.xticks(ticks = np.arange(len(list(df_corr.columns[1:]))),
               labels = list(df_corr.columns[1:]), 
               fontsize = 13)
    plt.yticks(fontsize = 15)
    
    # Correlation Threshold*: absolute value of minimum correlation to the target the included features have.
    plt.xlabel('Correlation Threshold (|x|<)', fontsize = 20)
    plt.ylabel('Score ($R^2$)', fontsize = 20)
    plt.title(title_name, fontsize = 27)
    plt.legend(fontsize=15)
    plt.tight_layout() 
    
def get_coef_corr(dataframe, cols,absolute = False):
    
    """
    Return dataframe with coefficient weight and correlation for all features
    
    Parameters:
        - df: dataframe/ dataframe with features and target (='SalePrice')
        - absolute: boolean/ returns absolute value of coefficient and correlation
    """
    
    test_df = dataframe[cols].dropna()
    X = test_df.drop('SalePrice', axis = 1)
    y = test_df['SalePrice']


    lr = LinearRegression()
    lr.fit(X,y)
    
    if absolute == False:
        out_df = pd.DataFrame( {'Feature':list(X.columns),
                       'Coef': list(lr.coef_), 
                       'Corr': list(test_df.corr().drop('SalePrice')['SalePrice'].values)})
    
    # get absolute coeffient and correlation
    if absolute == True:
        out_df = pd.DataFrame( {'Feature': list(X.columns),
                       'Coef': list(abs(lr.coef_)), 
                       'Corr': list(abs(test_df.corr().drop('SalePrice')['SalePrice'].values))})     
    
    
    return out_df.sort_values(by = 'Coef').reset_index(drop=True)

def coef_corr_plot(dataframe, cols, absolute = False):
    
    """
    Plot coefficient weight and correlation of features to the target
    
    Parameters:
        - df: Dataframe with features and target
        - cols: list/ Names of columns for features
        - absolute: boolean/ returns absolute value of coefficient and correlation        
    """
    
    df_test = get_coef_corr(dataframe, cols, absolute = absolute)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex = True) 
    ax = ax.ravel() # Ravel turns a matrix into a vector, which is easier to iterate
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.tight_layout()

    # Coefficient plot
    ax[0].bar(list(df_test.index), np.log(df_test.loc[:,'Coef']))
    ax[0].set_ylabel('Log Scaled Coeff. Weight', fontsize = 23)
    ax[0].tick_params(labelsize = 18)
    
    if absolute:
        fig.suptitle('Absolute Coeff. Weight and Corr. to the target', fontsize=30)
    else:
        fig.suptitle('Coeff. Weight and Corr. to the target', fontsize=30)
    fig.subplots_adjust(top=0.93) 
    
    # Correlation plot
    ax[1].bar(list(df_test.index),df_test.loc[:,'Corr'])
    ax[1].set_ylabel('Correlation to SalePrice', fontsize = 23)
    ax[1].set_xlabel('Features (Index)', fontsize = 23)
    ax[1].set_xticks(ticks=np.arange(len(list(df_test['Feature']))))
    ax[1].set_xticklabels(labels=list(df_test.index), fontsize = 15)
    ax[1].tick_params(labelsize = 18)
    
def r2_adj(X, y, y_hat):
    
    """
    Calculate r2_adj()
    
    As you add more variables, R2 will never decrease (with linear regression). 
    Adjusted R2 can handle this assumption better.
    
    -equation source:
    https://www.dummies.com/education/math/business-statistics/how-to-calculate-the-adjusted-coefficient-of-determination/"""
    
    n = len(y) #sample size
    k = len(X.columns) #the number of independent variables in the regression equation
    r2 = metrics.r2_score(y, y_hat)
    r2_adj = 1 - (1 - r2)* ((n -1) / (n - (k+1)))
    
    return r2_adj

def six_reg_metrics(X, y, y_hat):
    
    """
    Calculate 6 regression metrics (MAE', 'SSE', 'MSE', 'RMSE', 'R2', 'Adjusted_R2')
    """
    
    resids = y - y_hat
    
    mae= np.mean(abs(resids))
    sse= sum(resids**2)
    mse= sse/len(resids)
    rmse= np.sqrt(mse)
    
    ssto = np.sum((y - np.mean(y))**2)
    r2 = 1 - (sse / ssto)
    adj_r2 = r2_adj(X, y, y_hat)
    
    val_name = ['MAE', 'SSE', 'MSE', 'RMSE', 'R2', 'Adjusted_R2']
    val_list = [mae, sse, mse, rmse, r2, adj_r2] 
    
    return dict(zip(val_name, val_list))

def resid_plot(preds, resids, title=None):
    
    """
    Plot residual plot
    """
    
    plt.figure(figsize = (10,5))
    plt.scatter(preds, resids, label = 'actual prediction')
    plt.axhline(0, color = 'red', label = 'accurate prediction')
    plt.xlabel('Predction', fontsize = 15)
    plt.ylabel('Residual (error)', fontsize = 15)
    plt.legend(fontsize = 15)
    plt.title(title, fontsize = 20)

def accuracy_plot(y, preds, title=None):
    
    """
    Plot true value vs predicted value to measure the accuracy of the prediction
    """
    
    plt.figure(figsize = (10,5))
    plt.scatter(y, preds, label = 'actual prediction')
    plt.plot(y, y, color='red', label = 'accurate prediction')
    plt.xlabel('True value', fontsize = 15)
    plt.ylabel('Predicted value', fontsize = 15)
    plt.legend(fontsize = 15)
    plt.title(title, fontsize = 20)
    
def model_fit(X, y, model = LinearRegression()):
    
    """
    For models that do not require scaling,
    Plot risidual plot and accuracy plot (actual values vs predicted values)
        three times with different random_state values.
    Print six metrics (MAE', 'SSE', 'MSE', 'RMSE', 'R2', 'Adjusted_R2') for each trial.

    """
    
    rs_list = [42, 112, 23]
    
    for i, rs in enumerate(rs_list):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rs)

        lr = model
        lr.fit(X_train, y_train)
        
        preds = lr.predict(X_test)
        resids = y_test - preds
        
        title = f"Trial {i+1}"
        print(title)
        
        resid_plot(preds, resids, title = title + "/ Residual plot" )
        accuracy_plot(y_test, preds, title = title + "/ Accuracy plot")
        print(six_reg_metrics(X_test, y_test, preds))

def model_fit_sc(X, y, model):
    
    """
    For models that required scaling,
    Plot risidual plot and accuracy plot (actual values vs predicted values)
        three times with different random_state values.
    Print six metrics (MAE', 'SSE', 'MSE', 'RMSE', 'R2', 'Adjusted_R2') for each trial.

    """
    
    rs_list = [42, 112, 23]
    
    for i, rs in enumerate(rs_list):
        ss = StandardScaler()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rs)
        
        X_train_sc = ss.fit_transform(X_train)
        X_test_sc = ss.transform(X_test)

        lr = model
        lr.fit(X_train_sc, y_train)
        
        preds = lr.predict(X_test_sc)
        resids = y_test - preds
        
        title = f"Trial {i+1}"
        
        resid_plot(preds, resids, title = title + "/ Residual plot" )
        accuracy_plot(y_test, preds, title = title + "/ Accuracy plot")
        
        print(title)
        print(f"Train Score: {lr.score(X_train_sc, y_train)}")
        print(f"Test Score: {lr.score(X_test_sc, y_test)}")
        print(f"RMSE: {np.sqrt(metrics.mean_squared_error(y_test, preds))}")