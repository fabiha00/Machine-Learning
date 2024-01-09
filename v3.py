# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:10:57 2023

@author: bishal
"""

import pandas as pd
from pandas_profiling import ProfileReport as PR
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import gc

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc, roc_auc_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn.impute import KNNImputer

import joblib

from imblearn.over_sampling import SMOTENC

from catboost import CatBoostClassifier, Pool

import shap

seeds = 11235813
np.random.seed(seeds)

import pywaffle
import warnings
warnings.filterwarnings("ignore")

#Reading the dataset
data=pd.read_csv("healthcare-dataset-stroke-data.csv")
data.head().T

# Show Missing Data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print('======\tMissing Data\t======')
df.isnull().sum()

# Using Decision Tree to fill up missing Data of BMI

DT_bmi_pipe = Pipeline( steps=[ 
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
                              ])
X = df[['age','gender','bmi']].copy()
X.gender = X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8) # Classifying Gender

Missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X,Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]),index=Missing.index)
df.loc[Missing.index,'bmi'] = predicted_bmi
print('Present Missing Values:\t',sum(df.isnull().sum()))


from pywaffle import Waffle
background_color = "#FFFFFF"
fig = plt.figure(figsize=(3, 2),dpi=300,facecolor=background_color,
    FigureClass=Waffle,
    rows=2,
    values=[1, 19],
    colors=['#3a91e6', "#B0E2FF"],
    characters='O',
    font_size=18, vertical=True,
)

fig.text(0.035,0.78,'People Affected by a Stroke in our dataset',fontfamily='Calibri',fontsize=12,fontweight='bold')
fig.text(0.035,0.70,'This is around 1 in 20 people [249 out of 5000]',fontfamily='Calibri',fontsize=6)

plt.show()


#pandas profiling
data.drop(columns=["id"], inplace=True)
profile = PR(
    data, 
    title="Stroke Dataset Report", 
    dark_mode=True, 
    progress_bar=False,
    explorative=True,
    plot={"correlation": {"cmap": "coolwarm", "bad": "#000000"}}
)

profile.to_notebook_iframe()

data.loc[data.gender == "Other"]

data = data.loc[data.gender != "Other"]

#The continuous and categorical variables are designated
contVars = ["age", "avg_glucose_level", "bmi"]
catVars = [i for i in data.columns if i not in contVars and i != "stroke"]

#Target Separated Relationships 2D
def corPlot(df, color: str, title, bins=40):
    sns.set_theme(style="white", font_scale=1.3)

    pp=sns.pairplot(
        df, 
        hue=color, 
        kind="hist", 
        diag_kind="kde",
        corner=True,
        plot_kws={"alpha": 0.9, 'bins':bins},
        diag_kws = {'alpha':0.8, 'bw_adjust': 1, "fill": False, "cut": 0},
        palette="coolwarm",
        aspect=1.1,
        height=3.2
    )

    pp.fig.suptitle(title, fontsize=15)
    plt.show()
    
    contVars.append("stroke")
corPlot(data[contVars], "stroke", "Continuous Variables of the Data Set")

contVars.pop(-1)

#Target Separated Relationships 3D
def plot3d(df, cls: list, c: str, X: str, Y: str, Z: str, title):
    
    """
    Function to plot 3 dimensions, colored by category (c)
    
    df=pd.DataFrame
    cls=colors for no-stroke and stroke
    c=category to separate by color 
    X=X dimension
    Y=Y dimension
    Z=Z dimension
    title=title of the plot
    """
    
    fig = go.Figure()

    for i in range(len(df["%s" % (c)].unique())):
        fig.add_trace(
            go.Scatter3d(
                x=df.loc[df["%s" % (c)] == i, X], 
                y=df.loc[df["%s" % (c)] == i, Y], 
                z=df.loc[df["%s" % (c)] == i, Z],
                mode='markers',
                marker=dict(
                    size=7,
                    color=cls[i],            
                    opacity=0.6
                ), name = i,
                 hovertemplate = 
                    f"<i>{X}</i>: " +"%{x} <br>"+
                    f"<i>{Y}</i>: " +"%{y} <br>"+
                    f"<i>{Z}</i>: " +"%{z}"
            )
        )
        fig.update_layout(
            hoverlabel=dict(font=dict(color='white'))   
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=65, r=20, b=0, t=10),
        width=800,
        height=800,
        scene=dict(
                xaxis_title=X,
                yaxis_title=Y,
                zaxis_title=Z,
                camera={
                    "eye": {"x": 2, "y": 2, "z": 2}
                }
            ),
        title_y=0.95,
        legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.99, title="%s" % (c))
    )

    fig.show()
    
plot3d(data, ["#3a91e6", "red"], "stroke",
       "age", "avg_glucose_level", "bmi", 
       "3D Plot of Age, Average Glucose Level, and BMI"
      )

#Share of Stroke by Category
def barList(df, c, target="stroke"):
    uni=df[f"{c}"].unique()
    brs = []
    for u in uni:
        brs.append(df.loc[data[f"{c}"] == u, target].sum() / df[target].sum())
    return brs, uni

fig, p = plt.subplots(nrows=4, ncols=2, figsize=(10,16))
r=0
c=0
for i in catVars:
    bars, uniq = barList(data, f"{i}", target="stroke")    
    if i in ["hypertension", "heart_disease"]:
        uniq = ["Yes" if j == 1 else "No" for j in uniq]
    
    p[r,c].bar(
        uniq, 
        bars,
        color=["#3a91e6"]*len(bars)
    )
    p[r,c].set_title(f"Proportion of Stroke by {i}")
    p[r,c].set_ylabel("", fontsize=14)
    
    if len(uniq) > 2:
        p[r,c].set_xticklabels(uniq, rotation=25)
    if c == 1:
        c=0
        r+=1
    else:
        c+=1

plt.tight_layout(pad=1)
plt.delaxes(p[3,1])
plt.show()

#Target Separated Dispersion Continous vs. Categorical
def dispersions(cont: list, cat: list, target: str):
    sns.set_palette(sns.color_palette(["#3a91e6", "red"]))
    for co in cont:
        for ct in cat:
            l = (len(data["%s" % (ct)].unique()) - 2) * 2
            pp=sns.catplot(
                data=data, 
                x=ct, 
                y=co, 
                hue=target, 
                kind="box", 
                aspect=eval(f"1.{l}")
            )

            pp.fig.suptitle(
                "{} by {} and {}".format(co, ct, target), 
                fontsize=15
            )
            plt.show()
            
dispersions(contVars, catVars, "stroke")

#Features
 def dummies(df, feats: list, dropFirstAll=False) -> pd.DataFrame:
    """
    This function creates dummies from categorical features of type str.
    df=pd.DataFrame
    feats=list of feature names
    dropFirstAll=bool. If n-1 dummies for all features
    
    output: 
        Data Frame with dummies
    """
    df=df.copy()
    
    for f in feats:
        if len(df[f"{f}"].unique()) > 2:
            dropFirst = False
        else:
            dropFirst = True
            
        if dropFirstAll:
            dropFirst = True
            
        d = pd.get_dummies(df[f"{f}"], prefix=f, drop_first=dropFirst)
        df.drop(columns=[f], inplace=True)
        df[d.columns] = d
    return df

dataD = dummies(
    data, 
    ["gender", "work_type", "ever_married", "Residence_type", "smoking_status"],
    dropFirstAll=True
)
dataD.head(n=3)

#Cross Validation Strategy
test_holdout=0.985

CV_X, test_X, CV_y, test_y = train_test_split(
    dataD.drop(columns=["stroke"]), 
    dataD.stroke,
    train_size=test_holdout,
    random_state=seeds,
    shuffle=True,
    stratify=dataD.stroke
) 

dataD = pd.concat([CV_X, CV_y], axis=1)

X1, X2, y1, y2 = train_test_split(
    dataD.drop(columns=["stroke"]), 
    dataD.stroke,
    train_size=0.5,
    random_state=seeds,
    shuffle=True,
    stratify=dataD.stroke
) 

#folds for the crossvalidation grid search
folds=4

plt.figure(figsize=(3,3))
plt.title("Proportion of Class Stroke", fontsize = 14)
plt.bar(
    ["No Stroke", "Stroke"], 
    [1-y1.sum() / y1.shape[0], y1.sum() / y1.shape[0]],
    color=["#3a91e6", "red"]
)
plt.show()

#To avoid data leakage, the target variable "strok" is omitted when imputing the missing BMI values.
def bmiImputer(XOut, model):
    """
    Function to train imputer on training dataset and apply to another or the same one.
    """
    XOut=XOut.copy()
    
    model.fit(XOut)
    XOut[[i for i in XOut.columns if i != "stroke"]] = model.fit_transform(XOut)
    XOut["bmi"] = np.round(XOut.bmi.values, 1)
    XOut["bmi"] = XOut.bmi
    
    return XOut

imputer = KNNImputer(n_neighbors=10, weights="distance")
 
X1 = bmiImputer(X1,imputer)
X2 = bmiImputer(X2, imputer)
test_X = bmiImputer(test_X, imputer)

bmi = pd.concat([X1.bmi, X2.bmi, test_X.bmi])
data["bmi"] = bmi

#SMOTE
SyntheticShare=1/3
neighbors=2

SMOTE = SMOTENC(
    k_neighbors=neighbors,
    random_state=seeds, 
    n_jobs=-1, 
    categorical_features=np.where([i not in contVars for i in X1.columns])[0],
    sampling_strategy=SyntheticShare
)
Xr1, yr1 = SMOTE.fit_resample(X1, y1)
Xr2, yr2 = SMOTE.fit_resample(X2, y2)

plt.figure(figsize=(3,3))
plt.title("Proportion of Class Stroke after SMOTE", fontsize = 14)
plt.bar(
    ["No Stroke", "Stroke"], 
    [1-yr1.sum() / yr1.shape[0], yr1.sum() / yr1.shape[0]],
    color=["#3a91e6", "red"]
)
plt.show()

contVars.append("stroke")
corPlot(pd.concat([Xr1, yr1.astype(int)], axis=1)[contVars], "stroke", "Minority Class Oversampled")

contVars.pop(-1)

plot3d(
    pd.concat([Xr1, yr1.astype(int)], axis=1), 
    ["#3a91e6", "red"], "stroke",
    "age", "avg_glucose_level", "bmi", 
    "3D Plot of Age, Average Glucose Level, and BMI (SMOTE)"
)

#SVM Modeling
SVM = make_pipeline(
    StandardScaler(),
    SVC(probability=True, random_state=seeds, kernel="rbf")
)

param_grid = {
    'svc__C': [0.1 ,1, 10, 50],
    'svc__gamma': [0.0001, 0.001, 0.01]
}

def CrossVal(model, grid, X: list, y: list, WisorBMI=56.6, **kwargs):
    """
    Function to perform a cross validation wit a grid search.
    We use winsorizing for bmi > 56.6. Default=56.6 is highest BMI stroke case.
    """
    for i in [0,1]:
        val = X[i]
        val["bmi"] = val.bmi.map(lambda x: WisorBMI if x > WisorBMI else x)
        globals()[f"grid{i+1}"] = GridSearchCV(model, grid, n_jobs=-1, cv=folds, scoring="roc_auc")
        eval(f"grid{i+1}").fit(val, y[i], **kwargs)
        
        
        CrossVal(SVM, param_grid, [X1[contVars], X2[contVars]], [y1, y2])
        
        def makeModelAndPred(g: list, m: str, X: list):
    """
    This function uses the best estimator from grid search to predict on validation set.
    """
    for i in [0,1]:
        globals()[f"best_params_model{m}{i+1}"] = g[i].best_params_
        globals()[f"model{m}{i+1}"] = g[i].best_estimator_
        globals()[f"y_hat{i+1}"] = eval(f"model{m}{i+1}").predict(X[i])
        
    return globals()[f"best_params_model{m}1"], globals()[f"best_params_model{m}2"]

makeModelAndPred([grid1, grid2], "SVM", [X2[contVars], X1[contVars]])

#SVM Evaluation without SMOTE
for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
        
    
    def confMat(df, y: list, preds: list, target: str):
    """
    Function to print confusion matrix. 
    The matrix corresponds to a cut-off criterion of 50%. 
    This statement must be relativised for methods such as support vector machines, 
    since a hyperplane is used here and not a probability.
    """
    fig, p = plt.subplots(nrows=1, ncols=2, figsize=(7,8))
    for i in [0,1]:
        mat = confusion_matrix(y[i], preds[i])
        sns.heatmap(
            mat.T, 
            square=True, 
            annot=True, 
            fmt='d', 
            cbar=False,
            xticklabels=np.sort(df[f"{target}"].unique()),
            yticklabels=np.sort(df[f"{target}"].unique()),
            cmap="coolwarm",
            ax=p[i]
        )
        p[i].set_xlabel('true label')
        p[i].set_ylabel('predicted label')
        p[i].set_title(f'Confusion Matrix Fold {i+1}')
    plt.tight_layout(pad=1)
    plt.show()
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    def roc(yt: list, ph: list, title: str, di=False, cut=0.5):
    """
    Function for calculating the receiver operating characteristic curve. 
    In addition, the point that lies at a cut-off criterion of 50% is marked to make comparison easier. 
    
    Output: List of AUCs by fold, list of standard deviation of the prediction errors by fold
    """
    
    fig, p = plt.subplots(nrows=1, ncols=2, figsize=(16,7))
    
    auc=[]
    s=[]
    
    for i in [0,1]:
        fpr, tpr, thresholds = roc_curve(yt[i], ph[i], drop_intermediate=di)
        p[i].plot([0, 1], [0, 1], 'k--')
        p[i].plot(fpr, tpr)
        cutOff = np.where(np.min((thresholds - cut)**2) == (thresholds - cut)**2)[0][0]
        p[i].plot([fpr[cutOff], fpr[cutOff]], [0,1], "darkred")
        p[i].text(fpr[cutOff]+0.01, 0, s=f"cutoff {cut*100}%", c="darkred")
        p[i].set_xlabel("False Positive Rate")
        p[i].set_ylabel("True Positive Rate")
        
        auc.append(roc_auc_score(yt[i], ph[i]))
        p[i].set_title("ROC Curve for {} with AUC of {}% for Fold {}".format(
            title, 
            round(auc[i]*100, 1),
            i+1
            )
        )
        
        s.append(np.std(yt[i] - ph[i]))
    
    print('AUC value for Fold1 = ', auc[0])
    print('AUC value for Fold2 = ', auc[1])
    auc.append(f"{title}")
    s.append(f"{title}")
    plt.show()   
    
    return auc, s

aucSVM, std_SVM = roc(
    [y2, y1], 
    [modelSVM1.predict_proba(X2[contVars])[:, 1], modelSVM2.predict_proba(X1[contVars])[:, 1]], 
    "SVM"
)

#SVM Evaluation using SMOTE
CrossVal(SVM, param_grid, [Xr1[contVars], Xr2[contVars]], [yr1, yr2])

makeModelAndPred([grid1, grid2], "SVM", [X2[contVars], X1[contVars]])

for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    aucSVM, std_SVM = roc(
    [y2, y1], 
    [modelSVM1.predict_proba(X2[contVars])[:, 1], modelSVM2.predict_proba(X1[contVars])[:, 1]], 
    "SVM"
)
    
    def save(m: list, name: str):
    """
    Function to save the learned hypothesis in working directory for implementation purpose.
    """
    for i, M in enumerate(m):
        hyp = f"{name}{i+1}.pkl"
        joblib.dump(M, hyp)
        
        save([modelSVM1, modelSVM2], "svm")
        
        #KNN Modeling and Evaluation without SMOTE
        KNN = make_pipeline(
    StandardScaler(),
    knn()
)

param_grid = {
    'kneighborsclassifier__n_neighbors': [5, 10, 50, 100, 200],
    'kneighborsclassifier__weights': ['uniform', 'distance']
}

CrossVal(KNN, param_grid, [X1[contVars], X2[contVars]], [y1, y2])
makeModelAndPred([grid1, grid2], "KNN", [X2[contVars], X1[contVars]])
for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    
    aucKNN, std_KNN = roc(
    [y2, y1], 
    [modelKNN1.predict_proba(X2[contVars])[:, 1], modelKNN2.predict_proba(X1[contVars])[:, 1]], 
    "KNN"
)
    
    #KNN Evaluation using SMOTE
    CrossVal(KNN, param_grid, [Xr1[contVars], Xr2[contVars]], [yr1, yr2])
    
    makeModelAndPred([grid1, grid2], "KNN", [X2[contVars], X1[contVars]])
    
    for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    aucKNN, std_KNN = roc(
    [y2, y1], 
    [modelKNN1.predict_proba(X2[contVars])[:, 1], modelKNN2.predict_proba(X1[contVars])[:, 1]], 
    "KNN"
)
    
    save([modelKNN1, modelKNN2], "knn")
    
    #Random Forest Modeling and Evaluation without SMOTE
    RF = RandomForestClassifier(random_state=seeds)

param_gridRF = {
    'n_estimators': [150, 180],
    'max_depth': [15, 18, 20],
    'max_features': [0.6, 0.75, 0.9],
    'max_samples': [0.6, 0.75, 0.9],
    'min_samples_leaf': [1, 2],
    'ccp_alpha': [0, 0.0001]#Pruning
}
    CrossVal(RF, param_gridRF, [X1, X2], [y1, y2])
    makeModelAndPred([grid1, grid2], "RF", [X2, X1])
    for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    aucRF, std_RF = roc(
    [y2, y1], 
    [modelRF1.predict_proba(X2)[:, 1], modelRF2.predict_proba(X1)[:, 1]], 
    "RF"
)
    
    
    #Random Forest Evaluation using SMOTE
    CrossVal(RF, param_gridRF, [Xr1, Xr2], [yr1, yr2])
    makeModelAndPred([grid1, grid2], "RF", [X2, X1])
    save([modelRF1, modelRF2], "rf")
    
    for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    aucRF, std_RF = roc(
    [y2, y1], 
    [modelRF1.predict_proba(X2)[:, 1], modelRF2.predict_proba(X1)[:, 1]], 
    "RF"
)
    
    #Naive Bayes Classifier Modeling and Evaluation without SMOTE
    NBC = make_pipeline(
    StandardScaler(),
    GaussianNB()
)

param_grid = {
    'gaussiannb__var_smoothing': [1e-10, 1e-09, 1e-08, 1e-07]
}

CrossVal(NBC, param_grid, [X1[contVars], X2[contVars]], [y1, y2])

makeModelAndPred([grid1, grid2], "NBC", [X2[contVars], X1[contVars]])

for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    aucNBC, std_NBC = roc(
    [y2, y1], 
    [modelNBC1.predict_proba(X2[contVars])[:, 1], modelNBC2.predict_proba(X1[contVars])[:, 1]], 
    "NBC"
)
    
    #Naive Bayes Classifier Evaluation using SMOTE
    
    CrossVal(NBC, param_grid, [Xr1[contVars], Xr2[contVars]], [yr1, yr2])
    makeModelAndPred([grid1, grid2], "NBC", [X2[contVars], X1[contVars]])
    save([modelNBC1, modelNBC2], "nbc")
    
    for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    
    aucNBC, std_NBC = roc(
    [y2, y1], 
    [modelNBC1.predict_proba(X2[contVars])[:, 1], modelNBC2.predict_proba(X1[contVars])[:, 1]], 
    "NBC"
)
    
    
    
    #Logistic Regression Modeling and Evaluation without SMOTE
    Logit = LogisticRegression(random_state=seeds, solver='liblinear')

param_grid = {
    'penalty': ["l1", "l2"],
    'C': np.linspace(0, 500, 26)
}
CrossVal(Logit, param_grid, [X1, X2], [y1, y2])
parmsLogit1, parmsLogit2 = makeModelAndPred([grid1, grid2], "Logit", [X2, X1])
print(parmsLogit1, parmsLogit2)  
for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    aucLogit, std_Logit = roc(
    [y2, y1], 
    [modelLogit1.predict_proba(X2)[:, 1], modelLogit2.predict_proba(X1)[:, 1]], 
    "Logit"
)
  
    
    #Logistic RegressionEvaluation using SMOTE
    
    CrossVal(Logit, param_grid, [Xr1, Xr2], [yr1, yr2])
    parmsLogit1, parmsLogit2 = makeModelAndPred([grid1, grid2], "Logit", [X2, X1])
print(parmsLogit1, parmsLogit2)

save([modelLogit1, modelLogit2], "logit")
for i in [1, 2]:
    print(classification_report(globals()[f"y{1 if i == 2 else 2}"], globals()[f"y_hat{i}"]))
    confMat(data, [y2, y1], [y_hat1, y_hat2], "stroke")
    aucLogit, std_Logit = roc(
    [y2, y1], 
    [modelLogit1.predict_proba(X2)[:, 1], modelLogit2.predict_proba(X1)[:, 1]], 
    "Logit"
)
    
    #Assessment of the Different Algorithms (Error Analysis)
    def errors(x: list, y: list, m: list, titleSupplements: list, var: str):
    
    x=x.copy()
    
    fig, p = plt.subplots(nrows=3, ncols=2, figsize=(10,14))
    
    r=0
    c=0
    for i, M in enumerate(m):
        X=x[i].copy()
        X["preds"] = M.predict_proba(X)[:,1]
        X["err"] = y[i] - X.preds
        
        p[r,c].scatter(x=X[var], y=X["err"], c=y[i].astype("int"), cmap="coolwarm", alpha=0.5)
        p[r,c].set_title("Error of %s for model %s" % (var, titleSupplements[i]))
        p[r,c].set_ylim([-1, 1])
        p[r,c].set_xlabel(var)
        
        if c == 0:
            p[r,c].set_ylabel("Prediction Error")
        
        if c == 1:
            c=0
            r+=1
        else:
            c+=1

    plt.tight_layout(pad=3)
    plt.delaxes(p[2,1])
    plt.show()
    
    
    errors(
    [X2[contVars], X2, X2[contVars], X2, X2[contVars]], [y2, y2, y2, y2, y2], 
    [modelSVM1, modelRF1, modelNBC1, modelLogit1, modelKNN1], 
    [
        "SVM Fold 2",
        "RF Fold 2",
        "NBC Fold 2",
        "Logit Fold 2", 
        "KNN Fold 2"
    ], 
    "age"
)
    
    errors(
    [X1[contVars], X1, X1[contVars], X1, X1[contVars]], [y1, y1, y1, y1, y1], 
    [modelSVM2, modelRF2, modelNBC2, modelLogit2, modelKNN1], 
    [
        "SVM Fold 1",
        "RF Fold 1",
        "NBC Fold 1",
        "Logit Fold 1", 
        "KNN Fold 1"
    ], 
    "bmi"
)
    
    
    testX2 = pd.concat([y2, X2], axis=1)
# We use SVM1 becaus of the good performance
testX2["preds"] = modelRF1.predict_proba(X2)[:,1]
testX2["err"] = np.abs(testX2.stroke - testX2.preds)
errGBR = GBR(loss="absolute_error", max_depth=6, n_estimators=100)
errGBR.fit(testX2.drop(columns=["preds", "err", "stroke"]), testX2["err"])
from sklearn.metrics import mean_absolute_error
print("Mean absolute Error for Random Forest:", round(mean_absolute_error(y1, errGBR.predict(X1)), 4))
#And we save the learned model hypothesis
joblib.dump(errGBR, "errGBR.pkl");


testX2 = pd.concat([y2, X2], axis=1)
# We use SVM1 becaus of the good performance
testX2["preds"] = modelKNN1.predict_proba(X2[contVars])[:,1]
testX2["err"] = np.abs(testX2.stroke - testX2.preds)
errGBR = GBR(loss="absolute_error", max_depth=6, n_estimators=100)
errGBR.fit(testX2.drop(columns=["preds", "err", "stroke"]), testX2["err"])
from sklearn.metrics import mean_absolute_error
print("Mean absolute Error for KNN:", round(mean_absolute_error(y1, errGBR.predict(X1)), 4))
#And we save the learned model hypothesis
joblib.dump(errGBR, "errGBR.pkl");

testX2 = pd.concat([y2, X2], axis=1)
# We use SVM1 becaus of the good performance
testX2["preds"] = modelLogit1.predict_proba(X2)[:,1]
testX2["err"] = np.abs(testX2.stroke - testX2.preds)
errGBR = GBR(loss="absolute_error", max_depth=6, n_estimators=100)
errGBR.fit(testX2.drop(columns=["preds", "err", "stroke"]), testX2["err"])
from sklearn.metrics import mean_absolute_error
print("Mean absolute Error for Logit:", round(mean_absolute_error(y1, errGBR.predict(X1)), 4))
#And we save the learned model hypothesis
joblib.dump(errGBR, "errGBR.pkl");


testX2 = pd.concat([y2, X2], axis=1)
# We use SVM1 becaus of the good performance
testX2["preds"] = modelSVM1.predict_proba(X2[contVars])[:,1]
testX2["err"] = np.abs(testX2.stroke - testX2.preds)
errGBR = GBR(loss="absolute_error", max_depth=6, n_estimators=100)
errGBR.fit(testX2.drop(columns=["preds", "err", "stroke"]), testX2["err"])
from sklearn.metrics import mean_absolute_error
print("Mean absolute Error for SVM:", round(mean_absolute_error(y1, errGBR.predict(X1)), 4))
#And we save the learned model hypothesis
joblib.dump(errGBR, "errGBR.pkl");

testX2 = pd.concat([y2, X2], axis=1)
# We use SVM1 becaus of the good performance
testX2["preds"] = modelNBC1.predict_proba(X2[contVars])[:,1]
testX2["err"] = np.abs(testX2.stroke - testX2.preds)
errGBR = GBR(loss="absolute_error", max_depth=6, n_estimators=100)
errGBR.fit(testX2.drop(columns=["preds", "err", "stroke"]), testX2["err"])
from sklearn.metrics import mean_absolute_error
print("Mean absolute Error for Naive Bayes:", round(mean_absolute_error(y1, errGBR.predict(X1)), 4))
#And we save the learned model hypothesis
joblib.dump(errGBR, "errGBR.pkl");
    

exp=95
for i in [1,2]:
    ssa=aucSVM[i-1]**exp+aucRF[i-1]**exp+aucLogit[i-1]**exp+aucKNN[i-1]**exp+aucNBC[i-1]**exp
    
    svmp = eval(f"modelSVM{i}").predict_proba(eval(f"X{2 if i == 1 else 1}[contVars]"))[:, 1]\
        * aucSVM[i-1]**exp/ssa
    print("Weight Support Vector Machines fold %i" % i, ":", round(aucSVM[i-1]**exp/ssa, 2))
    rfp = eval(f"modelRF{i}").predict_proba(eval(f"X{2 if i == 1 else 1}"))[:, 1]\
        * aucRF[i-1]**exp/ssa
    print("Weight Random Forest fold %i" % i, ":", round(aucRF[i-1]**exp/ssa, 2))  
    
   # knn = eval(f"modelKNN{i}").predict_proba(eval(f"X{2 if i == 1 else 1}"))[:, 1]\
    #    * aucKNN[i-1]**exp/ssa
    #print("Weight KNN fold %i" % i, ":", round(aucKNN[i-1]**exp/ssa, 2))    
    
    logp = eval(f"modelLogit{i}").predict_proba(eval(f"X{2 if i == 1 else 1}"))[:, 1]\
        * aucLogit[i-1]**exp/ssa
    print("Weight Logit fold %i" % i, ":", round(aucLogit[i-1]**exp/ssa, 2))    
   
    nbcp = eval(f"modelNBC{i}").predict_proba(eval(f"X{2 if i == 1 else 1}[contVars]"))[:, 1]\
        * aucNBC[i-1]**exp/ssa
    print("Weight Naive Bayes Classifier fold %i" % i, ":", round(aucNBC[i-1]**exp/ssa, 2))
    print("")    

    #globals()[f"p{i}"] = svmp + rfp + logp + knn + nbcp 
    globals()[f"p{i}"] = svmp + rfp + logp + nbcp 
    
    
    for i, k in zip([1, 2], [0.79, 0.79]):
    print(classification_report(
        globals()[f"y{1 if i == 2 else 2}"], 
        [1 if i >= k else 0 for i in globals()[f"p{i}"]]
    ))
    
    aucEnsemble, std_Ensemble = roc(
    [y2, y1], 
    [
        p1, 
        p2
    ], 
    "Ensemble",
    cut=0.50
)
    
    #Cross Comparison
    
    def tempDf(inp: list, colnames: list, by: str, i: str, j: str, name:str):
    """
    Function to create a dataframe to compare model metrics.
    """
    tmp = pd.DataFrame(
        inp,
        columns=colnames,
    )
    tmp["id"] = tmp.index
    
    tmp = pd.wide_to_long(tmp, by, i=i, j=j)
    tmp.reset_index(inplace=True)
    tmp.rename(columns={by: name, j: by}, inplace=True)
    
    return tmp

    Std_dev = tempDf([
    std_SVM, 
    std_KNN, 
    std_NBC, 
    std_RF, 
    std_Logit, 
    std_Ensemble
    ], ["Fold 1", "Fold 2", "Method"], "Fold ", "id", "partition", "Standard Deviation")
    
    model_AUC = tempDf([
    aucSVM, 
    aucKNN, 
    aucNBC, 
    aucRF, 
    aucLogit, 
    aucEnsemble
    ], ["Fold 1", "Fold 2", "Method"], "Fold ", "id", "partition", "AUC")
    
    plt.figure(figsize=(10,6))
plt.title("AUC of Models for Fold 1 and 2")

p=sns.swarmplot(
    data=model_AUC[["AUC", "Method", "Fold "]], 
    x="Method", 
    y="AUC", 
    hue="Fold ",
    palette=["#3a91e6", "red"]
)
rect=mpatches.Rectangle(
    (-1,aucEnsemble[1]),8,aucEnsemble[0]-aucEnsemble[1], 
    fill = True,
    color = "grey",
    alpha=0.1,
    linewidth = 2
)
plt.gca().add_patch(rect)
p.set_ylabel("AUC")
p.tick_params(axis='x', rotation=45)
plt.show()
    
plt.figure(figsize=(10,6))
plt.title("Standard Deviations of Prediction Errors for Fold 1 and 2 ")

p=sns.swarmplot(
    data=Std_dev[["Standard Deviation", "Method", "Fold "]], 
    x="Method", 
    y="Standard Deviation", 
    hue="Fold ",
    palette=["#3a91e6", "red"]
)
rect=mpatches.Rectangle(
    (-1,std_Ensemble[1]),8,std_Ensemble[0]-std_Ensemble[1], 
    fill = True,
    color = "grey",
    alpha=0.1,
    linewidth = 2
)
plt.gca().add_patch(rect)
p.set_ylabel("Standard Deviation")
p.tick_params(axis='x', rotation=45)
plt.show()
    
    
    #Unseen Test Data
    for i in [1,2]:
    ssa=aucSVM[i-1]**exp+aucRF[i-1]**exp+aucLogit[i-1]**exp+aucNBC[i-1]**exp
    
    svmp = eval(f"modelSVM{i}").predict_proba(test_X[contVars])[:, 1]\
        * aucSVM[i-1]**exp/ssa
    rfp = eval(f"modelRF{i}").predict_proba(test_X)[:, 1]\
        * aucRF[i-1]**exp/ssa
    logp = eval(f"modelLogit{i}").predict_proba(test_X)[:, 1]\
        * aucLogit[i-1]**exp/ssa
   
    nbcp = eval(f"modelNBC{i}").predict_proba(test_X[contVars])[:, 1]\
        * aucNBC[i-1]**exp/ssa

    globals()[f"p{i}"] = svmp + rfp + logp + nbcp 
    
    for i, k in zip([1, 2], [0.79, 0.79]):
    print(classification_report(
        test_y, 
        [1 if i >= k else 0 for i in globals()[f"p{i}"]]
    ))
    
    _, _ = roc(
    [test_y, test_y], 
    [
        p1, 
        p2
    ], 
    "Ensemble",
    cut=0.50
)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    