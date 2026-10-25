from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from orbitalsense import FEATURES

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_assets"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 10})

def pre():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

df = pd.read_csv(ROOT / "OrbitalSense_AI_Kaggle_Style_Dataset.csv").drop_duplicates().reset_index(drop=True)
X = df[FEATURES]
y_bw = df["available_bandwidth_mbps"]
y_q = df["link_suitability"]
y_lt = df["remaining_contact_time_s"]
y_c = pd.cut(df["congestion_level_pct"], bins=[-np.inf, 33.33, 66.67, np.inf], labels=["Low", "Medium", "High"])
y_r = df["link_success"]
summary = {"dataset_file": "OrbitalSense_AI_Kaggle_Style_Dataset.csv", "records": int(df.shape[0]), "columns": int(df.shape[1]), "split_ratio": "80% train / 20% held-out test", "random_seed": 42, "stacking_cv": "5-fold StratifiedKFold, shuffle=True, random_state=42", "classification_averages": "macro (zero_division=0)"}
reg_rows=[]; cls_rows=[]; pred_store={}

xtr,xte,ytr,yte=train_test_split(X,y_bw,test_size=.2,random_state=42)
m1=Pipeline([("preprocessing",pre()),("model",RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1))]).fit(xtr,ytr); p1=m1.predict(xte); pred_store['m1']=(yte,p1)
reg_rows.append({"Model":"M1 Random Forest - Bandwidth","MAE":mean_absolute_error(yte,p1),"RMSE":mean_squared_error(yte,p1)**.5,"R2":r2_score(yte,p1)})
xtr,xte,ytr,yte=train_test_split(X,y_q,test_size=.2,random_state=42,stratify=y_q)
m2=Pipeline([("preprocessing",pre()),("model",SVC(kernel="rbf",C=2,class_weight="balanced"))]).fit(xtr,ytr); pq=m2.predict(xte); pred_store['m2']=(yte,pq)
cls_rows.append({"Model":"M2 SVM - Link Quality","Accuracy":accuracy_score(yte,pq),"Precision":precision_score(yte,pq,average="macro",zero_division=0),"Recall":recall_score(yte,pq,average="macro",zero_division=0),"F1-score":f1_score(yte,pq,average="macro",zero_division=0)})
xtr,xte,ytr,yte=train_test_split(X,y_lt,test_size=.2,random_state=42)
m3=Pipeline([("preprocessing",pre()),("model",ExtraTreesRegressor(n_estimators=200,random_state=42,n_jobs=-1))]).fit(xtr,ytr); p3=m3.predict(xte); pred_store['m3']=(yte,p3)
reg_rows.append({"Model":"M3 Extra Trees - Link Lifetime","MAE":mean_absolute_error(yte,p3),"RMSE":mean_squared_error(yte,p3)**.5,"R2":r2_score(yte,p3)})
xtr,xte,ytr,yte=train_test_split(X,y_c,test_size=.2,random_state=42,stratify=y_c)
m4=Pipeline([("preprocessing",pre()),("model",KNeighborsClassifier(n_neighbors=9,weights="distance"))]).fit(xtr,ytr); pc=m4.predict(xte); pred_store['m4']=(yte,pc)
cls_rows.append({"Model":"M4 KNN - Congestion","Accuracy":accuracy_score(yte,pc),"Precision":precision_score(yte,pc,average="macro",zero_division=0),"Recall":recall_score(yte,pc,average="macro",zero_division=0),"F1-score":f1_score(yte,pc,average="macro",zero_division=0)})
xtr,xte,ytr,yte=train_test_split(X,y_r,test_size=.2,random_state=42,stratify=y_r)
stack=StackingClassifier(estimators=[("svm",Pipeline([("preprocessing",pre()),("model",SVC(kernel="rbf",probability=True,class_weight="balanced",random_state=42))])),("knn",Pipeline([("preprocessing",pre()),("model",KNeighborsClassifier(n_neighbors=9,weights="distance"))]))],final_estimator=LogisticRegression(max_iter=2000,class_weight="balanced"),cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42),stack_method="predict_proba",n_jobs=-1).fit(xtr,ytr); pr=stack.predict(xte); pred_store['stack']=(yte,pr)
cls_rows.append({"Model":"Stacking - Route Performance","Accuracy":accuracy_score(yte,pr),"Precision":precision_score(yte,pr,average="macro",zero_division=0),"Recall":recall_score(yte,pr,average="macro",zero_division=0),"F1-score":f1_score(yte,pr,average="macro",zero_division=0)})
reg=pd.DataFrame(reg_rows); cls=pd.DataFrame(cls_rows); reg.to_csv(OUT/'regression_metrics_final.csv',index=False); cls.to_csv(OUT/'classification_metrics_final.csv',index=False)
summary["regression"] = reg.to_dict(orient="records"); summary["classification"] = cls.to_dict(orient="records")
(OUT/'report_values.json').write_text(json.dumps(summary,indent=2,default=float))

# Fig 4.3
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
for axis,key,title,color in [(ax[0],'m1','M1 Bandwidth (Mbps)','#1769aa'),(ax[1],'m3','M3 Link Lifetime (s)','#138a72')]:
    y,p=pred_store[key]; axis.scatter(y,p,s=9,alpha=.55,c=color,edgecolors='none'); lo=min(y.min(),p.min()); hi=max(y.max(),p.max()); axis.plot([lo,hi],[lo,hi],'k--',lw=1); axis.set_title(title); axis.set_xlabel('Actual'); axis.set_ylabel('Predicted'); axis.grid(alpha=.2)
fig.suptitle('Regression Models: Actual vs Predicted Values'); fig.tight_layout(); fig.savefig(OUT/'fig_4_3_regression_actual_vs_predicted.png',bbox_inches='tight'); plt.close(fig)
# Fig 4.4
fig,ax=plt.subplots(1,2,figsize=(9,3.8))
for axis,key,title,labels in [(ax[0],'m2','M2 Link Quality',sorted(y_q.unique())),(ax[1],'m4','M4 Congestion',list(y_c.cat.categories))]:
    y,p=pred_store[key]; cm=confusion_matrix(y,p,labels=labels); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=axis,cbar=False); axis.set_title(title); axis.set_xlabel('Predicted'); axis.set_ylabel('Actual')
fig.suptitle('Classification Confusion Matrices'); fig.tight_layout(); fig.savefig(OUT/'fig_4_4_classification_confusion_matrices.png',bbox_inches='tight'); plt.close(fig)
# Fig 4.5
fig,axis=plt.subplots(figsize=(4.8,4)); y,p=pred_store['stack']; labels=sorted(y.unique()); cm=confusion_matrix(y,p,labels=labels); sns.heatmap(cm,annot=True,fmt='d',cmap='Purples',xticklabels=labels,yticklabels=labels,ax=axis,cbar=False); axis.set_title('Stacked Model: Route Performance'); axis.set_xlabel('Predicted'); axis.set_ylabel('Actual'); fig.tight_layout(); fig.savefig(OUT/'fig_4_5_stacked_confusion_matrix.png',bbox_inches='tight'); plt.close(fig)
# Fig 4.6
fig,axis=plt.subplots(figsize=(8,4.2)); labels=['M2 SVM','M4 KNN','Stacked']; vals_acc=[float(cls.iloc[0]['Accuracy']),float(cls.iloc[1]['Accuracy']),float(cls.iloc[2]['Accuracy'])]; vals_f1=[float(cls.iloc[0]['F1-score']),float(cls.iloc[1]['F1-score']),float(cls.iloc[2]['F1-score'])]; x=np.arange(3); w=.35; axis.bar(x-w/2,vals_acc,w,label='Accuracy',color='#1769aa'); axis.bar(x+w/2,vals_f1,w,label='Macro F1-score',color='#f28e2b'); axis.set_ylim(0,1); axis.set_xticks(x,labels); axis.set_ylabel('Score'); axis.set_title('Stacked Model vs Individual Base Models'); axis.legend(); axis.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(OUT/'fig_4_6_model_comparison.png',bbox_inches='tight'); plt.close(fig)
print(json.dumps(summary,indent=2,default=float))
