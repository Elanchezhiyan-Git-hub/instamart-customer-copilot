"""Part B: RFM segmentation and leakage-conscious lapse prediction."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import joblib

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=pd.Timestamp("2025-10-01 00:00:00")
CUTOFF=pd.Timestamp("2025-09-01 00:00:00")

def main():
    out=ROOT/"outputs"; out.mkdir(exist_ok=True)
    orders=pd.read_csv(ROOT/"data/clean/orders.csv",parse_dates=["order_ts"])
    customers=pd.read_csv(ROOT/"data/clean/customers.csv",parse_dates=["signup_date"])
    products=pd.read_csv(ROOT/"data/clean/products.csv")
    completed=orders[orders.status.eq("completed")].copy()
    rfm=completed.groupby("customer_id").agg(last_order_ts=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm["recency"]=(SNAPSHOT-rfm.last_order_ts).dt.days
    rfm=rfm[["customer_id","recency","frequency","monetary"]]
    print("RFM",{"rows":len(rfm),"recency_median":rfm.recency.median(),"min":rfm.recency.min(),"max":rfm.recency.max(),
                 "frequency_mean":rfm.frequency.mean(),"frequency_max":rfm.frequency.max(),"monetary_mean":rfm.monetary.mean(),"monetary_max":rfm.monetary.max()})
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[["recency","frequency","monetary"]])
    rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42)
        labels=km.fit_predict(X)
        rows.append({"k":k,"inertia":km.inertia_,"silhouette":silhouette_score(X,labels)})
    pd.DataFrame(rows).to_csv(out/"kmeans_metrics.csv",index=False)
    print("KMEANS",pd.DataFrame(rows).round(3).to_dict("records"))
    K=4
    km=KMeans(n_clusters=K,n_init=10,random_state=42)
    labels=km.fit_predict(X)
    rfm["segment_id"]=labels
    profiles=rfm.groupby("segment_id").agg(size=("customer_id","size"),mean_recency=("recency","mean"),
        mean_frequency=("frequency","mean"),mean_monetary=("monetary","mean")).reset_index()
    premium_id=int(profiles.loc[profiles.mean_monetary.idxmax(),"segment_id"])
    # Segment names are derived by profiles, not by numeric cluster ordering.
    names={}
    for _,r in profiles.iterrows():
        sid=int(r.segment_id)
        if sid==premium_id: name="Premium"
        elif r.mean_frequency >= profiles.mean_frequency.quantile(.75): name="Frequent Essentials"
        elif r.mean_recency >= 50: name="Dormant Re-engagement"
        elif r.mean_frequency <= profiles.mean_frequency.median(): name="Steady Value"
        else: name="Frequent Essentials"
        names[sid]=name
    rfm["segment_name"]=rfm.segment_id.map(names)
    rfm["is_premium"]=rfm.segment_id.eq(premium_id)
    seg=rfm.merge(customers[["customer_id"]],on="customer_id",how="right")
    # Customers with no completed orders receive an explicit unclustered segment and are not Premium.
    seg["segment_id"]=seg["segment_id"].fillna(-1).astype(int)
    seg["segment_name"]=seg["segment_name"].fillna("No Completed Orders")
    seg["is_premium"]=seg["is_premium"].fillna(False).astype(bool)
    seg.to_csv(out/"customer_segments.csv",index=False)
    print("SEGMENT PROFILES",profiles.round(2).to_dict("records"),"names",names)
    print("PCA",PCA(n_components=2).fit(X).explained_variance_ratio_.tolist())

    # Cohort and labels: cohort members have at least one completed order before cutoff.
    pre=orders[(orders.order_ts<CUTOFF)&(orders.status=="completed")].copy()
    cohort_ids=set(pre.customer_id)
    sept=completed[(completed.order_ts>=CUTOFF)&(completed.order_ts<pd.Timestamp("2025-10-01"))].groupby("customer_id").size()
    ydf=pd.DataFrame({"customer_id":sorted(cohort_ids)})
    ydf["lapsed"]=(~ydf.customer_id.isin(sept.index)).astype(int)
    # All model features use pre-cutoff orders only.
    all_pre=orders[orders.order_ts<CUTOFF].copy()
    pre_comp=all_pre[all_pre.status.eq("completed")].merge(products[["product_id","category"]],on="product_id",how="left")
    f=pre_comp.groupby("customer_id").agg(last_order_ts=("order_ts","max"),frequency=("order_id","nunique"),
        monetary=("order_amount","sum"),avg_discount=("discount_pct","mean"),avg_delivery=("delivery_minutes","mean"),
        favourite_category=("category",lambda x:x.mode().sort_values().iloc[0] if len(x.mode()) else "Unknown")).reset_index()
    f["recency"]=(CUTOFF-f.last_order_ts).dt.days
    # Explicit cancellation-rate calculation avoids aggregation ambiguity.
    status_counts=all_pre[all_pre.status.isin(["completed","cancelled"])].pivot_table(index="customer_id",columns="status",values="order_id",aggfunc="count",fill_value=0)
    for col in ["completed","cancelled"]:
        if col not in status_counts: status_counts[col]=0
    status_counts["cancellation_rate"]=status_counts["cancelled"]/(status_counts["completed"]+status_counts["cancelled"]).replace(0,np.nan)
    f=f.merge(status_counts[["cancellation_rate"]],left_on="customer_id",right_index=True,how="left")
    f=f.merge(customers[["customer_id","city","age","gender","signup_date"]],on="customer_id",how="left")
    f["tenure_days"]=(CUTOFF-f.signup_date).dt.days
    data=ydf.merge(f,on="customer_id",how="left")
    # Cohort requires pre-cutoff completed history, so missing feature values are unexpected but safely imputed.
    data["favourite_category"]=data["favourite_category"].fillna("Unknown")
    num=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","age","tenure_days"]
    cat=["city","gender","favourite_category"]
    for col in num: data[col]=pd.to_numeric(data[col],errors="coerce")
    # Fill any data-quality holes using cohort medians; feature table contains no missing values.
    for col in num: data[col]=data[col].fillna(data[col].median())
    for col in cat: data[col]=data[col].fillna("Unknown")
    assert not data[num+cat].isna().any().any()
    X=data[num+cat]; y=data.lapsed
    X_train,X_test,y_train,y_test,idx_train,idx_test=train_test_split(
        X,y,data.index,test_size=.2,stratify=y,random_state=42)
    print("COHORT",{"rows":len(data),"lapsed":int(y.sum()),"lapsed_pct":round(y.mean()*100,2),
        "train":len(X_train),"test":len(X_test)})
    preprocessor=ColumnTransformer([
        ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),
        ("categorical",OneHotEncoder(drop="first",handle_unknown="ignore"),cat)])
    lr_pipe=Pipeline([("prep",preprocessor),("model",LogisticRegression(max_iter=1000))])
    lr_grid=GridSearchCV(lr_pipe,{"model__C":[.01,.1,1,10,100]},cv=5,scoring="f1",n_jobs=1)
    lr_grid.fit(X_train,y_train)
    rf_pipe=Pipeline([("prep",preprocessor),("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    rf_grid=GridSearchCV(rf_pipe,{"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]},cv=5,scoring="f1",n_jobs=1)
    rf_grid.fit(X_train,y_train)
    # baseline
    baseline=np.zeros(len(y_test),dtype=int)
    print("BASELINE",{"accuracy":accuracy_score(y_test,baseline),"precision":precision_score(y_test,baseline,zero_division=0),
        "recall":recall_score(y_test,baseline,zero_division=0),"f1":f1_score(y_test,baseline,zero_division=0)})
    reports={}
    for name,grid in [("logistic",lr_grid),("random_forest",rf_grid)]:
        pred=grid.predict(X_test)
        reports[name]={"best_params":grid.best_params_,"cv_f1":grid.best_score_,
            "confusion_matrix":confusion_matrix(y_test,pred).tolist(),"precision":precision_score(y_test,pred,zero_division=0),
            "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0)}
    print("MODEL_REPORTS",json.dumps(reports,indent=2))
    # choose RF when its F1 is competitive and better, otherwise logistic.
    selected=rf_grid if reports["random_forest"]["f1"]>=reports["logistic"]["f1"] else lr_grid
    selected_name="random_forest" if selected is rf_grid else "logistic"
    # Class balance: oversample only the training rows; keep test unchanged.
    train=pd.concat([X_train,y_train.rename("lapsed")],axis=1)
    minority=train[train.lapsed==1]
    majority=train[train.lapsed==0]
    balanced_minority=minority.sample(n=len(majority)-len(minority),replace=True,random_state=42)
    balanced=pd.concat([majority,minority,balanced_minority],axis=0).sample(frac=1,random_state=42)
    balanced_lr=clone(lr_grid.best_estimator_)
    balanced_lr.fit(balanced[num+cat],balanced.lapsed)
    orig_pred=lr_grid.predict(X_test); bal_pred=balanced_lr.predict(X_test)
    print("BALANCE",{"original":{"precision":precision_score(y_test,orig_pred,zero_division=0),"recall":recall_score(y_test,orig_pred,zero_division=0),"f1":f1_score(y_test,orig_pred,zero_division=0)},
        "balanced":{"precision":precision_score(y_test,bal_pred,zero_division=0),"recall":recall_score(y_test,bal_pred,zero_division=0),"f1":f1_score(y_test,bal_pred,zero_division=0)},
        "train_counts":train.lapsed.value_counts().to_dict(),"balanced_counts":balanced.lapsed.value_counts().to_dict()})
    # Feature interpretation.
    rf=rf_grid.best_estimator_; names_out=rf.named_steps["prep"].get_feature_names_out()
    importances=rf.named_steps["model"].feature_importances_
    print("RF_IMPORTANCE",sorted(zip(names_out,importances),key=lambda z:z[1],reverse=True)[:5])
    fitted_scaler=rf.named_steps["prep"].named_transformers_["numeric"].named_steps["scaler"]
    print("FREQUENCY_SCALER_MEAN",{"training_mean":float(fitted_scaler.mean_[1]),"full_cohort_mean":float(data.frequency.mean())})
    lr=lr_grid.best_estimator_; coef_names=lr.named_steps["prep"].get_feature_names_out()
    coef=lr.named_steps["model"].coef_[0]
    print("LR_COEFFICIENTS",sorted(zip(coef_names,coef),key=lambda z:abs(z[1]),reverse=True)[:10])
    model_path=out/"lapse_model.joblib"; joblib.dump(selected,model_path)
    reloaded=joblib.load(model_path)
    assert np.array_equal(selected.predict(X_test),reloaded.predict(X_test))
    print("RELOAD_PREDICTIONS_IDENTICAL",True,"selected",selected_name)
    # Top test-set Premium customers by probability; whole-window segment is explicitly post-hoc.
    test_rows=data.loc[X_test.index,["customer_id","lapsed"]].copy()
    premium_ids=set(seg.loc[seg.is_premium,"customer_id"])
    test_rows=test_rows[test_rows.customer_id.isin(premium_ids)]
    probs=selected.predict_proba(X_test.loc[test_rows.index])[:,1] if len(test_rows) else np.array([])
    test_rows["lapse_probability"]=probs
    test_rows=test_rows.sort_values("lapse_probability",ascending=False).head(10)
    test_rows.rename(columns={"lapsed":"actually_lapsed"})[["customer_id","lapse_probability","actually_lapsed"]].to_csv(out/"retention_list.csv",index=False)
    print("RETENTION_TRUE_LAPSED",int(test_rows.lapsed.sum()),"retention_rows",len(test_rows))
if __name__=="__main__": main()
