"""Part A: audit, clean, persist, validate, and query synthetic source data."""
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW, CLEAN = ROOT / "data/raw", ROOT / "data/clean"
DB = ROOT / "data/instamart.db"

def audit():
    orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)
    customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
    rows = []
    for name, df in [("orders", orders), ("customers", customers), ("products", products)]:
        for col in df.columns:
            missing = df[col].astype(str).str.strip().eq("").sum()
            rows.append({"table":name,"shape":str(df.shape),"column":col,"dtype":str(df[col].dtype),
                         "missing_count":int(missing),"missing_pct":round(100*missing/len(df),2)})
    pd.DataFrame(rows).to_csv(ROOT/"outputs/raw_audit_columns.csv", index=False)
    print("RAW AUDIT")
    print(pd.DataFrame(rows).to_string(index=False))
    order_num = pd.to_numeric(orders.order_amount, errors="coerce")
    city = customers.city.astype(str)
    gender = customers.gender.astype(str)
    print("AUDIT CHECKPOINTS", {
        "raw_order_rows":len(orders), "exact_duplicate_rows":int(orders.duplicated().sum()),
        "pending_rows":int((orders.status=="pending").sum()), "distinct_order_id":int(orders.order_id.nunique()),
        "nonnumeric_amounts":int(order_num.isna().sum()), "negative_delivery":int((pd.to_numeric(orders.delivery_minutes,errors="coerce")<0).sum()),
        "blank_delivery":int(orders.delivery_minutes.astype(str).str.strip().eq("").sum()),
        "blank_age":int(customers.age.astype(str).str.strip().eq("").sum()),
        "age_outside_10_100":int((pd.to_numeric(customers.age,errors="coerce").dropna().pipe(lambda x:(x<10)|(x>100))).sum()),
        "raw_city_spellings":int(city.nunique()), "raw_nonblank_gender_spellings":int(gender[gender.str.strip()!=""].nunique()),
        "blank_gender":int(gender.str.strip().eq("").sum())
    })

def clean_customers(customers, products):
    c = customers.copy()
    c["city"] = c["city"].astype(str).str.strip().str.title()
    c["gender"] = c["gender"].astype(str).str.strip().str.lower().map(
        {"male":"male","m":"male","female":"female","f":"female"}).fillna("unknown")
    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    c.loc[~c["age"].between(10,100), "age"] = np.nan
    c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
    c["age"] = c["age"].round().astype(int)
    c["signup_date"] = pd.to_datetime(c["signup_date"])
    return c

def clean_orders(orders, products):
    o = orders.copy()
    o["order_ts"] = pd.to_datetime(o["order_ts"], errors="coerce")
    exact_before=len(o)
    o=o.drop_duplicates()
    exact_removed=exact_before-len(o)
    before_status=o.status.value_counts().to_dict()
    # Latest timestamp per order_id wins; stable sorting makes ties deterministic.
    o=o.sort_values(["order_id","order_ts"], kind="mergesort").drop_duplicates("order_id", keep="last")
    o["quantity"]=pd.to_numeric(o["quantity"],errors="coerce")
    o["discount_pct"]=pd.to_numeric(o["discount_pct"],errors="coerce")
    o["order_amount"]=pd.to_numeric(o["order_amount"],errors="coerce")
    o["delivery_minutes"]=pd.to_numeric(o["delivery_minutes"],errors="coerce")
    o.loc[o["delivery_minutes"]<0,"delivery_minutes"]=np.nan
    o=o.merge(products[["product_id","price_inr","category"]],on="product_id",how="left",validate="many_to_one")
    missing_amount=o["order_amount"].isna()
    o.loc[missing_amount,"order_amount"]=(o.loc[missing_amount,"price_inr"]*o.loc[missing_amount,"quantity"]*(1-o.loc[missing_amount,"discount_pct"]/100)).round(2)
    completed=o["status"].eq("completed")
    # Transform computes each completed row's product-category median without changing row alignment.
    completed_rows=o.loc[completed,["category","delivery_minutes"]].copy()
    completed_rows["category_median"]=completed_rows.groupby("category")["delivery_minutes"].transform("median")
    miss=completed & o["delivery_minutes"].isna()
    missing_idx=o.index[miss]
    o.loc[missing_idx,"delivery_minutes"]=completed_rows.loc[missing_idx,"category_median"]
    o.loc[o["status"].eq("cancelled"),"delivery_minutes"]=np.nan
    o=o.drop(columns=["price_inr","category"])
    o=o.sort_values("order_id").reset_index(drop=True)
    return o, {"exact_duplicates_removed":exact_removed, "amounts_imputed":int(missing_amount.sum()),
               "negative_or_missing_completed_delivery_filled":int(miss.sum())}

def main():
    RAW.mkdir(parents=True,exist_ok=True); CLEAN.mkdir(parents=True,exist_ok=True)
    audit()
    customers=pd.read_csv(RAW/"customers.csv",keep_default_na=False)
    products=pd.read_csv(RAW/"products.csv",keep_default_na=False)
    orders=pd.read_csv(RAW/"orders.csv",keep_default_na=False)
    c=clean_customers(customers,products)
    p=products.copy()
    p["price_inr"]=pd.to_numeric(p["price_inr"])
    o,changes=clean_orders(orders,p)
    c.to_csv(CLEAN/"customers.csv",index=False,date_format="%Y-%m-%d")
    p.to_csv(CLEAN/"products.csv",index=False)
    o.to_csv(CLEAN/"orders.csv",index=False,date_format="%Y-%m-%d %H:%M:%S")
    DB.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(DB) as conn:
        c.to_sql("customers",conn,if_exists="replace",index=False)
        p.to_sql("products",conn,if_exists="replace",index=False)
        o.to_sql("orders",conn,if_exists="replace",index=False)
    print("CLEANING CHANGES",changes)
    print("CLEAN CHECKPOINTS",{
        "customers":len(c),"cities":c.city.value_counts().to_dict(),"genders":c.gender.value_counts().to_dict(),
        "missing_ages":int(c.age.isna().sum()),"orders":len(o),"status":o.status.value_counts().to_dict(),
        "missing_amount":int(o.order_amount.isna().sum()),
        "completed_missing_delivery":int(o.loc[o.status=="completed","delivery_minutes"].isna().sum()),
        "cancelled_missing_delivery":int(o.loc[o.status=="cancelled","delivery_minutes"].isna().sum()),
        "completed_revenue":round(o.loc[o.status=="completed","order_amount"].sum(),2)})
    validate()
    run_sql_queries()

def validate():
    c=pd.read_csv(CLEAN/"customers.csv")
    o=pd.read_csv(CLEAN/"orders.csv")
    checks = [
        len(c)==1200, c.city.nunique()==6, c.gender.value_counts().to_dict()=={"female":610,"male":560,"unknown":30},
        c.age.isna().sum()==0, c.age.between(18,60).all(), len(o)==12309,
        o.order_id.nunique()==12309, (o.status=="pending").sum()==0,
        int((o.status=="completed").sum())==11110, int((o.status=="cancelled").sum())==1199,
        o.order_amount.isna().sum()==0, o.loc[o.status=="completed","delivery_minutes"].isna().sum()==0,
        o.loc[o.status=="cancelled","delivery_minutes"].isna().sum()==1199,
        round(o.loc[o.status=="completed","order_amount"].sum(),2)==2998668.80
    ]
    if not all(checks):
        raise AssertionError(f"Checkpoint failure: {checks}")
    print("ALL CHECKPOINTS PASSED")

def run_sql_queries():
    with sqlite3.connect(DB) as conn:
        qs=(ROOT/"sql/queries.sql").read_text(encoding="utf-8").split("-- QUERY ")
        results=[]
        for block in qs:
            block=block.strip()
            if not block: continue
            label,sql=block.split("\n",1)
            df=pd.read_sql_query(sql,conn)
            results.append((label,df))
            print(f"\n{label}: {len(df)} rows")
            print(df.head(10).to_string(index=False))
        # pandas cross-check
        o=pd.read_sql_query("SELECT * FROM orders WHERE status='completed'",conn)
        c=pd.read_sql_query("SELECT customer_id, city FROM customers",conn)
        p=pd.read_sql_query("SELECT product_id, category FROM products",conn)
        joined=o.merge(c,on="customer_id").merge(p,on="product_id")
        pivot=pd.pivot_table(joined,index="city",columns="category",values="order_amount",aggfunc="sum",fill_value=0,margins=True)
        pivot.to_csv(ROOT/"outputs/pandas_city_category_pivot.csv")
        city=o.merge(c,on="customer_id").groupby("city").order_amount.agg(total="sum",mean="mean",count="count")
        city.to_csv(ROOT/"outputs/city_spend_summary.csv")
        x=o.order_amount
        q1,q3=x.quantile([.25,.75]); iqr=q3-q1
        corr=o[["order_amount","discount_pct","quantity","delivery_minutes"]].corr()
        corr.to_csv(ROOT/"outputs/completed_order_correlations.csv")
        print("PANDAS PIVOT GRAND TOTAL",pivot.loc["All","All"])
        print("CITY SPEND",city.to_dict("index"))
        print("AMOUNT STATS",{"mean":x.mean(),"median":x.median(),"skew":x.skew(),"q1":q1,"q3":q3,
              "lower_fence":q1-1.5*iqr,"upper_fence":q3+1.5*iqr,
              "below":int((x<q1-1.5*iqr).sum()),"above":int((x>q3+1.5*iqr).sum()),
              "max_abs_corr_pair":max(((abs(corr.loc[a,b]),a,b,corr.loc[a,b]) for i,a in enumerate(corr.columns) for b in corr.columns[i+1:]))})
if __name__=="__main__": main()
