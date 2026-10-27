from pathlib import Path
import sqlite3
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

class ProjectCheckpoints(unittest.TestCase):
    def test_clean_data_checkpoints(self):
        c=pd.read_csv(ROOT/"data/clean/customers.csv")
        o=pd.read_csv(ROOT/"data/clean/orders.csv")
        self.assertEqual(len(c),1200)
        self.assertEqual(c.city.nunique(),6)
        self.assertEqual(len(o),12309)
        self.assertEqual(o.order_id.nunique(),12309)
        self.assertEqual(o.status.value_counts().to_dict(),{"completed":11110,"cancelled":1199})
        self.assertEqual(round(o.loc[o.status=="completed","order_amount"].sum(),2),2998668.80)
        self.assertEqual(o.loc[o.status=="completed","delivery_minutes"].isna().sum(),0)

    def test_database_tables(self):
        with sqlite3.connect(ROOT/"data/instamart.db") as conn:
            tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"customers","products","orders"} <= tables)

    def test_required_outputs(self):
        for name in ["customer_segments.csv","lapse_model.joblib","retention_list.csv",
                     "triage_results.csv","triage_budget_demo.csv","agent_runs.jsonl","refund_ledger.jsonl"]:
            self.assertTrue((ROOT/"outputs"/name).exists(),name)

if __name__=="__main__":
    unittest.main()
