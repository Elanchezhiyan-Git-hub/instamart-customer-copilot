'''Part C: privacy-safe, resilient support ticket triage.'''
from pathlib import Path
import re, json, random, time
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mock_llm
MAX_INPUT_TOKENS=400
BASE_DELAY=.05
ALLOWED={"billing","delivery","technical","fraud"}
PATTERNS=[
    ("email",re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),"[EMAIL]"),
    ("phone",re.compile(r"\b[6-9]\d{9}\b"),"[PHONE]"),
    ("order_id",re.compile(r"ORD-\d{6}"),"[ORDER_ID]")
]
ZERO_PROMPT='''You are a CLASSIFY assistant. Classify the support ticket into exactly one allowed category: billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}; do not add markdown or other text.'''
FEW_PROMPT='''You are a CLASSIFY assistant. Allowed categories are billing, delivery, technical, fraud. Reply with JSON only in the form {"category": "...", "reason": "..."}.
EXAMPLES:
Ticket: [EMAIL] I was charged twice for order [ORDER_ID]. Please refund the extra payment.
Reply: {"category":"billing","reason":"duplicate charge"}
Ticket: There is an unauthorized transaction on my card for [ORDER_ID]. Call [PHONE].
Reply: {"category":"fraud","reason":"unauthorized transaction"}
Ticket: The app crashes and I cannot login.
Reply: {"category":"technical","reason":"app error"}'''

def redact(text):
    counts={}
    value=text
    for name,pat,placeholder in PATTERNS:
        value,n=pat.subn(placeholder,value); counts[name]=n
    return value,counts

def remaining_pii(text):
    return any(p.search(text) for _,p,_ in PATTERNS)

def classify_one(ticket_id, text, system_prompt, running_tokens=0, token_budget=None):
    if not text or not text.strip():
        return {"status":"skipped_empty","category":"","reason":"","tokens_used":0,"redacted_text":""}
    redacted,counts=redact(text)
    if mock_llm.est_tokens(text)>MAX_INPUT_TOKENS:
        return {"status":"flagged_long","category":"","reason":"","tokens_used":0,"redacted_text":redacted,"redactions":counts}
    assert not remaining_pii(redacted), f"PII remained in {ticket_id}"
    messages=[{"role":"system","content":system_prompt},{"role":"user","content":redacted}]
    used=0; retries=0
    while True:
        if token_budget is not None and running_tokens+used >= token_budget:
            return {"status":"budget_exhausted","category":"","reason":"","tokens_used":used,"redacted_text":redacted,"redactions":counts}
        try:
            response=mock_llm.chat(messages)
            used += response.usage.total_tokens
            answer=response.choices[0].message.content or ""
            try:
                parsed=json.loads(answer)
                category=parsed.get("category")
                if category in ALLOWED:
                    status="classified_json"; reason=str(parsed.get("reason",""))
                else:
                    return {"status":"flagged_invalid_output","category":"","reason":"","tokens_used":used,"redacted_text":redacted,"redactions":counts,"redactions":counts}
            except (json.JSONDecodeError, TypeError):
                words=re.findall(r"[a-z]+",answer.lower())
                found=[w for w in words if w in ALLOWED]
                if len(set(found))==1:
                    category=found[0]; status="classified_keyword"; reason="category extracted from non-JSON reply"
                else:
                    return {"status":"flagged_invalid_output","category":"","reason":"","tokens_used":used,"redacted_text":redacted,"redactions":counts}
            return {"status":status,"category":category,"reason":reason,"tokens_used":used,"redacted_text":redacted,"redactions":counts}
        except mock_llm.TransientLLMError:
            if retries>=3:
                return {"status":"permanent_error","category":"","reason":"","tokens_used":used,"redacted_text":redacted,"redactions":counts,"redactions":counts}
            delay=BASE_DELAY*(2**retries)*random.uniform(.9,1.15)
            print(f"RETRY ticket={ticket_id} retry={retries} wait={delay:.5f}s")
            time.sleep(delay); retries+=1

def run_pipeline(tickets, prompt, budget=None, reset=True):
    if reset: mock_llm.reset()
    random.seed(42)
    rows=[]; token_total=0; total_redactions={"email":0,"phone":0,"order_id":0}
    for _,t in tickets.iterrows():
        result=classify_one(t.ticket_id,t.text,prompt,token_total,budget)
        token_total += result["tokens_used"]
        for k,v in result.get("redactions",{}).items(): total_redactions[k]+=v
        rows.append({"ticket_id":t.ticket_id,"customer_id":t.customer_id,"status":result["status"],
            "category":result["category"],"reason":result["reason"],"tokens_used":result["tokens_used"],
            "redacted_text":result["redacted_text"],"true_category":t.true_category})
    df=pd.DataFrame(rows)
    eligible=df[df.status.isin(["classified_json","classified_keyword"])]
    accuracy=float((eligible.category==eligible.true_category).mean()) if len(eligible) else 0.0
    summary={"rows":len(df),"status_counts":df.status.value_counts().to_dict(),"classified":len(eligible),
        "accuracy":accuracy,"total_tokens":int(token_total),"tokens_per_classified":token_total/len(eligible) if len(eligible) else 0,
        "redactions":total_redactions,"redactions_total":sum(total_redactions.values())}
    return df,summary

def main():
    out=ROOT/"outputs"; out.mkdir(exist_ok=True)
    tickets=pd.read_csv(ROOT/"data/raw/tickets.csv",keep_default_na=False)
    zero,zsum=run_pipeline(tickets,ZERO_PROMPT)
    print("ZERO_SHOT",json.dumps(zsum,indent=2))
    few,fsum=run_pipeline(tickets,FEW_PROMPT)
    print("FEW_SHOT",json.dumps(fsum,indent=2))
    eligible=zero[zero.status.isin(["classified_json","classified_keyword"])]
    confusion=pd.crosstab(eligible.true_category,eligible.category,dropna=False)
    confusion.to_csv(out/"triage_zero_shot_confusion.csv")
    print("ZERO_SHOT_CONFUSION\n",confusion)
    assert not any(remaining_pii(x) for x in pd.concat([zero.redacted_text,few.redacted_text]).tolist())
    budget_demo,bsummary=run_pipeline(tickets,ZERO_PROMPT,budget=6000)
    if not (budget_demo.status=="budget_exhausted").any():
        budget_demo,bsummary=run_pipeline(tickets,ZERO_PROMPT,budget=5000)
    budget_demo[["ticket_id","customer_id","status","category","tokens_used"]].to_csv(out/"triage_budget_demo.csv",index=False)
    print("BUDGET_DEMO",bsummary)
    final,finalsum=run_pipeline(tickets,FEW_PROMPT,budget=100000)
    seg=pd.read_csv(out/"customer_segments.csv")
    segmap=seg.set_index("customer_id")[["segment_name","is_premium"]].to_dict("index")
    routed=[]
    for _,r in final.iterrows():
        s=segmap.get(r.customer_id,{"segment_name":"","is_premium":False})
        if r.status in {"skipped_empty","flagged_long","flagged_invalid_output","budget_exhausted","permanent_error"}:
            route,priority="human_review","NORMAL"
        elif r.category=="fraud":
            route,priority="security_team","HIGH"
        elif bool(s["is_premium"]):
            route,priority="premium_desk","HIGH"
        else:
            route,priority="support_queue","NORMAL"
        routed.append({"ticket_id":r.ticket_id,"customer_id":r.customer_id,"status":r.status,
            "category":r.category,"segment_name":s["segment_name"],"priority":priority,"route":route,"tokens_used":r.tokens_used})
    pd.DataFrame(routed,columns=["ticket_id","customer_id","status","category","segment_name","priority","route","tokens_used"]).to_csv(out/"triage_results.csv",index=False)
    print("FINAL_FEW_SHOT",finalsum)
if __name__=="__main__": main()
