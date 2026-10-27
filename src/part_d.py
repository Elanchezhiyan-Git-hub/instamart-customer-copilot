"""Part D: retrieval-grounded policy assistant and resumable refund agent."""
from pathlib import Path
import json, re, time, sqlite3, sys
from datetime import datetime, timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mock_llm
POL=ROOT/"data/policies"; OUT=ROOT/"outputs"; STATE=ROOT/"state"
OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
DOCS=["refund_policy.txt","delivery_policy.txt","payment_policy.txt","privacy_policy.txt","escalation_policy.txt"]

class PolicyRetriever:
    def __init__(self, overlap=10):
        self.overlap=overlap
        self.chunks=[]
        for fn in DOCS:
            words=(POL/fn).read_text(encoding="utf-8").split()
            start=0
            while True:
                end=min(start+40,len(words))
                self.chunks.append({"source":fn,"text":" ".join(words[start:end])})
                if end>=len(words): break
                start += 40-overlap
        self.vectorizer=TfidfVectorizer()
        self.matrix=self.vectorizer.fit_transform([c["text"] for c in self.chunks])
    def retrieve(self,query,k=3):
        q=self.vectorizer.transform([query])
        scores=cosine_similarity(q,self.matrix).ravel()
        idx=sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:k]
        return [{"source":self.chunks[i]["source"],"score":float(scores[i]),"text":self.chunks[i]["text"]} for i in idx]

retriever=None
def answer(question, ret=None):
    ret=ret or retriever
    hits=ret.retrieve(question,3)
    context="\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    messages=[{"role":"system","content":"You are a POLICY ASSISTANT. Answer only from the supplied policy context. If the answer is not present, say I could not find that information in the policy documents."},
              {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    response=mock_llm.chat(messages)
    return response.choices[0].message.content,[h["source"] for h in hits]

def evaluate_retrieval(ret, label):
    qa=pd.read_csv(ROOT/"data/raw/policy_eval.csv",keep_default_na=False)
    rows=[]
    for _,r in qa.iterrows():
        hits=ret.retrieve(r.question,3)
        ans,_=answer(r.question,ret)
        expected_doc=r.expected_doc
        rows.append({"question":r.question,"expected_doc":expected_doc,
            "doc_hit":expected_doc=="NONE" or expected_doc in [h["source"] for h in hits],
            "fact_hit":bool(r.expected_fact and r.expected_fact in ans),
            "fallback":ans.startswith("I could not find") if expected_doc=="NONE" else False,
            "expected_fact":r.expected_fact,"answer":ans,"sources":";".join(h["source"] for h in hits)})
    df=pd.DataFrame(rows)
    df.to_csv(OUT/f"policy_eval_{label}.csv",index=False)
    answerable=df[df.expected_doc!="NONE"]; unanswerable=df[df.expected_doc=="NONE"]
    print("POLICY_EVAL",label,{"answerable_doc_hit":int(answerable.doc_hit.sum()),"answerable":len(answerable),
        "fact_hit":int(answerable.fact_hit.sum()),"unanswerable_fallback":int(unanswerable.fallback.sum()),"unanswerable":len(unanswerable)})
    return df

def get_order(order_id):
    with sqlite3.connect(ROOT/"data/instamart.db") as conn:
        row=conn.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",(order_id,)).fetchone()
    if not row: return {"error":"order not found"}
    return {"order_id":row[0],"status":row[1],"order_amount":float(row[2]),"delivery_minutes":None if row[3] is None else int(row[3])}

def search_policy(query):
    return {"results":retriever.retrieve(query,3)}

def calculate_refund(order_amount,delivery_minutes):
    amount=float(order_amount)
    minutes=None if delivery_minutes is None else int(delivery_minutes)
    return {"refund_inr":round(amount,2) if minutes is not None and minutes>45 else 0.0}

def _ledger(tool_name, order_id, refund_inr):
    path=OUT/"refund_ledger.jsonl"
    record={"timestamp":datetime.now(timezone.utc).isoformat(),"tool":tool_name,"order_id":order_id,"refund_inr":round(float(refund_inr),2)}
    with path.open("a",encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
    return {"refund_inr":round(float(refund_inr),2)}

def issue_refund(order_id,refund_inr): return _ledger("issue_refund",order_id,refund_inr)
def escalate_to_human(order_id,refund_inr): return _ledger("escalate_to_human",order_id,refund_inr)

TOOLS_SPEC=[
 {"name":"get_order","description":"Look up an order in the trusted SQLite database.","parameters":{"type":"object","properties":{"order_id":{"type":"string"}},"required":["order_id"]}},
 {"name":"search_policy","description":"Retrieve policy chunks relevant to a query.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}},
 {"name":"calculate_refund","description":"Calculate the full refund only when delivery exceeded 45 minutes.","parameters":{"type":"object","properties":{"order_amount":{"type":"number"},"delivery_minutes":{"type":["integer","null"]}},"required":["order_amount","delivery_minutes"]}},
 {"name":"issue_refund","description":"Record an automatically approved refund of INR 1000 or less.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"refund_inr":{"type":"number"}},"required":["order_id","refund_inr"]}},
 {"name":"escalate_to_human","description":"Record a refund above INR 1000 for human lead approval.","parameters":{"type":"object","properties":{"order_id":{"type":"string"},"refund_inr":{"type":"number"}},"required":["order_id","refund_inr"]}}
]
TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}

class SimulatedCrash(RuntimeError): pass

def _save_checkpoint(run_id, scenario_id, messages, tokens_used, step_number, tools_executed, steps, llm_calls):
    payload={"run_id":run_id,"scenario_id":scenario_id,"messages":messages,"tokens_used":tokens_used,
             "step_number":step_number,"tools_executed":tools_executed,"steps":steps,"llm_calls":llm_calls}
    (STATE/f"{run_id}.json").write_text(json.dumps(payload,default=str,indent=2),encoding="utf-8")

def _status_from_answer(answer_text):
    return answer_text.split(":",1)[0] if ":" in answer_text else "UNKNOWN"

def _run_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    started=time.monotonic()
    for _ in range(max_iterations):
        if time.monotonic()-started >= time_limit_seconds:
            return {"status":"FALLBACK_TIMEOUT","final_answer":"FALLBACK_TIMEOUT: handing off to a human agent; time limit reached.",
                    "steps":state["steps"],"llm_calls":state["llm_calls"],"tokens_used":state["tokens_used"]}
        response=mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"]+=1
        state["tokens_used"]+=response.usage.total_tokens
        if state["tokens_used"]>token_limit:
            return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: handing off to a human agent; token limit reached.",
                    "steps":state["steps"],"llm_calls":state["llm_calls"],"tokens_used":state["tokens_used"]}
        msg=response.choices[0].message
        if not msg.tool_calls:
            final=msg.content or "FALLBACK: empty model response"
            return {"status":_status_from_answer(final),"final_answer":final,"steps":state["steps"],
                    "llm_calls":state["llm_calls"],"tokens_used":state["tokens_used"]}
        calls=[]
        for tc in msg.tool_calls:
            args=tc.arguments
            if isinstance(args,str): args=json.loads(args)
            calls.append({"id":tc.id,"name":tc.name,"arguments":args})
        state["messages"].append({"role":"assistant","content":None,"tool_calls":calls})
        for tc in calls:
            name=tc["name"]
            if name not in TOOL_MAP:
                observation={"error":"unknown tool"}
            else:
                observation=TOOL_MAP[name](**tc["arguments"])
            state["tools_executed"].append(name)
            step={"tool":name,"arguments":tc["arguments"],"observation":observation}
            state["steps"].append(step)
            state["messages"].append({"role":"tool","tool_call_id":tc["id"],"name":name,"content":json.dumps(observation,default=str)})
            _save_checkpoint(state["run_id"],state["scenario_id"],state["messages"],state["tokens_used"],
                             len(state["steps"]),state["tools_executed"],state["steps"],state["llm_calls"])
            if crash_after_tools is not None and len(state["tools_executed"])>=crash_after_tools:
                raise SimulatedCrash(f"Simulated crash after {crash_after_tools} tool executions")
    return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: handing off to a human agent; maximum model-call iterations reached.",
            "steps":state["steps"],"llm_calls":state["llm_calls"],"tokens_used":state["tokens_used"]}

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, run_id=None, crash_after_tools=None):
    run_id=run_id or f"{scenario_id}_run"
    state={"run_id":run_id,"scenario_id":scenario_id,"messages":[
        {"role":"system","content":"You are an AGENT. Use tools to verify order status and policy before acting."},
        {"role":"user","content":user_message}],"tokens_used":0,"step_number":0,"tools_executed":[],"steps":[],"llm_calls":0}
    result=_run_loop(state,max_iterations,token_limit,time_limit_seconds,crash_after_tools)
    result.update({"run_id":run_id,"tools_executed":state["tools_executed"]})
    return result

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    path=STATE/f"{run_id}.json"
    state=json.loads(path.read_text(encoding="utf-8"))
    # Continue with the stored transcript and completed observations; no tool is replayed.
    result=_run_loop(state,max_iterations,token_limit,time_limit_seconds)
    result.update({"run_id":run_id,"tools_executed":state["tools_executed"]})
    return result

def main():
    global retriever
    STATE.mkdir(exist_ok=True)
    for checkpoint in STATE.glob("*.json"):
        checkpoint.unlink()
    retriever=PolicyRetriever(overlap=10)
    print("CHUNKS",len(retriever.chunks),pd.Series([x["source"] for x in retriever.chunks]).value_counts().to_dict())
    print("TOP REFUND RETRIEVAL",[(x["source"],round(x["score"],3)) for x in retriever.retrieve("refund policy late delivery",3)])
    evaluate_retrieval(retriever,"overlap10")
    no_overlap=PolicyRetriever(overlap=0)
    evaluate_retrieval(no_overlap,"overlap0")
    # Reset ledger and run log for a reproducible full reproduction.
    (OUT/"refund_ledger.jsonl").write_text("",encoding="utf-8")
    scenarios=pd.read_csv(ROOT/"data/raw/agent_scenarios.csv",keep_default_na=False)
    results=[]
    with (OUT/"agent_runs.jsonl").open("w",encoding="utf-8") as f:
        for _,s in scenarios.iterrows():
            result=run_agent(s.message,s.scenario_id)
            record={"scenario_id":s.scenario_id,"status":result["status"],"final_answer":result["final_answer"],
                "llm_calls":result["llm_calls"],"tokens_used":result["tokens_used"],"steps":result["steps"]}
            f.write(json.dumps(record)+"\n")
            results.append(record)
    print("AGENT_SCENARIOS",json.dumps([{"scenario_id":r["scenario_id"],"status":r["status"],"llm_calls":r["llm_calls"],"tokens_used":r["tokens_used"],"tools":[s["tool"] for s in r["steps"]]} for r in results],indent=2))
    # Stop-safely demonstrations.
    s1=scenarios.iloc[0]
    token_demo=run_agent(s1.message,"S1_TOKEN_DEMO",token_limit=200)
    timeout_demo=run_agent(s1.message,"S1_TIMEOUT_DEMO",time_limit_seconds=0)
    print("SAFE_DEMOS",token_demo["status"],timeout_demo["status"])
    # Crash after second tool execution; checkpoint is written after observation and before raising.
    crash_id="S1_CRASH_RESUME"
    try: run_agent(s1.message,"S1",run_id=crash_id,crash_after_tools=2)
    except SimulatedCrash: print("SIMULATED_CRASH_AFTER_SECOND_TOOL")
    resumed=resume_agent(crash_id)
    print("CRASH_RESUME",resumed["status"],resumed["final_answer"],resumed["tools_executed"])
if __name__=="__main__": main()
