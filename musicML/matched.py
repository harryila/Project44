import json, statistics as st
def bykey(p):
    d=json.load(open(p)); out={}
    for r in d["per_performance"]:
        s=r.get("sim")
        if s and (s.get("muster") or {}).get("MeanER") is not None:
            out[r["midi"]]=s["muster"]["MeanER"]
    return out
B=bykey("benchmark/diag/baseline_cuda.json")
A=bykey("benchmark/diag/arm1_last.json")
common=[k for k in A if k in B]
ba=st.mean(B[k] for k in common); aa=st.mean(A[k] for k in common)
print(f"MATCHED on {len(common)} pieces both scored:")
print(f"  baseline MeanER = {ba:.2f}")
print(f"  ARM-1    MeanER = {aa:.2f}")
print(f"  delta = {aa-ba:+.2f}  ({'ARM-1 BETTER' if aa<ba else 'ARM-1 WORSE'})")
diffs=sorted(((A[k]-B[k],k) for k in common), reverse=True)
print("  ARM-1 worse on:")
for d,k in diffs[:3]: print(f"    {d:+5.1f}  {k.split('asap-dataset/')[-1][:42]}")
print("  ARM-1 better on:")
for d,k in diffs[-3:]: print(f"    {d:+5.1f}  {k.split('asap-dataset/')[-1][:42]}")
