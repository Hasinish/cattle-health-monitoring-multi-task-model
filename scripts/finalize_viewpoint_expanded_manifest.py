"""
finalize_viewpoint_expanded_manifest.py

Applies the user's adjudication of the 21 agent vs. ChatGPT disagreements to
artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv.
"""

import os
import pandas as pd

MANIFEST_PATH = "artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv"

# 21 Disagreements and their resolution from docs/audits/phase3_viewpoint_mismatch_user_review.md
# User reviewed and checked:
# 20 cases: "[X] ChatGPT is correct"
# 1 case (vp2_0060): User override: "Rear"
ADJUDICATIONS = {
    "vp2_0010": {"agent": "rear-oblique", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0012": {"agent": "rear-oblique", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0017": {"agent": "front-oblique", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0018": {"agent": "rear-oblique", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0022": {"agent": "rear-oblique", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0023": {"agent": "front-oblique", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0027": {"agent": "rear-oblique", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0029": {"agent": "front-oblique", "chatgpt": "rear-oblique", "decision": "chatgpt_accepted", "final": "rear-oblique"},
    "vp2_0030": {"agent": "front", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0032": {"agent": "rear", "chatgpt": "rear-oblique", "decision": "chatgpt_accepted", "final": "rear-oblique"},
    "vp2_0041": {"agent": "front-oblique", "chatgpt": "side", "decision": "chatgpt_accepted", "final": "side"},
    "vp2_0044": {"agent": "unknown / ambiguous", "chatgpt": "side", "decision": "chatgpt_accepted", "final": "side"},
    "vp2_0048": {"agent": "front-oblique", "chatgpt": "side", "decision": "chatgpt_accepted", "final": "side"},
    "vp2_0049": {"agent": "side", "chatgpt": "rear", "decision": "chatgpt_accepted", "final": "rear"},
    "vp2_0054": {"agent": "side", "chatgpt": "rear-oblique", "decision": "chatgpt_accepted", "final": "rear-oblique"},
    "vp2_0055": {"agent": "side", "chatgpt": "unknown / ambiguous", "decision": "chatgpt_accepted", "final": "unknown / ambiguous"},
    "vp2_0060": {"agent": "rear-oblique", "chatgpt": "unknown / ambiguous", "decision": "user_override", "final": "rear"},
    "vp2_0061": {"agent": "side", "chatgpt": "unknown / ambiguous", "decision": "chatgpt_accepted", "final": "unknown / ambiguous"},
    "vp2_0064": {"agent": "rear-oblique", "chatgpt": "side", "decision": "chatgpt_accepted", "final": "side"},
    "vp2_0065": {"agent": "side", "chatgpt": "unknown / ambiguous", "decision": "chatgpt_accepted", "final": "unknown / ambiguous"},
    "vp2_0084": {"agent": "rear-oblique", "chatgpt": "side", "decision": "chatgpt_accepted", "final": "side"},
}

def main():
    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded {len(df)} rows from {MANIFEST_PATH}")
    
    chatgpt_col = []
    final_col = []
    status_col = []
    notes_col = []
    
    for _, row in df.iterrows():
        s_id = row["sample_id"]
        agent_vp = row["agent_viewpoint"]
        
        if s_id in ADJUDICATIONS:
            adj = ADJUDICATIONS[s_id]
            assert adj["agent"] == agent_vp, f"Mismatch for {s_id}: manifest has {agent_vp}, expected {adj['agent']}"
            chatgpt_vp = adj["chatgpt"]
            final_vp = adj["final"]
            status = "user_adjudicated_crosscheck"
            if adj["decision"] == "chatgpt_accepted":
                note = f"Disagreement resolved via user review: accepted ChatGPT label '{chatgpt_vp}' over agent '{agent_vp}'"
            else:
                note = f"Disagreement resolved via user review: user explicit override to '{final_vp}' (Agent: '{agent_vp}', ChatGPT: '{chatgpt_vp}')"
        else:
            chatgpt_vp = agent_vp
            final_vp = agent_vp
            status = "crosschecked_consensus"
            note = "Consensus between agent visual inspection and independent ChatGPT vision cross-check"
            
        chatgpt_col.append(chatgpt_vp)
        final_col.append(final_vp)
        status_col.append(status)
        notes_col.append(note)
        
    df["chatgpt_viewpoint"] = chatgpt_col
    df["final_viewpoint"] = final_col
    df["proposed_viewpoint"] = final_col  # For downstream script compatibility
    df["review_status"] = status_col
    df["adjudication_notes"] = notes_col
    
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"[OK] Successfully updated and saved: {MANIFEST_PATH}")
    
    print("\n--- Summary Statistics ---")
    print(f"Total samples: {len(df)}")
    print(f"Consensus samples (79%): {(df['review_status'] == 'crosschecked_consensus').sum()}")
    print(f"User adjudicated samples (21%): {(df['review_status'] == 'user_adjudicated_crosscheck').sum()}")
    print("\nFinal Viewpoint Distribution (N=100):")
    print(df["final_viewpoint"].value_counts().to_string())
    print("\nFinal Viewpoint Distribution by Dataset:")
    ct = pd.crosstab(df["dataset"], df["final_viewpoint"], margins=True)
    print(ct.to_string())

if __name__ == "__main__":
    main()
