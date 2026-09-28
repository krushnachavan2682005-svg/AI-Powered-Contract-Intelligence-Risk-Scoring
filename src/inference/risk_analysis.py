def analyze_risk(extractions):
    indicators = []
    for ext in extractions:
        cat = ext["category"]
        if cat in ["Uncapped Liability", "Liquidated Damages", "Termination For Convenience"]:
            indicators.append({
                "category": cat,
                "severity": "high",
                "reason": f"A {cat} clause was detected.",
                "evidence": ext["answer"],
                "requires_manual_review": True
            })
        elif cat in ["Audit Rights", "Anti-Assignment"]:
            indicators.append({
                "category": cat,
                "severity": "medium",
                "reason": f"A {cat} clause requires review.",
                "evidence": ext["answer"],
                "requires_manual_review": True
            })
    return indicators
