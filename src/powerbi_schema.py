
def to_powerbi(errors, flags):
    return [
        {
            "AnomalyScore": e,
            "IsAnomaly": f,
            "Severity": "High" if f else "Low"
        }
        for e, f in zip(errors, flags)
    ]
