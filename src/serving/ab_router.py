import hashlib
import numpy as np
from statsmodels.stats.proportion import proportions_ztest  # <-- Correct library!

class ABRouter:
    def get_variant(self, transaction_id: str) -> str:
        # Hash the transaction ID to deterministically assign it to a bucket (0-99)
        h = int(hashlib.md5(f"fraud_exp_1:{transaction_id}".encode()).hexdigest(), 16)
        return "challenger" if h % 100 < 20 else "champion"

def check_significance(champion_results, challenger_results):
    count = np.array([sum(challenger_results), sum(champion_results)])
    nobs  = np.array([len(challenger_results), len(champion_results)])
    
    # 2-proportion Z-test to see if the difference is statistically significant
    stat, p = proportions_ztest(count, nobs)
    
    mean_champ = np.mean(champion_results)
    mean_chall = np.mean(challenger_results)
    uplift = (mean_chall - mean_champ) / mean_champ if mean_champ > 0 else 0
    
    return {
        "p_value": round(p, 4),
        "uplift_pct": round(uplift * 100, 2),
        "significant": p < 0.05 and len(challenger_results) >= 500
    }

if __name__ == "__main__":
    print("Initializing A/B Router (80% Champion / 20% Challenger)...")
    router = ABRouter()
    
    print("\nSimulating 5,000 live production transactions...")
    champion_metrics = []
    challenger_metrics = []
    
    for i in range(5000):
        # Route the transaction
        variant = router.get_variant(f"txn_{i}")
        
        # Simulate the model predicting correctly (1) or incorrectly (0)
        # We will heavily stack the deck to simulate the Challenger actually being better
        if variant == "champion":
            success = np.random.choice([0, 1], p=[0.15, 0.85]) # 85% accuracy
            champion_metrics.append(success)
        else:
            success = np.random.choice([0, 1], p=[0.08, 0.92]) # 92% accuracy
            challenger_metrics.append(success)
            
    print(f"Traffic Split Results: {len(champion_metrics)} to Champion, {len(challenger_metrics)} to Challenger")
    
    print("\nCalculating Statistical Significance...")
    results = check_significance(champion_metrics, challenger_metrics)
    
    print(f"P-Value: {results['p_value']}")
    print(f"Uplift: +{results['uplift_pct']}%")
    
    if results['significant']:
        print("✅ RESULT: Statistically significant (p < 0.05)! Safe to promote Challenger to 100% traffic.")
    else:
        print("❌ RESULT: Not significant yet. Keep collecting data.")