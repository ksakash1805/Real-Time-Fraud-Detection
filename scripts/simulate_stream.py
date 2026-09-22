"""
Script to simulate real-time streaming of transactions to the API.

WHY:
In a real production environment, transactions arrive sequentially in real-time.
This script tests how our API handles a stream of data and allows us to see
the system working in action. 

MENTOR NOTE (Real-time ML Inference):
For real-time systems, latency is critical. Notice how we track 'avg_response_time'.
If predictions take too long (e.g., > 100ms), the credit card transaction might timeout
at the point of sale. 
"""

import time
import random
import argparse
import requests
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Assume src.config is available
from src.config import DATA_PATH, TARGET_COLUMN

# ANSI colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def load_test_data() -> pd.DataFrame:
    """
    Load a subset of the dataset to simulate incoming traffic.
    Returns a mix of fraud and non-fraud cases for interesting simulation.
    """
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Get all frauds and a random sample of legit transactions
        frauds = df[df[TARGET_COLUMN] == 1]
        legits = df[df[TARGET_COLUMN] == 0].sample(n=len(frauds)*10, random_state=42)
        
        # Combine and shuffle
        test_df = pd.concat([frauds, legits]).sample(frac=1, random_state=42).reset_index(drop=True)
        return test_df
    except Exception as e:
        print(f"Error loading data: {e}")
        # Return dummy data if file not found
        return pd.DataFrame()

def send_transaction(transaction: dict, api_url: str) -> dict:
    """
    POST a single transaction to the API endpoint.
    """
    endpoint = f"{api_url.rstrip('/')}/predict"
    try:
        start_time = time.time()
        response = requests.post(endpoint, json=transaction, timeout=2.0)
        response.raise_for_status()
        
        result = response.json()
        result['network_latency_ms'] = (time.time() - start_time) * 1000
        return result
    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        return None

def run_simulation(n_transactions=100, delay_range=(0.1, 0.5), api_url='http://localhost:8000'):
    """
    Stream transactions one by one with a random delay.
    """
    df = load_test_data()
    if df.empty:
        return
        
    n_transactions = min(n_transactions, len(df))
    print(f"Starting simulation for {n_transactions} transactions...")
    print("-" * 50)
    
    stats = {
        'total': 0,
        'frauds_detected': 0,
        'legit': 0,
        'total_latency': 0.0
    }
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    for idx, row in df.head(n_transactions).iterrows():
        # Remove target column before sending
        tx_dict = row.drop(TARGET_COLUMN).to_dict()
        true_label = int(row[TARGET_COLUMN])
        
        result = send_transaction(tx_dict, api_url)
        
        if result:
            stats['total'] += 1
            stats['total_latency'] += result['network_latency_ms']
            
            is_fraud = result['is_fraud']
            prob = result['fraud_probability']
            latency = result['network_latency_ms']
            
            if is_fraud:
                stats['frauds_detected'] += 1
                color = RED
                status = f"FRAUD ALERT! (Prob: {prob:.4f})"
            else:
                stats['legit'] += 1
                color = GREEN
                status = f"LEGIT (Prob: {prob:.4f})"
                
            # Log output
            print(f"{color}[Tx {idx+1:03d}] True: {true_label} | Pred: {status} | Risk: {result['risk_level']} | Latency: {latency:.1f}ms{RESET}")
            
        # Simulate arrival delay
        delay = random.uniform(*delay_range)
        time.sleep(delay)
        
    # Print Summary
    avg_latency = stats['total_latency'] / max(1, stats['total'])
    print("-" * 50)
    print("SIMULATION COMPLETE")
    print(f"Total processed:   {stats['total']}")
    print(f"Frauds flagged:    {stats['frauds_detected']}")
    print(f"Legit passed:      {stats['legit']}")
    print(f"Avg response time: {avg_latency:.2f} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simulate streaming transactions to API")
    parser.add_argument('--n', type=int, default=100, help="Number of transactions to send")
    parser.add_argument('--url', type=str, default='http://localhost:8000', help="API base URL")
    args = parser.parse_args()
    
    run_simulation(n_transactions=args.n, api_url=args.url)
