import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import srt_config as config  

def parse_tenor_to_years(tenor_str):
    """Converts standard tenor formats (e.g., '5Y', '6M') to float years."""
    if pd.isna(tenor_str): return None
    t = str(tenor_str).upper().strip()
    try:
        if t.endswith('Y'): return float(t[:-1])
        elif t.endswith('M'): return float(t[:-1]) / 12.0
        elif t.endswith('W'): return float(t[:-1]) / 52.0
        else: return float(t)
    except: return None

def get_rating_rank(rating):
    """Assigns numerical rank for sorting ratings from AAA to Default."""
    order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 
             'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D', 'NR']
    clean = str(rating).replace('*', '').strip()
    return order.index(clean) if clean in order else 999

def bootstrap_hazard_rates(group):
    """Calculates forward hazard rates and survival probabilities from par CDS spreads."""
    group = group.sort_values('Years')
    forward_lambdas = []
    survival_probs = [] 
    times = []
    prev_time = 0.0
    prev_loss_area = 0.0 
    
    for i, row in group.iterrows():
        t = row['Years']
        s = row['ParSpreadMid'] 
        if t <= prev_time: continue
            
        current_total_loss_area = (s * t) / (1 - config.FIXED_RECOVERY_CDS)
        
        if prev_time == 0:
            lambda_val = current_total_loss_area / t
        else:
            risk_added = max(0.0, current_total_loss_area - prev_loss_area)
            lambda_val = risk_added / (t - prev_time)
        
        forward_lambdas.append(lambda_val)
        times.append(t)
        survival_probs.append(np.exp(-current_total_loss_area))
        
        prev_time = t
        prev_loss_area = current_total_loss_area
        
    return pd.DataFrame({'Years': times, 'Lambda': forward_lambdas, 'Survival_Prob': survival_probs})

def generate_curves():
    file_path = config.CDS_RAW_FILE
    if not os.path.exists(file_path):
        file_path = os.path.join("MEX", config.CDS_RAW_FILE)
        if not os.path.exists(file_path):
            print(f"[ERROR] Could not locate {config.CDS_RAW_FILE}")
            return

    print(f"Reading market data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    df = df[df['Currency'].isin(['USD', 'EUR'])].copy()
    
    if 'Tenor' not in df.columns and 'Maturity' in df.columns:
        df.rename(columns={'Maturity': 'Tenor'}, inplace=True)
    
    df['Years'] = df['Tenor'].apply(parse_tenor_to_years)
    df = df.dropna(subset=['Years'])
    df = df[df['Years'] <= config.MAX_YEARS]
    
    if 'AvRating' in df.columns:
        df['Rating'] = df['AvRating'].fillna(df.get('ImpliedRating', pd.Series()))
    else:
        df['Rating'] = df['ImpliedRating']
        
    df = df.dropna(subset=['Rating', 'ParSpreadMid'])
    df['Rating'] = df['Rating'].astype(str).str.replace('*', '').str.strip()
    
    df['ParSpreadMid'] = pd.to_numeric(df['ParSpreadMid'], errors='coerce')
    if df['ParSpreadMid'].median() > 1: df['ParSpreadMid'] = df['ParSpreadMid'] / 10000

    for currency in ['EUR', 'USD']:
        print(f"Processing {currency} curves...")
        df_curr = df[df['Currency'] == currency].copy()
        if df_curr.empty: continue
        
        df_grouped = df_curr.groupby(['Rating', 'Years'])['ParSpreadMid'].median().reset_index()
        results = []
        for rating in df_grouped['Rating'].unique():
            rating_data = df_grouped[df_grouped['Rating'] == rating]
            bootstrapped = bootstrap_hazard_rates(rating_data)
            bootstrapped['Rating'] = rating
            results.append(bootstrapped)
            
        if not results: continue
        final_df = pd.concat(results)
        
        matrix_lambda = final_df.pivot(index='Rating', columns='Years', values='Lambda')
        matrix_surv = final_df.pivot(index='Rating', columns='Years', values='Survival_Prob')
        
        matrix_lambda['SortKey'] = matrix_lambda.index.map(get_rating_rank)
        matrix_lambda = matrix_lambda.sort_values('SortKey').drop(columns=['SortKey'])
        matrix_lambda = matrix_lambda.interpolate(axis=1, limit_direction='both')
        
        matrix_surv['SortKey'] = matrix_surv.index.map(get_rating_rank)
        matrix_surv = matrix_surv.sort_values('SortKey').drop(columns=['SortKey'])
        matrix_surv = matrix_surv.interpolate(axis=1, limit_direction='both')
        
        output_file_lambda = f"bootstrapped_hazard_rates_{currency}.csv"
        output_file_surv = f"bootstrapped_survival_probs_{currency}.csv"
        matrix_lambda.to_csv(output_file_lambda)
        matrix_surv.to_csv(output_file_surv)
        
        print(f"✅ Exported {currency} survival matrix: {output_file_surv}")
        
        # Plot generation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ratings_to_plot = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        
        existing_ratings = [r for r in ratings_to_plot if r in matrix_lambda.index]
        if not existing_ratings: existing_ratings = matrix_lambda.index[:5]

        for rating in existing_ratings:
            ax1.plot(matrix_lambda.columns, matrix_lambda.loc[rating], marker='o', label=rating)
            ax2.plot(matrix_surv.columns, matrix_surv.loc[rating], marker='o', label=rating)
                
        ax1.set_title(f"Forward Hazard Rates ({currency})")
        ax1.set_xlabel("Maturity (Years)")
        ax1.set_ylabel("Annual Default Intensity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title(f"Survival Curves ({currency}) - S(t)")
        ax2.set_xlabel("Maturity (Years)")
        ax2.set_ylabel("Cumulative Survival Probability")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_filename = f"Survival_Curves_Plot_{currency}.png"
        plt.savefig(plot_filename, dpi=300)
        
        plt.show()

if __name__ == "__main__":
    generate_curves()
