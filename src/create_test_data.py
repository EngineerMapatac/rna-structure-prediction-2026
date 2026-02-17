import pandas as pd
import os
try:
    from src import config
except ImportError:
    import config

def create_dummy_test_data():
    # 1. Define some fake RNA sequences
    # These look like real data but are just for testing the pipeline
    test_data = [
        {
            'target_id': 'Test_Target_001',
            'sequence': 'AGGUCAGGUCAGGUCAGGUC',  # A short RNA sequence
            'description': 'Dummy test sequence 1'
        },
        {
            'target_id': 'Test_Target_002',
            'sequence': 'CCCAAAUUUGGGCCCAAA',   # Another sequence
            'description': 'Dummy test sequence 2'
        },
        {
            'target_id': 'Test_Target_003',
            'sequence': 'UUGGAACCAAUUGGAACCAA', 
            'description': 'Dummy test sequence 3'
        }
    ]
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(test_data)
    
    # 3. Save to data/raw/test_sequences.csv
    output_path = config.TEST_CSV
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print(f"✅ Success! Created dummy test file at: {output_path}")
    print(f"Contains {len(df)} sequences.")
    print(df.head())

if __name__ == "__main__":
    create_dummy_test_data()