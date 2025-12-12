def clean_student_data(df):
    # 1. Standardize Headers
    df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
    
    # 2. Clean Numerics (Handle Missing & Negatives)
    df['stress_level'] = pd.to_numeric(df['stress_level'], errors='coerce').fillna(5)
    df['sleep_hours'] = pd.to_numeric(df['sleep_hours'], errors='coerce').abs().fillna(7)
    
    # 3. Standardize Text (Gender)
    df['gender'] = df['gender'].str.lower().replace({'f': 'Female', 'm': 'Male'})

    # 4. Logic: Calculate 'Risk Status'
    # High Risk = Stress > 8 OR Sleep < 4 hours
    conditions = [(df['stress_level'] >= 8) | (df['sleep_hours'] < 4)]
    df['risk_status'] = np.select(conditions, ['High'], default='Low')

    # 5. Deduplicate
    return df.drop_duplicates()

