import pandas as pd

def sectorseparate(file_path):
    """
    Reads a CSV file and filters data to create separate DataFrames for each sector.
    DataFrames are sorted by 'VALUE RATIO' in descending order.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        dict: A dictionary where keys are sector names and values are the filtered, sorted DataFrames.
    """
    try:
        # Attempt to read the CSV file with UTF-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Apply filters
    df = df[
        (df['PE Ratio'] > 0.25) & 
        (df['Beta'] > 0) & 
        (df['Employees'] > 10) & 
        (df['EBIT GROWTH YoY'] > 0)
    ]

    # Ensure the 'Sector' and 'VALUE RATIO' columns exist
    if 'Sector' not in df.columns or 'VALUE RATIO' not in df.columns:
        raise ValueError("The required columns 'Sector' and 'VALUE RATIO' are not present in the dataset.")

    # Create a dictionary to store DataFrames for each sector
    sector_dfs = {}

    # Group by 'Sector' and sort each group by 'VALUE RATIO' descending
    for sector, group in df.groupby('Sector'):
        sorted_group = group.sort_values(by='VALUE RATIO', ascending=False).reset_index(drop=True)
        sector_dfs[sector] = sorted_group

    return sector_dfs
