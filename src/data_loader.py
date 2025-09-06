import pandas as pd

def load_csv(path, require_target=False):
    """Load CSV, rename columns flexibly, validate.
    
    Prompt: instruction/text/prompt
    Target (optional): response/label/target
    ID: id or index
    
    Returns DF with 'id', 'prompt', optional target. Validate columns and IDs. Rename columns if needed and provide helpful errors.
    """
    df = pd.read_csv(path)
    
    # Find columns case-insensitively
    prompt_col = next((col for col in df.columns if col.lower() in ['instruction', 'text', 'prompt']), None)
    if prompt_col is None:
        raise ValueError("No prompt column found (expected: instruction, text, or prompt)")
    
    target_col = next((col for col in df.columns if col.lower() in ['response', 'label', 'target']), None)
    if require_target and target_col is None:
        raise ValueError("No target column found (expected: response, label, or target)")
    
    id_col = next((col for col in df.columns if col.lower() == 'id'), None)
    if id_col is None:
        df = df.reset_index()
        id_col = 'index'
    
    # Rename
    rename_map = {prompt_col: 'prompt'}
    if target_col:
        rename_map[target_col] = 'target'
    rename_map[id_col] = 'id'
    df = df.rename(columns=rename_map)
    
    # Select columns
    cols = ['id', 'prompt']
    if 'target' in df.columns:
        cols.append('target')
    if 'test_list' in df.columns:
        cols.append('test_list')
    df = df[cols]
    
    # Validate unique IDs
    if df['id'].duplicated().any():
        raise ValueError("Duplicate IDs in data")
    
    # Ensure types
    df['id'] = df['id'].astype(str)
    
    return df