import pandas as pd
import re

def parse_seismic_modes(text_file, output_csv):
    with open(text_file, 'r', encoding='latin-1') as f:
        raw_text = f.read()

    # Fix specific OCR errors from the PDF extraction
    clean_text = raw_text.replace('ą', '±')
    
    parsed_data = []
    
    # Process line by line
    lines = clean_text.split('\n')
    for line in lines:
        tokens = line.split()
        
        # Skip empty lines, headers, or lines without enough data
        if len(tokens) < 3 or 'fobs' in line or 'Mode' in line:
            continue
            
        row = {'mode': tokens[0], 'f_obs': None, 'f_obs_unc': None, 
               'f_prem': None, 'q_obs': None, 'q_unc': None, 'q_prem': None}
               
        # Re-join the line to parse values with the ± symbol
        joined_line = " ".join(tokens[1:])
        
        # Find all numbers (with decimals) and numbers containing ±
        measurements = re.findall(r'(\d+\.\d+(?:\s*±\s*\.\d+|\s*±\s*\d+\.\d+)?|\.\d+)', joined_line)
        
        # Map extracted measurements to the appropriate columns
        if len(measurements) > 0:
            row['f_obs'], row['f_obs_unc'] = split_uncertainty(measurements[0])
        if len(measurements) > 1:
            row['f_prem'] = measurements[1]
        if len(measurements) > 2:
            row['q_obs'], row['q_unc'] = split_uncertainty(measurements[2])
        if len(measurements) > 3:
            row['q_prem'] = measurements[3]
            
        parsed_data.append(row)

    # Export
    df = pd.DataFrame(parsed_data)
    df.dropna(subset=['f_obs', 'f_prem'], how='all', inplace=True) # Drop invalid rows
    df.to_csv(output_csv, index=False)
    print(f"Data successfully exported to {output_csv}")

def split_uncertainty(val_str):
    """Splits strings like '309.25 ± 0.25' into a value and uncertainty."""
    if '±' in val_str:
        parts = val_str.split('±')
        return parts[0].strip(), parts[1].strip()
    return val_str.strip(), None

if __name__ == "__main__":
    parse_seismic_modes('raw_data.txt', 'Normal_Modes.csv')