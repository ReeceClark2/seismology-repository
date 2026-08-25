import pdfplumber
import pandas as pd
import re

def parse_normal_modes(pdf_path, output_csv):
    print(f"Opening {pdf_path} for spatial extraction...")
    all_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract all words with their bounding box coordinates (x0, top, x1, bottom, text)
            words = page.extract_words()
            
            if not words:
                continue

            # 1. Filter out the footer block on the last page (or any page)
            # Looking at the layout, the table data never goes below y = 720
            table_words = [w for w in words if w['top'] < 725 and w['top'] > 75]

            # 2. Separate into Left Table and Right Table based on horizontal X coordinate
            # Left table roughly X: 30 to 300, Right table roughly X: 320 to 580
            left_words = [w for w in table_words if w['x0'] < 315]
            right_words = [w for w in table_words if w['x0'] >= 315]

            for side_name, side_words in [('left', left_words), ('right', right_words)]:
                if not side_words:
                    continue

                # Group words into lines by their vertical 'top' position (tolerance ~ 4 points)
                lines = []
                side_words = sorted(side_words, key=lambda w: (w['top'], w['x0']))
                
                current_line = []
                last_top = None
                for w in side_words:
                    if last_top is None or abs(w['top'] - last_top) > 4:
                        if current_line:
                            lines.append(current_line)
                        current_line = [w]
                        last_top = w['top']
                    else:
                        current_line.append(w)
                if current_line:
                    lines.append(current_line)

                # 3. Process each line into structured fields
                for line in lines:
                    line_text_tokens = [w['text'] for w in line]
                    line_str = " ".join(line_text_tokens)

                    # Skip header rows like "Mode fobs fPREM..."
                    if 'fobs' in line_str or 'TABLE' in line_str or 'Mode' in line_str:
                        continue

                    # A valid mode name is the first token (e.g., 0S2, 20S10, 3T9)
                    first_token = line_text_tokens[0]
                    if not re.match(r'^\d*[ST]\d+$', first_token.replace(' ', '')):
                        continue

                    # Reconstruct text with symbols and parse measurements
                    parsed_row = parse_line_tokens(line_text_tokens)
                    if parsed_row:
                        all_rows.append(parsed_row)

    # 4. Convert to DataFrame and Export
    columns = ['mode', 'f_obs', 'f_obs_unc', 'f_prem', 'q_obs', 'q_unc', 'q_prem', 'ref']
    df = pd.DataFrame(all_rows, columns=columns)
    
    # Clean up reference column (combine remaining tokens if any)
    df.to_csv(output_csv, index=False)
    print(f"Success! Extracted {len(df)} rows and saved to {output_csv}")


def parse_line_tokens(tokens):
    """
    Intelligently parses a list of words belonging to a single table row,
    handling missing q_obs cells and uncertainty bounds (±).
    """
    mode = tokens[0].replace(' ', '')
    remaining = tokens[1:]

    # Join tokens back together to easily extract values with '±'
    line_joined = " ".join(remaining)
    line_joined = re.sub(r'([ą\?]|Â±|A±)', '±', line_joined)

    # We expect numbers, decimals, '±', and reference text (like 'WT', '33c,tvf', 'I')
    # Let's tokenize by spaces, keeping '±' attached or separate
    parts = line_joined.split()

    vals = []
    ref_parts = []
    
    i = 0
    while i < len(parts):
        p = parts[i]
        # Check if this part is a number or part of an uncertainty pair
        if re.match(r'^\d*\.\d+$', p) or re.match(r'^\d+$', p):
            if i + 1 < len(parts) and parts[i+1] == '±':
                val = p
                unc = parts[i+2] if i + 2 < len(parts) else ""
                vals.append({'val': val, 'unc': unc})
                i += 3
                continue
            else:
                vals.append({'val': p, 'unc': None})
                i += 1
                continue
        elif p == '±':
            i += 1
            continue
        else:
            # It's part of the reference string (e.g., 'WT', '33c,tvf', 'I')
            ref_parts.append(p)
            i += 1

    # Map the extracted values (vals) into our fixed schema:
    # f_obs (val+unc), f_prem (val), q_obs (val+unc, optional), q_prem (val), ref
    row = {'mode': mode, 'f_obs': None, 'f_obs_unc': None, 
           'f_prem': None, 'q_obs': None, 'q_unc': None, 'q_prem': None, 'ref': " ".join(ref_parts)}

    # Let's map based on total count of value blocks found:
    # Typical full row has 4 value blocks: [f_obs, f_prem, q_obs, q_prem]
    # If q_obs is missing, it will have 3 value blocks: [f_obs, f_prem, q_prem] or similar.
    
    if len(vals) == 4:
        row['f_obs'] = vals[0]['val']
        row['f_obs_unc'] = vals[0]['unc']
        row['f_prem'] = vals[1]['val']
        row['q_obs'] = vals[2]['val']
        row['q_unc'] = vals[2]['unc']
        row['q_prem'] = vals[3]['val']
    elif len(vals) == 3:
        # Determine if q_obs is missing or f_obs is missing
        # Usually, if vals[0] has an uncertainty, it's f_obs. 
        # If vals[1] is close to f_obs magnitude, check columns. 
        # In PREM tables, f_obs and f_prem are always the first two.
        row['f_obs'] = vals[0]['val']
        row['f_obs_unc'] = vals[0]['unc']
        row['f_prem'] = vals[1]['val']
        # The 3rd value could be q_obs or q_prem. 
        # If q_obs was missing, vals[2] is q_prem. Let's check if vals[2] has an uncertainty.
        if vals[2]['unc'] is not None:
            # It has an uncertainty, so it must be q_obs! Thus q_prem is missing.
            row['q_obs'] = vals[2]['val']
            row['q_unc'] = vals[2]['unc']
        else:
            # No uncertainty, so it's q_prem (and q_obs is missing)
            row['q_prem'] = vals[2]['val']
    elif len(vals) == 2:
        row['f_prem'] = vals[0]['val']
        row['q_prem'] = vals[1]['val']
    elif len(vals) == 1:
        row['f_prem'] = vals[0]['val']

    return row

if __name__ == "__main__":
    parse_normal_modes('Normal_Mode_Table.pdf', 'Normal_Modes.csv')