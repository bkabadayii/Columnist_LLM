import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#################### CONFIGURATION ####################
# Enter the directory containing the JSON files.
# For example: input_directory = "/path/to/json_files"
input_directory = "../../../siu_questionnaire/agreement_ratings"

# Enter the output PDF file paths (adjust as needed).
output_pdf_means = "../../../siu_plots/means_table.pdf"
output_pdf_variances = "../../../siu_plots/variances_table.pdf"
#######################################################

def compute_stats_from_file(filepath, columns_order):
    """
    Reads a JSON file and computes the arithmetic mean and sample variance
    (using N-1 as the denominator) of the agreement scores for each key in columns_order.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stats_mean = {}
    stats_var = {}
    for key in columns_order:
        # Check if the key exists in the JSON
        if key in data:
            responses = data[key].get("responses", [])
            scores = [resp.get("agreement_score", 0) for resp in responses]
            if scores:
                mean_val = np.mean(scores)
                # Use sample variance (ddof=1); if only one score, variance is set to 0.
                var_val = np.var(scores, ddof=1) if len(scores) > 1 else 0
            else:
                mean_val = np.nan
                var_val = np.nan
        else:
            mean_val = np.nan
            var_val = np.nan
        stats_mean[key] = mean_val
        stats_var[key] = var_val
    return stats_mean, stats_var

def generate_pdf_table(df, output_path, title):
    """
    Renders a DataFrame as a table using matplotlib and saves it as a PDF.
    """
    fig, ax = plt.subplots(figsize=(max(12, len(df.columns) * 0.8), max(6, len(df.index) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    # Round the DataFrame values for display purposes
    table = ax.table(cellText=np.round(df.values, 2),
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

def main():
    # List of columnist names
    columnists = ["abdulkadirselvi", "ahmethakan", "fehimtastekin", 
                   "hilalkaplan", "ismailsaymaz", "mehmettezkan", 
                   "melihasik", "nagehanalci"]
    
    # Define the column order:
    # First, add the partyless questions: q1, q2, ..., q6.
    columns_order = [f"q{q}" for q in range(1, 7)]
    
    # Then, for each party, add the party-specific questions for q1 to q6.
    parties = ["akp", "chp", "mhp", "iyi", "hdp"]
    for party in parties:
        for q in range(1, 7):
            columns_order.append(f"{party}_q{q}")
    
    # Dictionaries to hold the computed means and variances per columnist
    means_data = {}
    variances_data = {}
    
    # Process each columnist's JSON file from the input directory
    for col in columnists:
        filename = f"{col}_ratings.json"
        filepath = os.path.join(input_directory, filename)
        if os.path.exists(filepath):
            m, v = compute_stats_from_file(filepath, columns_order)
            means_data[col] = m
            variances_data[col] = v
        else:
            print(f"File {filepath} not found. Skipping.")
    
    # Create DataFrames: rows are columnists, columns are the 36 keys as ordered above
    means_df = pd.DataFrame.from_dict(means_data, orient='index')
    variances_df = pd.DataFrame.from_dict(variances_data, orient='index')
    
    # Add a 9th row: overall (column-wise) mean computed across the 8 columnists
    means_df.loc["Overall"] = means_df.mean()
    variances_df.loc["Overall"] = variances_df.mean()
    
    # Reorder the rows so that the 8 columnists are in the desired order followed by "Overall"
    means_df = means_df.reindex(columnists + ["Overall"])
    variances_df = variances_df.reindex(columnists + ["Overall"])
    
    # Generate the PDF tables
    generate_pdf_table(means_df, output_pdf_means, "Means Table")
    generate_pdf_table(variances_df, output_pdf_variances, "Variances Table")
    
    print("PDF tables have been generated successfully.")

if __name__ == "__main__":
    main()