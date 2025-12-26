import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def generate_cdf_plot(db_path="tests/phase_retrieval_results.db"):
    """
    Connects to the results DB and plots a CDF of error rates
    for 'phastphase', cases 1-7, deduped by near_field_md5.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return

    # Query Logic:
    # 1. Filter by method_name = 'phastphase'
    # 2. Filter by case_id between 1 and 7
    # 3. Group by 'near_field_md5' to handle multiple runs on the same problem.
    # 4. Take MIN(error) to represent the best result achieved for that problem instance.
    query = """
    SELECT MIN(error) as best_error
    FROM results
    WHERE method_name = 'phastphase'
      AND case_id BETWEEN 1 AND 7
    GROUP BY near_field_md5
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No data found matching criteria (method='phastphase', case_id 1-7).")
        print("Run the tests first to generate data.")
        return

    # Extract errors (handle potential NULLs if any)
    errors = np.array([r[0] for r in rows if r[0] is not None])
    
    if len(errors) == 0:
        print("Found matching records, but all errors were NULL.")
        return

    # --- Plotting the CDF ---
    
    # 1. Sort the error data
    errors_sorted = np.sort(errors)
    
    # 2. Calculate Cumulative Probability
    # y-axis points from 1/N to 1
    p = 1. * np.arange(1, len(errors) + 1) / len(errors)

    plt.figure(figsize=(10, 6))
    
    # Use step plot for proper CDF visualization
    plt.step(errors_sorted, p, where='post', label='phastphase')
    
    # Formatting
    plt.xscale('log') # Log scale is standard for error plots
    plt.xlabel('Reconstruction Error (Relative Norm)')
    plt.ylabel('Probability (Fraction of Cases Solved)')
    plt.title(f'CDF of PhastPhase Convergence Error\n(Case IDs 1-7, {len(errors)} Cases, 1000 Iterations, 1e-8 Gradient Tolerance)')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    # Optional: Add a threshold line for "Success"
    # plt.axvline(x=1e-2, color='r', linestyle='--', alpha=0.5, label='Success Threshold (1e-2)')
    
    output_filename = "error_cdf_phastphase.png"
    plt.savefig(output_filename)
    print(f"Graph saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    generate_cdf_plot()
