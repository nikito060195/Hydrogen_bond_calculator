import hbond_xyz
import numpy as np
import glob
import os
from pathlib import Path

################################################################## 
# Function to calculate hydrogen bonds in frames within an .xyz file 
##################################################################
def run_recursive(file):
    # --- Simulation (with PBC) ---
    
    # Define box limits and Periodic Boundary Conditions (PBC)
    box = (0.0, 64.168, 0, 64.12052, -10, 10)
    pbc = (True, True, False) # Enable PBC for x and y
    
    real_analysis = hbond_xyz.HBondAnalysis(
        filename=file, # Your real trajectory file
        box_limits=box,
        pbc=pbc,
        set_len=3.5,
        set_angle=30,
        n_cpus=None
    )
    
    # Execute the analysis
    real_analysis.run()
    
    # --- Queries ---
    
    # 1. Get total donations and acceptances for the frames
    frame_donations = real_analysis.query_donors()
    frame_acceptances = real_analysis.query_acceptors()
    
    donor_count = real_analysis.query_number_donors()[0]
    acceptor_count = real_analysis.query_number_acceptors()[0]
    
    avg_donations = np.sum(frame_donations) / len(frame_donations)
    avg_acceptances = np.sum(frame_acceptances) / len(frame_acceptances)
    
    avg_donations_per_molecule = avg_donations / donor_count
    avg_acceptances_per_molecule = avg_acceptances / acceptor_count

    hb_total = avg_donations_per_molecule + avg_acceptances_per_molecule
    
    return hb_total
