# hbond_xyz.py
# Python Module for hydrogen bonds analysis
# by trajectory files .xyz, based on H--D--A angle.

import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import math

# =============================================================================
# Calc functions (internal)
# Outside the class 'multiprocessing'
# =============================================================================

def _app_minimum_image(r1, r2, xlo, xhi, ylo, yhi, zlo, zhi, px, py, pz):
    """Apply the condition of of minimum image to r2 in relation to r1."""
    box = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
    delta = r2 - r1
    for i, periodic in enumerate([px, py, pz]):
        if periodic:
            while delta[i] >  box[i] / 2.0: delta[i] -= box[i]
            while delta[i] < -box[i] / 2.0: delta[i] += box[i]
    return r1 + delta

def _angle_calculation(a, b, c):
    """Calculates the angle in the vertice 'a' (degrees)."""
    ba = b - a
    ca = c - a
    norm_ba = np.linalg.norm(ba)
    norm_ca = np.linalg.norm(ca)
    if norm_ba == 0 or norm_ca == 0:
        return 0.0
    cos_angle = np.dot(ba, ca) / (norm_ba * norm_ca)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def _replicates_with_pbc(coordinates, xlo, xhi, ylo, yhi, zlo, zhi, px, py, pz):
    """Creates coordeinate replicas based on PBC."""
    replicas = []
    box_size_x = xhi - xlo
    box_size_y = yhi - ylo
    box_size_z = zhi - zlo
    
    for coord in coordinates:
        x, y, z = coord
        displacements_x = [-1, 0, 1] if px else [0]
        displacements_y = [-1, 0, 1] if py else [0]
        displacements_z = [-1, 0, 1] if pz else [0]

        for dx in displacements_x:
            for dy in displacements_y:
                for dz in displacements_z:
                    replicas.append([
                        x + dx * box_size_x,
                        y + dy * box_size_y,
                        z + dz * box_size_z
                    ])
    return np.array(replicas)

def _finds_neighbors(tree, coords, r):
    """Finds neighbors using a cKDTree."""
    return tree.query_ball_point(coords, r)

def _process_frame_atom_unified(frame_str, donor_name, acceptor_name, hydrogen_name, box_limits, pbc):
    """
    Processes a only frame (string) and returns a list of atoms
    with unified IDs per molecule
    """
    lines = frame_str.strip().split('\n')
    if len(lines) < 2:
        return [] # Frame wrong
        
    # Filtrar lines relevantes
    lines_filtered = [line.split() for line in lines[2:] if
                        line.split()[0] in (acceptor_name, donor_name, hydrogen_name)]

    results = []
    mol_id = 0
    i = 0
    (lx_min, lx_max, ly_min, ly_max, lz_min, lz_max) = box_limits
    (px, py, pz) = pbc

    while i < len(lines_filtered):
        line = lines_filtered[i]
        atom_name = line[0]

        # Assuming that Donor and Acceptor are the same atom (ex.: 'O' water)
        if atom_name == acceptor_name and atom_name == donor_name:
            mol_id += 1 
            counter_hydrogen = 0
            coords = {
                "x": float(line[1]),
                "y": float(line[2]),
                "z": float(line[3])
            }
            results.append({"ID": mol_id, "type": 'acceptor', **coords})
            results.append({"ID": mol_id, "type": 'donor', **coords})
            
            i += 1 

            # Loop to get the hydrogens from this molecule
            while i < len(lines_filtered) and lines_filtered[i][0] == hydrogen_name:
                h_line = lines_filtered[i]
                counter_hydrogen += 1
                info_h = {
                    "ID": mol_id, 
                    "type": f"H{counter_hydrogen}",
                    "x": float(h_line[1]),
                    "y": float(h_line[2]),
                    "z": float(h_line[3])
                }
                results.append(info_h)
                i += 1
        else:
            # Logical to lead with donor/acceptors that no are the same
            if atom_name == acceptor_name:
                 mol_id += 1
                 results.append({
                    "ID": mol_id, "type": 'acceptor', 
                    "x": float(line[1]), "y": float(line[2]), "z": float(line[3])
                 })
            elif atom_name == donor_name:
                 mol_id += 1
                 results.append({
                    "ID": mol_id, "type": 'donor',
                    "x": float(line[1]), "y": float(line[2]), "z": float(line[3])
                 })
            # Ignore hydrogens that no belong from a donor recognized
            i += 1
    
    # --- Correction time of minimum image correction (PBC) ---
    donors_map = {}
    for atom in results:
        if atom["type"] == "donor":
            donors_map[atom["ID"]] = np.array([atom["x"], atom["y"], atom["z"]])

    for h_atom in results:
        if h_atom["type"].startswith("H"):
            h_id = h_atom["ID"] 
            if h_id in donors_map:
                donor_coords = donors_map[h_id]
                h_coords = np.array([h_atom["x"], h_atom["y"], h_atom["z"]])
                
                dist = np.linalg.norm(h_coords - donor_coords)
                if dist > 1.5: # Cut of 1.5Å to bonds O-H
                    h_fixed = _app_minimum_image(donor_coords, h_coords,
                                                        lx_min, lx_max, ly_min, ly_max, lz_min, lz_max,
                                                        px, py, pz)
                    
                    h_atom["x"] = h_fixed[0]
                    h_atom["y"] = h_fixed[1]
                    h_atom["z"] = h_fixed[2]
                            
    return results

def _caculates_bonds_hydrogen_PER_ATOM(packs_frame, box_limits, pbc, set_len, set_angle):
    """
    Calculates HBs to one frame, returning counters per atom
    Metode: D-A, H-D-A, allows bifurcation (no 'break').
    """
    (lx_min, lx_max, ly_min, ly_max, lz_min, lz_max) = box_limits
    (px, py, pz) = pbc
    
    mol_hb_counts = {}
    acceptors_info = [] 
    packs_donor = []  
    all_mol_ids = set()
    
    for mol_pack in packs_frame: 
        if not mol_pack: continue
        mol_id = mol_pack[0]["ID"]
        all_mol_ids.add(mol_id)
        current_donor_packet = []
        for atom in mol_pack:
            if atom["type"] == 'acceptor':
                acceptors_info.append({
                    "ID": atom["ID"],
                    "coords": np.array([atom["x"], atom["y"], atom["z"]])
                })
            elif atom["type"] == 'donor':
                current_donor_packet.insert(0, atom) 
            elif atom["type"].startswith("H"):
                current_donor_packet.append(atom)
        
        if current_donor_packet:
            packs_donor.append(current_donor_packet)
            
    for mol_id in all_mol_ids:
        mol_hb_counts[mol_id] = {"donated": 0, "accepted": 0}

    if not acceptors_info or not packs_donor:
        return mol_hb_counts 

    coordinates_acceptor_orig = np.array([info["coords"] for info in acceptors_info])
    ids_acceptor_orig = np.array([info["ID"] for info in acceptors_info])
    
    if coordinates_acceptor_orig.shape[0] == 0:
        return mol_hb_counts

    coordinates_acceptor_replicated = _replicates_with_pbc(coordinates_acceptor_orig,
                                                                 lx_min, lx_max, ly_min, ly_max, lz_min, lz_max,
                                                                 px, py, pz)
    
    if len(coordinates_acceptor_replicated) == 0:
        num_replicas_per_atom = 0
    else:
        # Ensure integer division
        num_replicas_per_atom = int(round(len(coordinates_acceptor_replicated) / len(ids_acceptor_orig)))

    # Critical correction (np.repeat):
    ids_acceptor_replicated = np.repeat(ids_acceptor_orig, num_replicas_per_atom)

    acceptor_tree = cKDTree(coordinates_acceptor_replicated)

    for pack_d in packs_donor: 
        donor_atom = pack_d[0]
        d_pos = np.array([donor_atom["x"], donor_atom["y"], donor_atom["z"]])
        d_id = donor_atom["ID"] 
        
        hydrogens_from_this_donor = []
        for atom in pack_d[1:]:
            if atom["type"].startswith("H"):
                hydrogens_from_this_donor.append(np.array([atom["x"], atom["y"], atom["z"]]))
        
        if not hydrogens_from_this_donor:
            continue

        acceptor_indices = _finds_neighbors(acceptor_tree, d_pos, set_len)
        if not acceptor_indices:
            continue

        for h_pos in hydrogens_from_this_donor:
            for a_idx in acceptor_indices:
                if a_idx >= len(ids_acceptor_replicated): continue # Security check
                a_pos = coordinates_acceptor_replicated[a_idx]
                a_id = ids_acceptor_replicated[a_idx]
                
                if d_id == a_id: 
                    continue 

                angulo = _angle_calculation(d_pos, h_pos, a_pos) 
                
                if abs(angulo) <= set_angle: 
                    mol_hb_counts[d_id]["donated"] += 1
                    mol_hb_counts[a_id]["accepted"] += 1
                    
    return mol_hb_counts

# --- Wrappers to the Multiprocessing ---

def _process_frame_wrapper(args):
    """Wrapper para a função de processamento de frame."""
    frame_str, donor, acceptor, hydrogen, box, pbc = args
    return _process_frame_atom_unified(frame_str, donor, acceptor, hydrogen, box, pbc)

def _calculates_hb_wrapper(args):
    """Wrapper to the function of HB calculation."""
    packs_frame, box, pbc, set_len, set_angle = args
    return _caculates_bonds_hydrogen_PER_ATOM(packs_frame, box, pbc, set_len, set_angle)


# =============================================================================
# MAIN CLASS OF THE LIBRARY
# =============================================================================

class HBondAnalysis:
    """
    Class to calculate and to analyse hydrogen bonds by .xyz files
    Ex. to use:
        import hbond_xyz
        
        # System with PBC
        box = (0, 30, 0, 30, 0, 30)
        pbc = (True, True, True)
        analysis = hbond_xyz.HBondAnalysis(
            filename='traj.xyz',
            box_limits=box,
            pbc=pbc
        )
        
        # without PBC (ex: cluster)
        analysis_cluster = hbond_xyz.HBondAnalysis(filename='cluster.xyz')

        # Executar o cálculo
        analysis.run()
        
        # Obter results
        print(analysis.query_donors())
        print(analysis.query_acceptors(frames=0))
        print(analysis.query_number_donors(frames=0))
    """
    def __init__(self, filename, 
                 donor="O", acceptor="O", hydrogen="H", 
                 box_limits=None, pbc=(False, False, False),
                 set_len=3.5, set_angle=30, n_cpus=None):
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File do not found: {filename}")
            
        self.filename = filename
        self.donor = donor
        self.acceptor = acceptor
        self.hydrogen = hydrogen
        self.set_len = set_len
        self.set_angle = set_angle
        
        if n_cpus is None:
            self.n_cpus = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
        else:
            self.n_cpus = n_cpus

        self.pbc = pbc
        if box_limits:
            if len(box_limits) != 6:
                raise ValueError("box_limits must be a tupla/lis with 6 values: (lx_min, lx_max, ly_min, ly_max, lz_min, lz_max)")
            self.box_limits = box_limits
        else:
            if any(pbc):
                raise ValueError("PBC can not be allowed (pbc=True) if box_limits are not provided.")
            print("Notice: box_limits do not provide. Calculating limits by file data...")
            self.box_limits = self._auto_detect_box()
            print(f"Box limits detected: {self.box_limits}")

        # Storage de results
        self.raw_frames = []
        self.atom_data_per_frame = []
        self.atom_packets_per_frame = []
        self.results_per_frame = []
        self.total_donations_per_frame = []
        self.total_acceptances_per_frame = []
        self.number_of_donors_per_frame = []
        self.number_of_acceptors_per_frame = []
        
        self._is_run_complete = False

    def _auto_detect_box(self):
        """Reads the folder to found the min/max coordinate limits."""
        min_x, max_x = math.inf, -math.inf
        min_y, max_y = math.inf, -math.inf
        min_z, max_z = math.inf, -math.inf
        
        try:
            with open(self.filename, 'r') as f:
                while True:
                    line1 = f.readline()
                    if not line1:
                        break 
                    
                    try:
                        num_atoms = int(line1.strip())
                    except ValueError:
                        print(f"Notice: Frame header invalid ignored: {line1.strip()}")
                        continue
                        
                    f.readline()
                    
                    for _ in range(num_atoms):
                        line = f.readline()
                        if not line:
                            break
                        
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                min_x, max_x = min(min_x, x), max(max_x, x)
                                min_y, max_y = min(min_y, y), max(max_y, y)
                                min_z, max_z = min(min_z, z), max(max_z, z)
                            except ValueError:
                                print(f"Notice: Atom line invalid ignored: {line.strip()}")
                                continue
            
            return (min_x, max_x, min_y, max_y, min_z, max_z)
        
        except Exception as e:
            print(f"Error! The file could not be read to detect the box: {e}")
            return (0,0,0,0,0,0)

    def _read_and_process_frames(self):
        """Reads the .xyz and processes the frames in parallel."""
        #print(f"Reading and processing frames of {self.filename}...")
        with open(self.filename, 'r') as f:
            content = f.read()
        
        try:
            num_atoms_header_line = content.split('\n', 1)[0]
            num_atoms_header = int(num_atoms_header_line.strip())
            num_lines_por_frame = num_atoms_header + 2
            lines = content.strip().split('\n')
            
            self.raw_frames = [
                '\n'.join(lines[i:i+num_lines_por_frame]) 
                for i in range(0, len(lines), num_lines_por_frame)
                if len(lines[i:i+num_lines_por_frame]) == num_lines_por_frame
            ]
        except Exception:
            print("Notice: Frame format is inconsistent. Trying vidision by 'Timestep' (Header of the line 1)...")
            header_line = content.split('\n', 1)[0].strip()
            self.raw_frames = [f"{header_line}\n{frame}" for frame in content.split(header_line)[1:] if frame.strip()]
            if not self.raw_frames:
                 raise ValueError("The file could not be split in frames. Check the .xyz file format.")

        #print(f"Found: {len(self.raw_frames)} frames. Processing atoms...")
        
        args_list = [(frame_str, self.donor, self.acceptor, self.hydrogen, self.box_limits, self.pbc) 
                     for frame_str in self.raw_frames]
        
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            self.atom_data_per_frame = list(executor.map(_process_frame_wrapper, args_list))
        
        #print("The processing of atoms is done.")

    def _group_data_by_id(self):
        """Clustering of processed atom data into packets by molecule/ID."""
        #print("Clustering atoms by ID...")
        self.atom_packets_per_frame = []
        for frame_data in self.atom_data_per_frame:
            current_packs_frame = []
            if not frame_data:
                self.atom_packets_per_frame.append(current_packs_frame)
                continue
            
            frame_data.sort(key=lambda x: x['ID'])
            
            current_id = -1
            current_pack = []
            for atom in frame_data:
                if atom['ID'] != current_id:
                    if current_pack:
                        current_packs_frame.append(current_pack)
                    current_pack = [atom]
                    current_id = atom['ID']
                else:
                    current_pack.append(atom)
            
            if current_pack:
                current_packs_frame.append(current_pack)
                
            self.atom_packets_per_frame.append(current_packs_frame)
        #print("Clustering completed.")

    def _run_parallel_hb_calc(self):
        """Performs the calculation of HB in parallel."""
        #print(f"Calculating HBs in {len(self.atom_packets_per_frame)} frames (using {self.n_cpus} CPUs)...")
        
        args_list = [(packs, self.box_limits, self.pbc, self.set_len, self.set_angle)
                     for packs in self.atom_packets_per_frame]
        
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            self.results_per_frame = list(executor.map(_calculates_hb_wrapper, args_list))
        
        #print("Calculation of HBs completed.")

    def _post_process_results(self):
        """It calculates the totals per frame from the data per atom."""
        #print("Processing the results...")
        self.total_donations_per_frame = []
        self.total_acceptances_per_frame = []
        self.number_of_donors_per_frame = []
        self.number_of_acceptors_per_frame = []
        
        # Iterate over the results of the HB calculation
        for frame_result_dict in self.results_per_frame:
            total_d = sum(counts['donated'] for counts in frame_result_dict.values())
            total_a = sum(counts['accepted'] for counts in frame_result_dict.values())
                
            self.total_donations_per_frame.append(total_d)
            self.total_acceptances_per_frame.append(total_a)
        
        # --- Iterare over the input atom data ---
        for frame_atom_list in self.atom_data_per_frame:
            # count how many atoms of the donor and acceptor type there are in current frame
            n_donors = 0
            n_acceptors = 0
            for atom in frame_atom_list:
                if atom['type'] == 'donor':
                    n_donors += 1
                elif atom['type'] == 'acceptor':
                    n_acceptors += 1
            
            self.number_of_donors_per_frame.append(n_donors)
            self.number_of_acceptors_per_frame.append(n_acceptors)
        
        #print("Processing the results completed.")


    def run(self):
        """
        It executes all the entire calculation pipeline
        """
        #print("--- The HB analysis has begun ---")
        self._read_and_process_frames()
        self._group_data_by_id()
        self._run_parallel_hb_calc()
        self._post_process_results()
        self._is_run_complete = True
        #print("--- The HB analysis has been completed ---")

    def _check_run_status(self):
        """Check if the analysis has been completed."""
        if not self._is_run_complete:
            print("Notice: The analysis has not yet been executed. Calling .run() automatically...")
            self.run()

    def query_donors(self, frames=None):
        """
        See the total number of HB donated per frame.
        
        :param frames: (optional)
            - None (default): Returns a list with the totals for all frames..
            - int: Returns the total for the specified frame index..
            - list[int]: Rreturns a list of totals for the specified frames..
        :return: int ou list[int]
        """
        self._check_run_status()
        
        if frames is None:
            return self.total_donations_per_frame
        
        try:
            if isinstance(frames, int):
                return self.total_donations_per_frame[frames]
            elif isinstance(frames, list):
                return [self.total_donations_per_frame[f] for f in frames]
        except IndexError:
            raise IndexError(f"ID frame outside the interval. Total of frames: {len(self.total_donations_per_frame)}")
        except TypeError:
            raise TypeError(f"The 'frames' argument must be None, int or list[int].")

    def query_acceptors(self, frames=None):
        """
        See the total number of HB accepted per frame
        
        :param frames: (optional)
            - None (default): Returns a list with the totals for all frames..
            - int: Returns the total for the specified frame index..
            - list[int]: Rreturns a list of totals for the specified frames..
        :return: int ou list[int]
        """
        self._check_run_status()
        
        if frames is None:
            return self.total_acceptances_per_frame
        
        try:
            if isinstance(frames, int):
                return self.total_acceptances_per_frame[frames]
            elif isinstance(frames, list):
                return [self.total_acceptances_per_frame[f] for f in frames]
        except IndexError:
            raise IndexError(f"ID frame outside the interval. Total of frames: {len(self.total_acceptances_per_frame)}")
        except TypeError:
            raise TypeError(f"The 'frames' argument must be None, int or list[int].")

    def get_atom_resolved_data(self, frames=None):
        """
        See the detailed data PER ATOM.
        
        :param frames: (optional)
            - None (default): Returns a complete list of results (list of dictionaries).
            - int: Returns the dictionaries of results to the specified frame.
            - list[int]: Returns a list of dictionaries to all the specified frames.
        :return: dict ou list[dict]
        """
        self._check_run_status()
        
        if frames is None:
            return self.results_per_frame
        
        try:
            if isinstance(frames, int):
                return self.results_per_frame[frames]
            elif isinstance(frames, list):
                return [self.results_per_frame[f] for f in frames]
        except IndexError:
            raise IndexError(f"ID frame outside the range. Total number of frames: {len(self.results_per_frame)}")
        except TypeError:
            raise TypeError(f"The 'frames' argument must be None, int or list[int].")

    # --- NOVAS FUNÇÕES ---
    
    def query_number_donors(self, frames=None):
        """
        See the TOTAL NUMBER of donor atoms per frame
        (Based on 'donor_name' provided in the inicialization)
        
        :param frames: (optional)
            - None (default): Returns a list with the total number of atoms in all the frames.
            - int: Retorns the total number to the specified frame ID.
            - list[int]: Returns a list of the total number to the specified frames.
        :return: int or list[int]
        """
        self._check_run_status()
        
        if frames is None:
            return self.number_of_donors_per_frame
        
        try:
            if isinstance(frames, int):
                return self.number_of_donors_per_frame[frames]
            elif isinstance(frames, list):
                return [self.number_of_donors_per_frame[f] for f in frames]
        except IndexError:
            raise IndexError(f"ID frame outside the range. Total number of frames: {len(self.number_of_donors_per_frame)}")
        except TypeError:
            raise TypeError(f"The 'frames' argument must be None, int or list[int].")

    def query_number_acceptors(self, frames=None):
        """
        See the TOTAL NUMBER of acceptor atoms per frame.
        (Based on 'acceptor_name' provided in the inicialization)
        
        :param frames: (optional)
            - None (padrão): Returns a list with the total number of atoms in all the frames.
            - int: Retorns the total number to the specified frame ID.
            - list[int]: Returns a list of the total number to the specified frames.
        :return: int or list[int]
        """
        self._check_run_status()
        
        if frames is None:
            return self.number_of_acceptors_per_frame
        
        try:
            if isinstance(frames, int):
                return self.number_of_acceptors_per_frame[frames]
            elif isinstance(frames, list):
                return [self.number_of_acceptors_per_frame[f] for f in frames]
        except IndexError:
            raise IndexError(f"ID frame outside the range. Total number of frames: {len(self.number_of_acceptors_per_frame)}")
        except TypeError:
            raise TypeError(f"The 'frames' argument must be None, int or list[int].")
