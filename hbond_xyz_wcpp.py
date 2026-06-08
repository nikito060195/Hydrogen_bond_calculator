# hbond_xyz_wcpp.py
# Python Module for hydrogen bonds analysis
# by trajectory files .xyz, based on H--D--A angle.
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import math
import hbond_core # Motor C++ importado

# =============================================================================
# Calc functions (internal)
# =============================================================================

def _app_minimum_image(r1, r2, xlo, xhi, ylo, yhi, zlo, zhi, px, py, pz):
    box = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
    delta = r2 - r1
    for i, periodic in enumerate([px, py, pz]):
        if periodic:
            while delta[i] >  box[i] / 2.0: delta[i] -= box[i]
            while delta[i] < -box[i] / 2.0: delta[i] += box[i]
    return r1 + delta

def _process_frame_atom_unified(frame_str, donor_name, acceptor_name, hydrogen_name, box_limits, pbc):
    lines = frame_str.strip().split('\n')
    if len(lines) < 2: return [] 
        
    # OTIMIZAÇÃO DE VELOCIDADE: Corta a linha apenas uma vez (evita milhões de operações inúteis)
    lines_filtered = []
    for line in lines[2:]:
        parts = line.split()
        if not parts: continue
        if parts[0] in (acceptor_name, donor_name, hydrogen_name):
            lines_filtered.append(parts)

    results = []
    mol_id = 0
    i = 0
    (lx_min, lx_max, ly_min, ly_max, lz_min, lz_max) = box_limits
    (px, py, pz) = pbc

    while i < len(lines_filtered):
        line = lines_filtered[i]
        atom_name = line[0]

        if atom_name == acceptor_name and atom_name == donor_name:
            mol_id += 1 
            counter_hydrogen = 0
            coords = {"x": float(line[1]), "y": float(line[2]), "z": float(line[3])}
            results.append({"ID": mol_id, "type": 'acceptor', **coords})
            results.append({"ID": mol_id, "type": 'donor', **coords})
            i += 1 
            
            while i < len(lines_filtered) and lines_filtered[i][0] == hydrogen_name:
                h_line = lines_filtered[i]
                counter_hydrogen += 1
                info_h = {
                    "ID": mol_id, "type": f"H{counter_hydrogen}",
                    "x": float(h_line[1]), "y": float(h_line[2]), "z": float(h_line[3])
                }
                results.append(info_h)
                i += 1
        else:
            if atom_name == acceptor_name:
                 mol_id += 1
                 results.append({"ID": mol_id, "type": 'acceptor', "x": float(line[1]), "y": float(line[2]), "z": float(line[3])})
            elif atom_name == donor_name:
                 mol_id += 1
                 results.append({"ID": mol_id, "type": 'donor', "x": float(line[1]), "y": float(line[2]), "z": float(line[3])})
            i += 1
    
    donors_map = {atom["ID"]: np.array([atom["x"], atom["y"], atom["z"]]) for atom in results if atom["type"] == "donor"}

    for h_atom in results:
        if h_atom["type"].startswith("H"):
            h_id = h_atom["ID"] 
            if h_id in donors_map:
                donor_coords = donors_map[h_id]
                h_coords = np.array([h_atom["x"], h_atom["y"], h_atom["z"]])
                dist = np.linalg.norm(h_coords - donor_coords)
                if dist > 1.5: 
                    h_fixed = _app_minimum_image(donor_coords, h_coords, lx_min, lx_max, ly_min, ly_max, lz_min, lz_max, px, py, pz)
                    h_atom["x"], h_atom["y"], h_atom["z"] = h_fixed[0], h_fixed[1], h_fixed[2]
                            
    return results

def _group_single_frame(frame_data):
    if not frame_data: return []
    frame_data.sort(key=lambda x: x['ID'])
    packs = []
    current_id = -1
    current_pack = []
    for atom in frame_data:
        if atom['ID'] != current_id:
            if current_pack: packs.append(current_pack)
            current_pack = [atom]
            current_id = atom['ID']
        else:
            current_pack.append(atom)
    if current_pack: packs.append(current_pack)
    return packs

def _worker_pipeline(args):
    frame_str, donor, acceptor, hydrogen, box, pbc, set_len, set_angle = args
    
    atom_data = _process_frame_atom_unified(frame_str, donor, acceptor, hydrogen, box, pbc)
    
    n_donors = sum(1 for a in atom_data if a['type'] == 'donor')
    n_acceptors = sum(1 for a in atom_data if a['type'] == 'acceptor')
    
    packs_frame = _group_single_frame(atom_data)
    
    donors_list, acceptors_list, hydrogens_list = [], [], []
    for mol_pack in packs_frame:
        mol_id = mol_pack[0]["ID"]
        for atom in mol_pack:
            if atom["type"] == 'donor':
                donors_list.append([mol_id, atom["x"], atom["y"], atom["z"]])
            elif atom["type"] == 'acceptor':
                acceptors_list.append([mol_id, atom["x"], atom["y"], atom["z"]])
            elif atom["type"].startswith("H"):
                hydrogens_list.append([mol_id, atom["x"], atom["y"], atom["z"]])
                
    d_array = np.array(donors_list, dtype=np.float64) if donors_list else np.empty((0, 4))
    a_array = np.array(acceptors_list, dtype=np.float64) if acceptors_list else np.empty((0, 4))
    h_array = np.array(hydrogens_list, dtype=np.float64) if hydrogens_list else np.empty((0, 4))
    
    results = hbond_core.calculate_frame_hbonds(
        d_array, a_array, h_array, list(box), list(pbc), set_len, set_angle
    )
    
    return results, n_donors, n_acceptors

# =============================================================================
# MAIN CLASS OF THE LIBRARY
# =============================================================================

class HBondAnalysis:
    def __init__(self, filename, 
                 donor="O", acceptor="O", hydrogen="H", 
                 box_limits=None, pbc=(False, False, False),
                 set_len=3.5, set_angle=30, n_cpus=None):
        
        if not os.path.exists(filename): raise FileNotFoundError(f"File do not found: {filename}")
            
        self.filename = filename
        self.donor = donor
        self.acceptor = acceptor
        self.hydrogen = hydrogen
        self.set_len = set_len
        self.set_angle = set_angle
        # Usaremos todos os núcleos disponíveis (com uma folga de 1 para não travar o SO)
        self.n_cpus = n_cpus if n_cpus is not None else max(1, os.cpu_count() - 1)

        self.pbc = pbc
        if box_limits:
            if len(box_limits) != 6: raise ValueError("box_limits must have 6 values.")
            self.box_limits = box_limits
        else:
            if any(pbc): raise ValueError("PBC requires box_limits.")
            self.box_limits = self._auto_detect_box()

        self.results_per_frame = []
        self.total_donations_per_frame = []
        self.total_acceptances_per_frame = []
        self.number_of_donors_per_frame = []
        self.number_of_acceptors_per_frame = []
        self._is_run_complete = False

    def _auto_detect_box(self):
        min_x, max_x, min_y, max_y, min_z, max_z = math.inf, -math.inf, math.inf, -math.inf, math.inf, -math.inf
        try:
            with open(self.filename, 'r') as f:
                while True:
                    line1 = f.readline()
                    if not line1: break 
                    try: num_atoms = int(line1.strip())
                    except ValueError: continue
                    f.readline()
                    for _ in range(num_atoms):
                        parts = f.readline().split()
                        if len(parts) >= 4:
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                min_x, max_x = min(min_x, x), max(max_x, x)
                                min_y, max_y = min(min_y, y), max(max_y, y)
                                min_z, max_z = min(min_z, z), max(max_z, z)
                            except: continue
            return (min_x, max_x, min_y, max_y, min_z, max_z)
        except:
            return (0,0,0,0,0,0)

    def _frame_generator(self):
        """Lê o arquivo sob demanda. Um frame por vez."""
        with open(self.filename, 'r') as f:
            while True:
                line_header = f.readline()
                if not line_header: break
                
                try: num_atoms = int(line_header.strip())
                except ValueError: continue
                
                frame_lines = [line_header, f.readline()]
                for _ in range(num_atoms):
                    frame_lines.append(f.readline())
                    
                yield "".join(frame_lines)

    def _arg_generator(self):
        """Acopla os parâmetros fixos em cada string de frame para o multiprocessing."""
        for frame_str in self._frame_generator():
            yield (frame_str, self.donor, self.acceptor, self.hydrogen, self.box_limits, self.pbc, self.set_len, self.set_angle)

    def run(self):
        print(f"Iniciando análise (C++ Otimizado + {self.n_cpus} CPUs em paralelo)...")
        frame_count = 0
        
        # O map consumirá o gerador e enviará em lotes de 5 para os núcleos do processador
        with ProcessPoolExecutor(max_workers=self.n_cpus) as executor:
            for result_dict, n_donors, n_acceptors in executor.map(_worker_pipeline, self._arg_generator(), chunksize=5):
                
                self.results_per_frame.append(result_dict)
                self.number_of_donors_per_frame.append(n_donors)
                self.number_of_acceptors_per_frame.append(n_acceptors)
                
                total_d = sum(counts['donated'] for counts in result_dict.values())
                total_a = sum(counts['accepted'] for counts in result_dict.values())
                self.total_donations_per_frame.append(total_d)
                self.total_acceptances_per_frame.append(total_a)
                
                frame_count += 1
                if frame_count % 50 == 0:
                    print(f"Frames processados: {frame_count}...")
                
        self._is_run_complete = True
        print(f"Análise concluída com sucesso! Total de frames: {frame_count}")

    def _check_run_status(self):
        if not self._is_run_complete: self.run()

    def query_donors(self, frames=None):
        self._check_run_status()
        if frames is None: return self.total_donations_per_frame
        if isinstance(frames, int): return self.total_donations_per_frame[frames]
        return [self.total_donations_per_frame[f] for f in frames]

    def query_acceptors(self, frames=None):
        self._check_run_status()
        if frames is None: return self.total_acceptances_per_frame
        if isinstance(frames, int): return self.total_acceptances_per_frame[frames]
        return [self.total_acceptances_per_frame[f] for f in frames]

    def get_atom_resolved_data(self, frames=None):
        self._check_run_status()
        if frames is None: return self.results_per_frame
        if isinstance(frames, int): return self.results_per_frame[frames]
        return [self.results_per_frame[f] for f in frames]

    def query_number_donors(self, frames=None):
        self._check_run_status()
        if frames is None: return self.number_of_donors_per_frame
        if isinstance(frames, int): return self.number_of_donors_per_frame[frames]
        return [self.number_of_donors_per_frame[f] for f in frames]

    def query_number_acceptors(self, frames=None):
        self._check_run_status()
        if frames is None: return self.number_of_acceptors_per_frame
        if isinstance(frames, int): return self.number_of_acceptors_per_frame[frames]
        return [self.number_of_acceptors_per_frame[f] for f in frames]
