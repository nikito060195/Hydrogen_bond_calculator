# hbond_xyz_wcpp.py
# Python Module for hydrogen bonds analysis
# by trajectory files .xyz and .lammpstrj, based on H--D--A angle.
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import math
import re
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

def _process_frame_atom_unified(frame_str, donor_name, acceptor_name, hydrogen_name, box_limits, pbc, file_format):
    lines = frame_str.strip().split('\n')
    if len(lines) < 2: return [] 
        
    lines_filtered = []
    
    # --- NOVO: Extração Padronizada baseada no formato ---
    if file_format == 'lammpstrj':
        start_idx = 0
        for idx, l in enumerate(lines):
            if l.startswith("ITEM: ATOMS"):
                start_idx = idx + 1
                break
        
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) >= 5:
                atom_type = parts[1] # Em lammpstrj: id(0) type(1) x(2) y(3) z(4)
                if atom_type in (acceptor_name, donor_name, hydrogen_name):
                    lines_filtered.append([atom_type, parts[2], parts[3], parts[4]])
    else: # xyz
        for line in lines[2:]:
            parts = line.split()
            if not parts: continue
            atom_type = parts[0] # Em xyz: type(0) x(1) y(2) z(3)
            if atom_type in (acceptor_name, donor_name, hydrogen_name):
                lines_filtered.append([atom_type, parts[1], parts[2], parts[3]])
    # ---------------------------------------------------

    results = []
    mol_id = 0
    i = 0
    (lx_min, lx_max, ly_min, ly_max, lz_min, lz_max) = box_limits
    (px, py, pz) = pbc

    # O resto da lógica flui normalmente, pois lines_filtered agora tem o formato padrão [type, x, y, z]
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
    frame_str, donor, acceptor, hydrogen, box, pbc, set_len, set_angle, file_format = args
    
    atom_data = _process_frame_atom_unified(frame_str, donor, acceptor, hydrogen, box, pbc, file_format)
    
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
    def __init__(self, filename, file_format=None,
                 donor="O", acceptor="O", hydrogen="H", 
                 box_limits=None, pbc=None,
                 set_len=3.5, set_angle=30, n_cpus=None):
        
        if not os.path.exists(filename): raise FileNotFoundError(f"File do not found: {filename}")
            
        self.filename = filename
        self.donor = donor
        self.acceptor = acceptor
        self.hydrogen = hydrogen
        self.set_len = set_len
        self.set_angle = set_angle
        self.n_cpus = n_cpus if n_cpus is not None else max(1, os.cpu_count() - 1)

        # Detecta formato pela extensão se não fornecido
        self.file_format = file_format
        if not self.file_format:
            if filename.endswith('.lammpstrj'): self.file_format = 'lammpstrj'
            else: self.file_format = 'xyz'
            
        # Extração inteligente da Box e PBC
        parsed_box, parsed_pbc = self._parse_header()
        
        self.pbc = pbc if pbc is not None else (parsed_pbc if parsed_pbc is not None else (False, False, False))
        
        if box_limits is not None:
            if len(box_limits) != 6: raise ValueError("box_limits must have 6 values.")
            self.box_limits = box_limits
        elif parsed_box is not None:
            self.box_limits = parsed_box
            print(f"Info: Configuração extraída do arquivo -> Box: {self.box_limits} | PBC: {self.pbc}")
        else:
            print("WARNING: Arquivo simples sem info de Lattice/Box e 'box_limits' não foi fornecido.")
            print("Utilizando _auto_detect_box para inferir limites. CUIDADO: O PBC pode não funcionar adequadamente sem uma caixa exata.")
            if any(self.pbc): print("WARNING: PBC foi ativado, mas as dimensões da caixa são aproximadas!")
            self.box_limits = self._auto_detect_box()

        self.results_per_frame = []
        self.total_donations_per_frame = []
        self.total_acceptances_per_frame = []
        self.number_of_donors_per_frame = []
        self.number_of_acceptors_per_frame = []
        self._is_run_complete = False

    def _parse_header(self):
        """Lê o cabeçalho do arquivo para extrair pbc e box_limits automaticamente."""
        box, pbc = None, None
        with open(self.filename, 'r') as f:
            if self.file_format == 'lammpstrj':
                for _ in range(100): # Lê o começo até achar a info
                    line = f.readline()
                    if not line: break
                    if line.startswith("ITEM: BOX BOUNDS"):
                        # Extrai PBC da linha: ITEM: BOX BOUNDS pp pp ff
                        parts = line.split()[3:]
                        pbc = tuple(('p' in p) for p in parts) if len(parts) >= 3 else (True, True, True)
                        
                        # Extrai dimensões da caixa
                        x_line = f.readline().split()
                        y_line = f.readline().split()
                        z_line = f.readline().split()
                        box = (float(x_line[0]), float(x_line[1]), 
                               float(y_line[0]), float(y_line[1]), 
                               float(z_line[0]), float(z_line[1]))
                        break
            else: # xyz estendido ou simples
                f.readline() # pula num de atomos
                line2 = f.readline()
                if not line2: return None, None
                
                # Procura Lattice="v1x v1y v1z v2x v2y v2z v3x v3y v3z"
                match_lattice = re.search(r'Lattice="([^"]+)"', line2)
                if match_lattice:
                    v = [float(x) for x in match_lattice.group(1).split()]
                    if len(v) >= 9:
                        # Assumindo caixa ortogonal com início em zero para o script funcionar (padrão XYZ)
                        box = (0.0, v[0], 0.0, v[4], 0.0, v[8])
                
                # Procura pbc="T T F"
                match_pbc = re.search(r'pbc="([^"]+)"', line2)
                if match_pbc:
                    p_flags = match_pbc.group(1).split()
                    pbc = tuple(p.upper() == 'T' for p in p_flags) if len(p_flags) >= 3 else None

        return box, pbc

    def _auto_detect_box(self):
        # Fallback (Mantido do original, apenas para xyz simples)
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
        """Lê o arquivo sob demanda adaptando-se ao formato do arquivo."""
        with open(self.filename, 'r') as f:
            if self.file_format == 'lammpstrj':
                frame_lines = []
                for line in f:
                    if line.startswith("ITEM: TIMESTEP"):
                        if frame_lines:
                            yield "".join(frame_lines)
                            frame_lines = []
                    frame_lines.append(line)
                if frame_lines:
                    yield "".join(frame_lines)
            else:
                # Lógica para XYZ
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
        for frame_str in self._frame_generator():
            # Agora envia o file_format para o worker também
            yield (frame_str, self.donor, self.acceptor, self.hydrogen, 
                   self.box_limits, self.pbc, self.set_len, self.set_angle, self.file_format)

    def run(self):
        print(f"Iniciando análise (Formato: {self.file_format} | C++ + {self.n_cpus} CPUs em paralelo)...")
        frame_count = 0
        
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

    # (Os metódos query_* continuam inalterados abaixo...)
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
