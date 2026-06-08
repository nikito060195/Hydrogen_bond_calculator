#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <unordered_map>

namespace py = pybind11;
using namespace pybind11::literals;

// Função auxiliar para aplicar PBC (Convenção de Imagem Mínima)
inline double apply_pbc(double delta, double box_length, bool pbc_flag) {
    if (pbc_flag) {
        while (delta >  box_length / 2.0) delta -= box_length;
        while (delta < -box_length / 2.0) delta += box_length;
    }
    return delta;
}

// Função principal de cálculo (Processa 1 frame)
py::dict calculate_frame_hbonds(
    py::array_t<double> donors,      
    py::array_t<double> acceptors,   
    py::array_t<double> hydrogens,   
    std::vector<double> box,         
    std::vector<bool> pbc,           
    double set_len,
    double set_angle
) {
    auto d_data = donors.unchecked<2>();
    auto a_data = acceptors.unchecked<2>();
    auto h_data = hydrogens.unchecked<2>();

    double box_x = box[1] - box[0];
    double box_y = box[3] - box[2];
    double box_z = box[5] - box[4];

    double set_len_sq = set_len * set_len;
    double cos_set_angle = std::cos(set_angle * M_PI / 180.0);

    // Usa um dicionário nativo do C++ para evitar problemas com o Python
    // Chave: ID da molécula -> Valor: Par <Doações, Aceitações>
    std::unordered_map<int, std::pair<int, int>> counts;

    // Inicialização
    for (py::ssize_t i = 0; i < d_data.shape(0); i++) {
        counts[static_cast<int>(d_data(i, 0))] = {0, 0};
    }
    for (py::ssize_t i = 0; i < a_data.shape(0); i++) {
        int a_id = static_cast<int>(a_data(i, 0));
        if (counts.find(a_id) == counts.end()) {
            counts[a_id] = {0, 0};
        }
    }

    // Loop direto (Sem OpenMP, pois o Python já está rodando múltiplos processos em paralelo)
    for (py::ssize_t i = 0; i < d_data.shape(0); i++) {
        int d_id = static_cast<int>(d_data(i, 0));
        double dx = d_data(i, 1), dy = d_data(i, 2), dz = d_data(i, 3);

        for (py::ssize_t j = 0; j < a_data.shape(0); j++) {
            int a_id = static_cast<int>(a_data(j, 0));
            if (d_id == a_id) continue;

            double diff_x = apply_pbc(a_data(j, 1) - dx, box_x, pbc[0]);
            double diff_y = apply_pbc(a_data(j, 2) - dy, box_y, pbc[1]);
            double diff_z = apply_pbc(a_data(j, 3) - dz, box_z, pbc[2]);
            
            double dist_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
            
            if (dist_sq <= set_len_sq) {
                for (py::ssize_t k = 0; k < h_data.shape(0); k++) {
                    if (static_cast<int>(h_data(k, 0)) == d_id) {
                        double hx = h_data(k, 1), hy = h_data(k, 2), hz = h_data(k, 3);
                        
                        double ba_x = apply_pbc(hx - dx, box_x, pbc[0]);
                        double ba_y = apply_pbc(hy - dy, box_y, pbc[1]);
                        double ba_z = apply_pbc(hz - dz, box_z, pbc[2]);
                        
                        double dot_product = ba_x*diff_x + ba_y*diff_y + ba_z*diff_z;
                        double norm_ba_sq = ba_x*ba_x + ba_y*ba_y + ba_z*ba_z;
                        
                        if (norm_ba_sq > 0 && dist_sq > 0) {
                            double cos_theta = dot_product / std::sqrt(norm_ba_sq * dist_sq);
                            if (cos_theta >= cos_set_angle) {
                                counts[d_id].first += 1;   // Doou
                                counts[a_id].second += 1;  // Aceitou
                            }
                        }
                    }
                }
            }
        }
    }

    // No final, converte os resultados para o formato py::dict de forma 100% segura
    py::dict mol_hb_counts;
    for (auto const& pair : counts) {
        mol_hb_counts[py::cast(pair.first)] = py::dict(
            "donated"_a = pair.second.first, 
            "accepted"_a = pair.second.second
        );
    }

    return mol_hb_counts;
}

PYBIND11_MODULE(hbond_core, m) {
    m.def("calculate_frame_hbonds", &calculate_frame_hbonds, "Calcula HBs do frame");
}
