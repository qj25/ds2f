#ifndef WIRE_STANDALONE_H_
#define WIRE_STANDALONE_H_

#include <vector>
#include <memory>
#include <optional>
#include <cstdint>

#include <mujoco/mjmodel.h>
#include <mujoco/mjdata.h>
#include "wire.h"

namespace mujoco {
namespace plugin {
namespace elasticity {

class WireStandalone {
public:
    WireStandalone(uintptr_t m_addr, uintptr_t d_addr, int instance);
    // WireStandalone(const mjModel* m, mjData* d, int instance);
    ~WireStandalone();

    // Run the plugin's compute function
    void compute();

    // Get the current qfrc_passive as a std::vector (for SWIG/numpy)
    std::vector<double> get_qfrc_passive() const;

    // Output qfrc_passive to a user-provided array (for SWIG/Numpy)
    void get_qfrc_passive_array(int dim_qp, double *qf_pas);

    // Optionally: expose mjData* and mjModel* for advanced users
    mjModel* get_model() const { return const_cast<mjModel*>(model_); }
    mjData* get_data() const { return data_; }

private:
    const mjModel* model_;
    mjData* data_;
    int instance_;
    std::unique_ptr<Wire> wire_;
};

} // namespace elasticity
} // namespace plugin
} // namespace mujoco

#endif // WIRE_STANDALONE_H_ 