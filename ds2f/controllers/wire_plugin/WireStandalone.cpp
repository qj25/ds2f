#include "WireStandalone.h"
#include <stdexcept>
#include <memory>

namespace mujoco {
namespace plugin {
namespace elasticity {

WireStandalone::WireStandalone(uintptr_t m_addr, uintptr_t d_addr, int instance)
    : model_(reinterpret_cast<const mjModel*>(m_addr)),
      data_(reinterpret_cast<mjData*>(d_addr)),
      instance_(instance)
{
    wire_ = Wire::Create(model_, data_, instance_);
    if (!wire_) {
        throw std::runtime_error("Failed to create Wire plugin instance");
    }
}

WireStandalone::~WireStandalone() = default;

void WireStandalone::compute() {
    if (wire_) {
        wire_->Compute(model_, data_, instance_);
    }
}

std::vector<double> WireStandalone::get_qfrc_passive() const {
    if (!data_ || !model_) return {};
    std::vector<double> out(model_->nv);
    for (int i = 0; i < model_->nv; ++i) {
        out[i] = data_->qfrc_passive[i];
    }
    return out;
}

void WireStandalone::get_qfrc_passive_array(int dim_qp, double *qf_pas) {
    if (!data_ || !model_ || dim_qp != model_->nv) return;
    for (int i = 0; i < model_->nv; ++i) {
        qf_pas[i] = data_->qfrc_passive[i];
    }
}

} // namespace elasticity
} // namespace plugin
} // namespace mujoco 