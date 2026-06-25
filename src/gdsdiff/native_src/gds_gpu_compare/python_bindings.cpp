#include "core.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_gds_compare_core, m) {
    m.doc() = "Optional native core helpers for gds_tools_tk.gpu_compare";
    m.def("floor_div", &gds_gpu_compare::floor_div, py::arg("value"), py::arg("divisor"));
    m.def(
        "bbox_tile_range",
        [](std::int64_t x0, std::int64_t y0, std::int64_t x1, std::int64_t y1, std::int64_t tile_size) {
            const auto bbox = gds_gpu_compare::bbox_tile_range(gds_gpu_compare::BBox{x0, y0, x1, y1}, tile_size);
            return py::make_tuple(bbox.x0, bbox.y0, bbox.x1, bbox.y1);
        },
        py::arg("x0"),
        py::arg("y0"),
        py::arg("x1"),
        py::arg("y1"),
        py::arg("tile_size"));
}
