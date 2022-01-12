from vega.common.class_factory import ClassFactory
from .search_algorithm import SearchAlgorithm
from .ea_conf import EAConfig
from .pareto_front_conf import ParetoFrontConfig
from .pareto_front import ParetoFront


ClassFactory.lazy_register("vega.core.search_algs", {
    "ps_differential": ["DifferentialAlgorithm"],
})
