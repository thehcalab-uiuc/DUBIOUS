from costfunctions.costfn import CostFunction
from costfunctions.dragancost import DraganCostFunction
from costfunctions.workaround_cost_fn import WorkaroundCostFunction
from costfunctions.avoid_red_costfn import AvoidRedCostFn
from costfunctions.legibility_cost_function import LegibilityCostFunction
from costfunctions.leg_obs_cost import LegObvsCostFunction

COST_FUNCTIONS = {
    'DraganCostFunction': DraganCostFunction,
    'WorkaroundCostFunction': WorkaroundCostFunction,
    'AvoidRedCostFn': AvoidRedCostFn,
    'LegibilityCostFunction': LegibilityCostFunction,
    'LegObvsCostFunction': LegObvsCostFunction
}