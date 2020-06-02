# Copyright 2013 Novo Nordisk Foundation Center for Biosustainability,
# Technical University of Denmark.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Interface for the GNU Linear Programming Kit (GLPK)

Wraps the GLPK solver by subclassing and extending :class:`Model`,
:class:`Variable`, and :class:`Constraint` from :mod:`interface`.

GLPK is an open source LP solver, with MILP capabilities.
To use GLPK you need to install the 'swiglpk' python package (with pip or from http://github.com/biosustain/swiglpk)
and make sure that 'import swiglpk' runs without error.
"""
import logging

import os
import six

from optlang.util import inheritdocstring, TemporaryFilename
from optlang.expression_parsing import parse_optimization_expression
from optlang import interface
from optlang import symbolics

from mip import Model, xsum, maximize, BINARY, INTEGER, CONTINUOUS

log = logging.getLogger(__name__)

# TODO: get the right types for coinor_cbc
# _COINOR_CBC_STATUS_TO_STATUS = {
#     GLP_UNDEF: interface.UNDEFINED,
#     GLP_FEAS: interface.FEASIBLE,
#     GLP_INFEAS: interface.INFEASIBLE,
#     GLP_NOFEAS: interface.INFEASIBLE,
#     GLP_OPT: interface.OPTIMAL,
#     GLP_UNBND: interface.UNBOUNDED
# }


_MIP_VTYPE_TO_VTYPE = {
    CONTINUOUS: 'continuous',
    INTEGER: 'integer',
    BINARY: 'binary'
}

_VTYPE_TO_MIP_VTYPE = dict(
    [(val, key) for key, val in six.iteritems(_MIP_VTYPE_TO_VTYPE)]
)


@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    # TOD0: implement _coinor_cbc_ methods (may not need some)

    @property
    def _index(self):
        if self.problem is not None:
            i = self.problem._coinor_cbc_find_col(str(self.name))
            if i:
                return i
            raise IndexError(
                "Could not determine column index for variable %s" % self)
        return None

    @interface.Variable.lb.setter
    def lb(self, value):
        interface.Variable.lb.fset(self, value)
        if self.problem is not None:
            self.problem._coinor_cbc_set_col_bounds(self)

    @interface.Variable.ub.setter
    def ub(self, value):
        interface.Variable.ub.fset(self, value)
        if self.problem is not None:
            self.problem._coinor_cbc_set_col_bounds(self)

    def set_bounds(self, lb, ub):
        super(Variable, self).set_bounds(lb, ub)
        if self.problem is not None:
            self.problem._coinor_cbc_set_col_bounds(self)

    @interface.Variable.type.setter
    def type(self, value):
        try:
            coinor_cbc_kind = _VTYPE_TO_MIP_VTYPE[value]
        except KeyError:
            raise ValueError(
                "COIN-OR CBC cannot handle variables of type %s. " % value +
                "The following variable types are available:\n" +
                " ".join(_VTYPE_TO_MIP_VTYPE.keys())
            )

        if self.problem is not None:
            self.problem._coinor_cbc_set_col_kind(self._index, coinor_cbc_kind)
        interface.Variable.type.fset(self, value)

    def _get_primal(self):
        if self.problem._coinor_cbc_is_mip():
            return self.problem._coinor_cbc_mip_col_val(self._index)
        return self.problem._coinor_cbc_get_col_prim(self._index)

    @property
    def primal(self):
        if self.problem is not None:
            return self.problem.problem.get_var_primal(self.name)
        return None

    @property
    def dual(self):
        if self.problem is not None:
            return self.problem.problem.get_var_dual(self.name)
        return None

    @interface.Variable.name.setter
    def name(self, value):
        old_name = getattr(self, 'name', None)
        super(Variable, Variable).name.fset(self, value)
        if getattr(self, 'problem', None) is not None:
            self.problem._coinor_cbc_set_col_name(old_name, str(value))



if __name__ == '__main__':
    import pickle

    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)

    assert x1.name == 'x1'
    assert x1.lb == 0
    assert x1.ub == None
    assert x1.type == 'continuous'

    assert x1.problem == None

    x1.name = 'x1_name_change'
    assert x1.name == 'x1_name_change'
    x1.name = 'x1'

    assert x1.primal == None

    # c1 = Constraint(x1 + x2 + x3, lb=-100, ub=100, name='c1')
    # c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600, name='c2')
    # c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300, name='c3')
    # obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    # model = Model(name='Simple model')
    # model.objective = obj
    # model.add([c1, c2, c3])
    # status = model.optimize()
    # print("status:", model.status)
    # print("objective value:", model.objective.value)

    # for var_name, var in model.variables.items():
    #     print(var_name, "=", var.primal)

    # print(model)

    # problem = glp_create_prob()
    # glp_read_lp(problem, None, "tests/data/model.lp")

    # solver = Model(problem=problem)
    # print(solver.optimize())
    # print(solver.objective)

    # import time

    # t1 = time.time()
    # print("pickling")
    # pickle_string = pickle.dumps(solver)
    # resurrected_solver = pickle.loads(pickle_string)
    # t2 = time.time()
    # print("Execution time: %s" % (t2 - t1))

    # resurrected_solver.optimize()
    # print("Halelujah!", resurrected_solver.objective.value)
