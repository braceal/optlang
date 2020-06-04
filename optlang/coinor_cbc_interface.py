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

from mip import Model as mipModel
from mip import xsum, maximize, BINARY, INTEGER, CONTINUOUS, OptimizationStatus

log = logging.getLogger(__name__)


_MIP_STATUS_TO_STATUS = {
    OptimizationStatus.CUTOFF: interface.CUTOFF,
    OptimizationStatus.ERROR: interface.ABORTED,
    OptimizationStatus.FEASIBLE: interface.FEASIBLE,
    OptimizationStatus.INFEASIBLE: interface.INFEASIBLE,
    OptimizationStatus.INT_INFEASIBLE: interface.SPECIAL,
    OptimizationStatus.LOADED: interface.LOADED,
    OptimizationStatus.NO_SOLUTION_FOUND: interface.NOFEASIBLE,
    OptimizationStatus.OPTIMAL: interface.OPTIMAL,
    OptimizationStatus.UNBOUNDED: interface.UNBOUNDED
}

# TODO: set sense (goes in objective)


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



@six.add_metaclass(inheritdocstring)
class Configuration(interface.MathematicalProgrammingConfiguration):
    def __init__(self, verbosity=0, tolerance=1e-10, timeout=float('inf'), *args, **kwargs):
        super(Configuration, self).__init__(*args, **kwargs)
        self.verbosity = verbosity
        self.tolerance = tolerance
        self.timeout = timeout

    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value):
        if value not in (0, 1):
            raise ValueError('Invalid verbosity')
        self._verbosity = value

    @property
    def presolve(self):
        return False

    @presolve.setter
    def presolve(self, value):
        if value is not False:
            raise ValueError("The COIN-OR Cbc solver has no presolve capabilities")

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value

@six.add_metaclass(inheritdocstring)
class Model(interface.Model):

    def _configure_model(self):
        self.problem.verbose = self.configuration.verbosity
        self.problem.max_mip_gap_abs = self.configuration.tolerance

    def _initialize_problem(self):
        self.problem = mipModel()

    def _initialize_model_from_problem(self, problem):
        if not isinstance(problem, mipModel):
            raise TypeError("Problem must be an instance of mipModel, not " + repr(type(problem)))
        self.problem = problem

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        for var in variables:
            # TODO: may need to handle obj, column options
            self.problem.add_var(name=var.name,
                                 var_type=_VTYPE_TO_MIP_VTYPE[var.type],
                                 lb=var.lb,
                                 ub=var.ub)

    def _remove_variables(self, variables):
        super(Model, self)._remove_variables(variables)
        for var in variables:
            self.problem.remove(self.problem.var_by_name(var.name))

    def _add_constraints(self, constraints, sloppy=False):
        super(Model, self)._add_constraints(constraints, sloppy=sloppy)

        for constraint in constraints:
            self.model += constraint # TODO: come back after impl constraint class


    def _remove_constraints(self, constraints):
        super(Model, self)._remove_constraints(constraints)
        for cons in constraints:
            self.problem.remove(self.problem.constr_by_name(cons.name))

    def _optimize(self):
        self._configure_model()
        # TODO: could pass in max_nodes, max_solutions, relax by setting them
        #       in configuration
        status = self.problem.optimize(max_seconds=self.configuration.timeout)
        # TODO: process status
        return status

    # TODO: implement after objective class is done
    @interface.Model.objective.setter
    def objective(self, value):
        super(Model, Model).objective.fset(self, value)
        value.problem = None
        if value is None:
            self.problem.objective = {}
        else:
            offset, coefficients, _ = parse_optimization_expression(value)
            self.problem.objective = {var.name: coef for var, coef in coefficients.items()}
            self.problem.offset = offset
            self.problem.direction = value.direction
        value.problem = self



if __name__ == '__main__':
    import pickle

    x1 = Variable('x1', lb=0)
    x2 = Variable('x2', lb=0)
    x3 = Variable('x3', lb=0)

    # Variable tests
    assert x1.name == 'x1'
    assert x1.lb == 0
    assert x1.ub == None
    assert x1.type == 'continuous'


    assert x1.problem == None

    x1.name = 'x1_name_change'

    assert x1.name == 'x1_name_change'
    x1.name = 'x1'
    assert x1.primal == None
    x1.lb = 1
    assert x1.lb == 1
    x1.lb = 0

    # c1 = Constraint(x1 + x2 + x3, lb=-100, ub=100, name='c1')
    # c2 = Constraint(10 * x1 + 4 * x2 + 5 * x3, ub=600, name='c2')
    # c3 = Constraint(2 * x1 + 2 * x2 + 6 * x3, ub=300, name='c3')
    # obj = Objective(10 * x1 + 6 * x2 + 4 * x3, direction='max')
    model = Model(name='Simple model')
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