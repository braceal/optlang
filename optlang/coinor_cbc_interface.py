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
from mip import (BINARY, INTEGER, CONTINUOUS, OptimizationStatus,
                 MINIMIZE, MAXIMIZE, xsum)

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

def mip_direction(direction):
    return MAXIMIZE if direction is 'max' else MINIMIZE

def to_float(number, ub=True):
    """Converts None type and sympy.core.numbers.Float to float."""
    if number is not None:
        return float(number)
    if ub:
        return float('inf')
    return -float('inf')

# TODO: take care of _coinor_cbc functions

@six.add_metaclass(inheritdocstring)
class Variable(interface.Variable):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, **kwargs)

    @interface.Variable.lb.setter
    def lb(self, value):
        interface.Variable.lb.fset(self, value)
        if self.problem is not None:
            self.problem._update_var_bounds(self.name, lb=value)

    @interface.Variable.ub.setter
    def ub(self, value):
        interface.Variable.ub.fset(self, value)
        if self.problem is not None:
            self.problem._update_var_bounds(self.name, ub=value)

    def set_bounds(self, lb, ub):
        super(Variable, self).set_bounds(lb, ub)
        if self.problem is not None:
            self.problem._update_var_bounds(self.name, lb=lb, ub=ub)

    @interface.Variable.type.setter
    def type(self, value):
        if not value in _VTYPE_TO_MIP_VTYPE:
            raise ValueError(
                'COIN-OR CBC cannot handle variables of type %s. ' % value +
                'The following variable types are available:\n' +
                ' '.join(_VTYPE_TO_MIP_VTYPE.keys()))

        if self.problem is not None:
            self.problem._update_var_type(self.name, _VTYPE_TO_MIP_VTYPE[value])
        interface.Variable.type.fset(self, value)

        if value == 'integer':
            self.lb, self.ub = round(self.lb), round(self.ub)
        elif value == 'binary':
            self.lb, self.ub = 0, 1

    @property
    def primal(self):
        if self.problem is None:
            return None
        return self.problem._var_primal(self.name)

    @property
    def dual(self):
        if self.problem is None:
            return None
        return self.problem._var_dual(self.name)

    @interface.Variable.name.setter
    def name(self, value):
        old_name = getattr(self, 'name', None)
        super(Variable, Variable).name.fset(self, value)
        if getattr(self, 'problem', None) is not None:
            # TODO: currently not supported
            self.problem._update_var_name(old_name, self)


@six.add_metaclass(inheritdocstring)
class Objective(interface.Objective):
    def __init__(self, expression, sloppy=False, **kwargs):
        super(Objective, self).__init__(expression, sloppy=sloppy, **kwargs)
        if not (sloppy or self.is_Linear):
            raise ValueError(
                'COIN-OR Cbc only supports linear objectives. %s is not linear.' % self)

    @property
    def value(self):
        if getattr(self, 'problem', None) is None:
            return None
        return self.problem.problem.objective_value

    @interface.Objective.direction.setter
    def direction(self, value):
        super(Objective, self.__class__).direction.fset(self, value)
        if getattr(self, 'problem', None) is not None:
            self.problem.problem.sense = mip_direction(value.direction)

    def set_linear_coefficients(self, coefficients):
        if self.problem is None:
            raise Exception('Can\'t change coefficients if objective is not associated with a model.')

        self.problem.update()

        # Update Objective member variable
        coeffs = self._expression.as_coefficients_dict()
        coeffs.update(coefficients)
        self._expression = symbolics.add(*(var * coef for var, coef in coeffs.items()))


        # Update corresponding model

        # Function returning mip.Var object given the var name
        # var_by_name = self.problem.problem.var_by_name
        # coeffs = {var_by_name(var.name): coef for var, coef in coefficients.items()}
        # self.problem.problem.objective.set_expr(coeffs)

        # TODO: consider using something similar to above code which may be faster.
        #       it currently has issues with offset var being of type int. May
        #       have other issues, see if there is an update_expr

        self.problem.objective = self


    def get_linear_coefficients(self, variables):
        if self.problem is None:
            raise Exception('Can\'t get coefficients from solver if objective is not in a model')

        self.problem.update()

        # Dictionary {mip.Var: coefficient}
        mip_coeffs = self.problem.problem.objective.expr
        # Function returning mip.Var object given the var name
        var_by_name = self.problem.problem.var_by_name

        return {var: float(mip_coeffs[var_by_name(var.name)]) for var in variables}


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
        if value:
            raise ValueError('The COIN-OR Cbc solver has no presolve capabilities')

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
        # problem has sense attribute
        self.problem = mipModel()

    def _initialize_model_from_problem(self, problem):
        if not isinstance(problem, mipModel):
            raise TypeError('Problem must be an instance of mipModel, not ' + repr(type(problem)))
        self.problem = problem

    def _var_primal(self, name):
        """Used by Variable class."""
        return self.problem.var_by_name(name).x

    # TODO: implement _var_dual
    def _var_dual(self, name):
        """Used by Variable class."""
        return None

    # TODO: implement _update_var_name
    def _update_var_name(self, old_name, var):
        """
        Steps

        1. find var in objective. copy the coefficient and then remove the old key
           and add a new key with the new name

        2. call _remove_variables, and then re add the variable with the new name
        """
        #self.update() # TODO: may not be necessary

        # TODO: does removing vars, remove them from the objective or constraints?
        #       if so we need to get the coefficient first
        #self._remove_variables([var])

        pass


    def _update_var_bounds(self, name, ub=None, lb=None):
        """Used by Variable class."""
        var = self.problem.var_by_name(name)
        if ub is not None:
            var.ub = to_float(ub, True)
        if lb is not None:
            var.lb = to_float(lb, False)

    def _update_var_type(self, name, var_type):
        """Used by Variable class."""
        self.problem.var_by_name(name).var_type = var_type

    def _add_variables(self, variables):
        super(Model, self)._add_variables(variables)
        for var in variables:
            # TODO: may need to handle obj, column options
            self.problem.add_var(name=var.name,
                                 var_type=_VTYPE_TO_MIP_VTYPE[var.type],
                                 lb=to_float(var.lb, False),
                                 ub=to_float(var.ub, True))

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
        # TODO: could set self.problem.threads = -1 to use all available cores
        # TODO: could pass in max_nodes, max_solutions, relax by setting them
        #       in configuration
        status = self.problem.optimize(max_seconds=self.configuration.timeout)

        # TODO: make more robust. See glpk_interface.py
        #       handle INT_INFEASIBLE case. mip.Model has a relax method that
        #       changes all integer and binary variable types to continuous.
        #       if we call this, make sure to update the associated Variable objects.

        return _MIP_STATUS_TO_STATUS[status]

    @interface.Model.objective.setter
    def objective(self, value):
        super(Model, Model).objective.fset(self, value)
        self.update() # update to get new vars

        offset, coeffs, _ = parse_optimization_expression(value)

        self.problem.objective = offset + xsum(to_float(coef) * self.problem.var_by_name(var.name)
                                               for var, coef in coeffs.items())

        self.problem.sense = mip_direction(value.direction)
        value.problem = self


if __name__ == '__main__':
    import pickle

    x1 = Variable('x1', lb=0, ub=5)
    x2 = Variable('x2', lb=0, ub=5)
    x3 = Variable('x3', lb=0, ub=5)

    # Variable tests
    assert x1.name == 'x1'
    assert x1.lb == 0
    assert x1.ub == 5
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
    obj = Objective(10 * x1 + 6 * x2 + 4 * x3 + 1, direction='max')

    assert obj.value == None
    assert obj.direction == 'max'

    model = Model(name='Simple model')
    model.objective = obj

    assert obj.get_linear_coefficients([x1, x2, x3]) == {x1: 10, x2: 6, x3: 4}

    print(obj)

    obj.set_linear_coefficients({x1: 11., x2: 6., x3: 4., 1:10})

    print(obj)

    x1.lb = 1
    x2.ub = 6
    x1.ub = 4.9

    x3.set_bounds(2, 6)

    # integer types rounds constraints x1.ub to nearest integer (5)
    x1.type = 'integer'
    x3.type = 'binary'

    assert x3.lb == 0 and x3.ub == 1
    assert x1.lb == 1 and x1.ub == 5

    # model.add([c1, c2, c3])
    status = model.optimize()
    print('status:', model.status)
    print('objective value:', model.objective.value)
    assert model.objective.value == 105.0

    for var_name, var in model.variables.items():
        print(var_name, '=', var.primal)

    print(model)
