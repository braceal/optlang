# Copyright (c) 2013 Novo Nordisk Foundation Center for Biosustainability, DTU.
# See LICENSE for details.

import unittest
import json
import os
import optlang.interface
import pickle
import copy
import sys

try:
    import mip
except ImportError as e:
    class TestMissingDependency(unittest.TestCase):

        @unittest.skip('Missing dependency - ' + str(e))
        def test_fail(self):
            pass

    sys.exit(0)

from optlang import coinor_cbc_interface
from optlang.tests import abstract_test_cases

TESTMODELPATH = os.path.join(os.path.dirname(__file__), 'data/coli_core.json')


class VariableTestCase(abstract_test_cases.AbstractVariableTestCase):
    interface = coinor_cbc_interface

    def test_get_primal(self):
        self.assertEqual(self.var.primal, None)
        with open(TESTMODELPATH) as infile:
            model = self.interface.Model.from_json(json.load(infile))

        model.optimize()
        self.assertEqual(model.status, optlang.interface.OPTIMAL)
        for var in model.variables:
            self.assertTrue(var.lb <= round(var.primal, 6) <= var.ub, (var.lb, var.primal, var.ub))

    @unittest.skip("COIN-OR Cbc doesn't support variable name change")
    def test_changing_variable_names_is_reflected_in_the_solver(self):
        pass

    @unittest.skip("COIN-OR Cbc doesn't support variable name change")
    def test_change_name(self):
        pass

    def test_set_wrong_type_raises(self):
        self.assertRaises(ValueError, self.interface.Variable, name="test", type="mayo")
        self.assertRaises(Exception, setattr, self.var, 'type', 'ketchup')
        self.model.add(self.var)
        self.model.update()
        self.assertRaises(ValueError, setattr, self.var, "type", "mustard")

    def test_change_type(self):
        self.var.type = "continuous"
        self.assertEqual(self.var.lb, None)
        self.assertEqual(self.var.ub, None)
        self.var.type = "integer"
        self.assertEqual(self.var.lb, None)
        self.assertEqual(self.var.ub, None)
        self.var.type = "binary"
        self.assertEqual(self.var.lb, 0)
        self.assertEqual(self.var.ub, 1)
        self.var.type = "integer"
        self.assertEqual(self.var.lb, 0)
        self.assertEqual(self.var.ub, 1)
        self.var.type = "continuous"
        self.assertEqual(self.var.lb, 0)
        self.assertEqual(self.var.ub, 1)
        self.var.lb = -1.4
        self.var.ub = 1.6
        self.var.type = "integer"
        self.assertEqual(self.var.lb, -1)
        self.assertEqual(self.var.ub, 2)


class ConstraintTestCase(abstract_test_cases.AbstractConstraintTestCase):
    interface = coinor_cbc_interface

    def test_get_primal(self):

        with open(TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))
        self.assertEqual(self.constraint.primal, None)
        self.model.optimize()
        for c in self.model.constraints:
            if c.lb is not None:
                self.assertTrue(c.lb <= round(c.primal, 6))
            if c.ub is not None:
                self.assertTrue(round(c.primal, 6) <= c.ub)

    @unittest.skip("COIN-OR Cbc doesn't support constraint name change")
    def test_change_constraint_name(self):
        pass

    @unittest.skip("NA")
    def test_indicator_constraint_support(self):
        # TODO: investigate this
        pass


class ObjectiveTestCase(abstract_test_cases.AbstractObjectiveTestCase):
    interface = coinor_cbc_interface

    def setUp(self):
        with open(TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))
        self.obj = self.model.objective

    def test_change_direction(self):
        from mip import MAXIMIZE, MINIMIZE
        self.obj.direction = "min"
        self.assertEqual(self.obj.direction, "min")
        self.assertEqual(self.model.problem.sense, MINIMIZE)

        self.obj.direction = "max"
        self.assertEqual(self.obj.direction, "max")
        self.assertEqual(self.model.problem.sense, MAXIMIZE)


class ConfigurationTestCase(abstract_test_cases.AbstractConfigurationTestCase):
    interface = coinor_cbc_interface


class ModelTestCase(abstract_test_cases.AbstractModelTestCase):
    interface = coinor_cbc_interface

    def setUp(self):
        with open(TESTMODELPATH) as infile:
            self.model = self.interface.Model.from_json(json.load(infile))

    def test_pickle_ability(self):
        self.model.optimize()
        value = self.model.objective.value
        pickle_string = pickle.dumps(self.model)
        from_pickle = pickle.loads(pickle_string)
        from_pickle.optimize()
        self.assertAlmostEqual(value, from_pickle.objective.value)
        self.assertEqual([(var.lb, var.ub, var.name, var.type) for var in from_pickle.variables.values()],
                         [(var.lb, var.ub, var.name, var.type) for var in self.model.variables.values()])
        self.assertEqual([(constr.lb, constr.ub, constr.name) for constr in from_pickle.constraints],
                         [(constr.lb, constr.ub, constr.name) for constr in self.model.constraints])

    def test_config_gets_copied_too(self):
        self.assertEqual(self.model.configuration.verbosity, 0)
        self.model.configuration.verbosity = 3
        model_copy = copy.copy(self.model)
        self.assertEqual(model_copy.configuration.verbosity, 3)

    def test_init_from_existing_problem(self):
        self.assertEqual(len(self.model.variables), len(self.model.problem.vars))
        # Divide by 2 because upper and lower constraints are represented seperately
        self.assertEqual(len(self.model.constraints), len(self.model.problem.constrs) / 2)
        self.assertEqual(self.model.variables.keys(),
                         [var.name for var in self.model.problem.vars])
        # Collect _lower and _upper constraints
        constrs= []
        for con in self.model.constraints:
            constrs.append(con.constraint_name(True))
            constrs.append(con.constraint_name(False))

        self.assertEqual(constrs, [constr.name for constr in self.model.problem.constrs])

    def test_add_non_cplex_conform_variable(self):
        var = self.interface.Variable('12x!!@#5_3', lb=-666, ub=666)
        self.model.add(var)
        self.assertTrue(var in self.model.variables.values())
        self.assertEqual(self.model.variables['12x!!@#5_3'].lb, -666)
        self.assertEqual(self.model.variables['12x!!@#5_3'].ub, 666)
        repickled = pickle.loads(pickle.dumps(self.model))
        var_from_pickle = repickled.variables['12x!!@#5_3']
        self.assertTrue(var_from_pickle.name in [var.name for var in self.model.problem.vars])

    @unittest.skip("COIN-OR Cbc doesn't support constraint name change")
    def test_change_constraint_name(self):
        pass

    @unittest.skip("NA")
    def test_clone_model_with_lp(self):
        pass

    def test_change_of_constraint_is_reflected_in_low_level_solver(self):
        x = self.interface.Variable('x', lb=0, ub=1, type='continuous')
        y = self.interface.Variable('y', lb=-181133.3, ub=12000., type='continuous')
        z = self.interface.Variable('z', lb=0., ub=10., type='continuous')
        constr1 = self.interface.Constraint(0.3 * x + 0.4 * y + 66. * z, lb=-100, ub=0., name='test')
        self.model.add(constr1)
        self.model.update()
        self.assertEqual(self.model.problem.constr_by_name('test_lower').rhs, 100)
        self.assertEqual(self.model.problem.constr_by_name('test_upper').rhs, 0)
        constr1.lb = -9
        constr1.ub = 10
        self.assertEqual(self.model.problem.constr_by_name('test_lower').rhs, 9)
        self.assertEqual(self.model.problem.constr_by_name('test_upper').rhs, 10)
        self.model.optimize()
        constr1.lb = -90
        constr1.ub = 100
        self.assertEqual(self.model.problem.constr_by_name('test_lower').rhs, 90)
        self.assertEqual(self.model.problem.constr_by_name('test_upper').rhs, 100)

    def test_constraint_set_problem_to_None_caches_the_latest_expression_from_solver_instance(self):
        x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        constraint = self.interface.Constraint(0.3 * x + 0.4 * y, lb=-100, name='test')
        self.model.add(constraint)
        z = self.interface.Variable('z', lb=2, ub=5, type='integer')
        constraint += 77. * z
        self.model.remove(constraint)
        self.assertEqual(
            (constraint.expression - (0.4 * y + 0.3 * x + 77.0 * z)).expand() - 0, 0
        )
        self.assertEqual(constraint.lb, -100)
        self.assertEqual(constraint.ub, None)

    def test_change_of_objective_is_reflected_in_low_level_solver(self):
        x = self.interface.Variable('x', lb=-83.3, ub=1324422.)
        y = self.interface.Variable('y', lb=-181133.3, ub=12000.)
        objective = self.interface.Objective(0.3 * x + 0.4 * y, name='test', direction='max')
        self.model.objective = objective
        self.model.update()
        grb_x = self.model.problem.var_by_name(x.name)
        grb_y = self.model.problem.var_by_name(y.name)
        expected = {grb_x: 0.3, grb_y: 0.4}

        self.assertEqual(self.model.problem.objective.expr, expected)

        z = self.interface.Variable('z', lb=4, ub=4, type='integer')
        self.model.objective += 77. * z
        self.model.update()
        grb_z = self.model.problem.var_by_name(z.name)
        expected[grb_z] = 77.

        self.assertEqual(self.model.problem.objective.expr, expected)

    def test_change_variable_bounds(self):
        import random
        inner_prob = self.model.problem
        inner_problem_bounds = [(var.lb, var.ub) for var in inner_prob.vars]
        bounds = [(var.lb, var.ub) for var in self.model.variables.values()]
        self.assertEqual(bounds, inner_problem_bounds)
        for var in self.model.variables.values():
            var.ub = random.uniform(var.lb, 1000)
            var.lb = random.uniform(-1000, var.ub)
        self.model.update()
        inner_problem_bounds_new = [(var.lb, var.ub) for var in inner_prob.vars]
        bounds_new = [(var.lb, var.ub) for var in self.model.variables.values()]
        self.assertNotEqual(bounds, bounds_new)
        self.assertNotEqual(inner_problem_bounds, inner_problem_bounds_new)
        self.assertEqual(bounds_new, inner_problem_bounds_new)

    def test_change_constraint_bounds(self):
        constraint = self.model.constraints[0]
        value = 42
        constraint.ub = value
        self.assertEqual(constraint.ub, value)
        constraint.lb = value
        self.assertEqual(constraint.lb, value)
        name = constraint.name
        self.assertEqual(self.model.problem.constr_by_name(name + '_upper').rhs, value)
        self.assertEqual(self.model.problem.constr_by_name(name + '_lower').rhs, -1*value)

    def test_initial_objective(self):
        self.assertIn('BIOMASS_Ecoli_core_w_GAM', self.model.objective.expression.__str__(), )
        self.assertEqual(
            (self.model.objective.expression - (
                1.0 * self.model.variables.BIOMASS_Ecoli_core_w_GAM -
                1.0 * self.model.variables.BIOMASS_Ecoli_core_w_GAM_reverse_712e5)).expand() - 0, 0
        )

    def test_change_objective(self):
        v1, v2 = self.model.variables.values()[0:2]

        self.model.objective = self.interface.Objective(1. * v1 + 1. * v2)
        self.assertIn(v1.name, str(self.model.objective))
        self.assertIn(v2.name, str(self.model.objective))
        self.assertEqual(self.model.objective._expression, 1.*v1 + 1.*v2)

        self.model.objective = self.interface.Objective(v1 + v2)
        self.assertIn(v1.name, str(self.model.objective))
        self.assertIn(v2.name, str(self.model.objective))
        self.assertEqual(self.model.objective._expression, 1.*v1 + 1.*v2)

    def test_iadd_objective(self):
        v2, v3 = self.model.variables.values()[1:3]
        obj_coeff = sorted(self.model.problem.objective.expr.values())
        self.assertEqual(obj_coeff, [-1.0, 1.0])
        self.model.objective += 2. * v2 - 3. * v3
        obj_coeff = sorted(self.model.problem.objective.expr.values())
        self.assertEqual(obj_coeff, [-3.0, -1.0, 1.0, 2.0])

    def test_imul_objective(self):
        self.model.objective *= 2.
        obj_coeff = sorted(self.model.problem.objective.expr.values())
        self.assertEqual(obj_coeff, [-2.0, 2.0])

        v2, v3 = self.model.variables.values()[1:3]

        self.model.objective += 4. * v2 - 3. * v3
        self.model.objective *= 3.
        obj_coeff = sorted(self.model.problem.objective.expr.values())
        self.assertEqual(obj_coeff, [-9.0, -6.0, 6.0, 12.0])

        self.model.objective *= -1
        obj_coeff = sorted(self.model.problem.objective.expr.values())
        self.assertEqual(obj_coeff, [-12.0, -6.0, 6.0, 9.0])

    def test_set_copied_objective(self):
        mip_expr= self.model.problem.objective.expr
        obj_copy = copy.copy(self.model.objective)
        self.model.objective = obj_copy
        self.assertEqual(self.model.objective.direction, "max")
        self.assertEqual(mip_expr, self.model.problem.objective.expr)

    @unittest.skip("NA")
    def test_timeout(self):
        self.model.configuration.timeout = 0
        status = self.model.optimize()
        self.assertEqual(status, 'time_limit')

    @unittest.skip("Not implemented yet")
    def test_set_linear_coefficients_objective(self):
        self.model.objective.set_linear_coefficients({self.model.variables.R_TPI: 666.})
        # self.assertEqual(glp_get_obj_coef(self.model.problem, self.model.variables.R_TPI.index), 666.)

    @unittest.skip("")
    def test_instantiating_model_with_different_solver_problem_raises(self):
        self.assertRaises(TypeError, self.interface.Model, problem='Chicken soup')

    @unittest.skip("Not implemented yet")
    def test_set_linear_coefficients_constraint(self):
        pass

    def test_scipy_coefficient_dict(self):
        x = self.interface.Variable("x")
        c = self.interface.Constraint(2 ** x, lb=0, sloppy=True)
        obj = self.interface.Objective(2 ** x, sloppy=True)
        model = self.interface.Model()
        self.assertRaises(Exception, setattr, model, "objective", obj)
        self.assertRaises(Exception, model._add_constraint, c)

        c = self.interface.Constraint(0, lb=0)
        obj = self.interface.Objective(0)
        model.add(c)
        model.objective = obj
        self.assertEqual(model.optimize(), optlang.interface.OPTIMAL)

    def test_is_integer(self):
        self.skipTest("No integers with scipy")

    def test_integer_variable_dual(self):
        self.skipTest("No duals with scipy")

    def test_integer_constraint_dual(self):
        self.skipTest("No duals with scipy")

    def test_integer_batch_duals(self):
        self.skipTest("No duals with scipy")

    def test_large_objective(self):
        self.skipTest("Quite slow and not necessary")

    def test_binary_variables(self):
        self.skipTest("No integers with scipy")

    def test_implicitly_convert_milp_to_lp(self):
        self.skipTest("No integers with scipy")
