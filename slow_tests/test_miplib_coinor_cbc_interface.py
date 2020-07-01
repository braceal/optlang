# Copyright 2014 Novo Nordisk Foundation Center for Biosustainability, DTU.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest

import nose

try:

    import mip

except ImportError as e:

    raise nose.SkipTest('Skipping MILP tests because MIP is not available.')

else:

    from swiglpk import glp_read_mps, GLP_MPS_FILE, glp_create_prob
    from optlang import glpk_interface, coinor_cbc_interface
    def load_problem(mps_file):
        prob_tmp_file = tempfile.mktemp(suffix='.mps')
        with open(prob_tmp_file, 'wb') as tmp_handle:
            f = open(mps_file, 'rb')
            tmp_handle.write(f.read())
            f.close()

        problem = glp_create_prob()
        glp_read_mps(problem, GLP_MPS_FILE, None, prob_tmp_file)
        model = coinor_cbc_interface.Model.clone(glpk_interface.Model(problem=problem))
        model.configuration.presolve = True
        model.configuration.verbosity = 3
        return model

    # problems from http://miplib.zib.de/miplib2003/miplib2003.php

    TRAVIS = os.getenv('TRAVIS', False)
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
    MPS_BENCHMARK_1 = os.path.join(DATA_PATH, '30n20b8.mps')

    def test_mps_benchmark_1():
        if TRAVIS:
            raise nose.SkipTest('Skipping extensive MILP tests on travis-ci.')

        model = load_problem(MPS_BENCHMARK_1)
        model.configuration.threads = -1
        model.optimize()
        self.assertEqual(model.status, 'optimal')
        self.assertEqual(model.objective.value, 302)

if __name__ == '__main__':
    nose.runmodule()
