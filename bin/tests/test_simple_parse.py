import unittest
import tidy

class TestSimpleParse(unittest.TestCase):
    def test_101m(self):
        pdbid = "101m"
        crystal_conditions = "CRYSTAL SOLVENT CONTENT, VS (%): 60.20 MATTHEWS COEFFICIENT, VM (ANGSTROMS**3/DA): " + \
            "3.09 CRYSTALLIZATION CONDITIONS: 3.0 M AMMONIUM SULFATE, 20 MM TRIS, 1MM EDTA, PH 9.0"
        values_exp = {'ph':9.0, 'matthews':60.2, 'ammonium_sulfate_mm':3000.0, 'tris_mm':20.0, 'vm_a_pwr_da':3.09}
        values = tidy.parseEntry(crystal_conditions, pdbid)
        self.assertEqual(values, values_exp)
