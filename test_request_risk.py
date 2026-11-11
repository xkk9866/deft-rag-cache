"""Regression checks for probabilities and the IPW estimating identity."""
import unittest
from request_risk import certification_probability

class RiskTests(unittest.TestCase):
    def test_negative_budget_forces_certification_with_zero_ipw(self):
        rho=certification_probability(.2,-.04,.05)
        self.assertEqual(rho,1.)
        self.assertEqual((1-rho)/rho,0.)

    def test_expected_ipw_equals_expected_stale_serve(self):
        for p in [.001,.05,.2,1.]:
            for a in [-.2,0.,.01,.5,1.]:
                rho=certification_probability(p,a,.05)
                self.assertTrue(.05 <= rho <= 1.)
                # Enumerate both Bernoulli outcomes for a stale request.
                expected_ipw=rho*(1-rho)/rho
                expected_stale=(1-rho)
                self.assertAlmostEqual(expected_ipw,expected_stale)

    def test_positive_budget_rule_and_zero_oracle_risk(self):
        self.assertAlmostEqual(certification_probability(.2,.01,.05),.95)
        self.assertEqual(certification_probability(.01,.5,.05),.05)
        self.assertEqual(certification_probability(0.,.01,.05),.05)

if __name__=='__main__': unittest.main()
