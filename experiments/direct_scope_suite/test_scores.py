import unittest
import common as c
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
class ScoreChecks(unittest.TestCase):
    def test_metrics_against_sklearn(self):
        y=[0,1,0,1,0,1]
        for p in [[.1,.8,.2,.7,.3,.9],[.8,.1,.8,.2,.3,.2],[.2,.2,.5,.5,.7,.7]]:
            actual=c.metrics(y,p)
            self.assertAlmostEqual(actual['brier'],brier_score_loss(y,p),places=12)
            self.assertAlmostEqual(actual['log_loss'],log_loss(y,p),places=12)
            self.assertAlmostEqual(actual['auc'],roc_auc_score(y,p),places=12)
if __name__=='__main__': unittest.main()
