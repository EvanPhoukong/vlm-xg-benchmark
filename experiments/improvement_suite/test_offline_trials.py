import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import fetch_context as context
import pixel_control as pixels


def events():
    common = {'period': 1, 'possession': 2, 'team': {'id': 7}}
    return [{**common, 'index': 1, 'type': {'name': 'Pass'}, 'timestamp': '00:01:00.000',
             'pass': {'cross': True, 'height': {'name': 'High Pass'}}},
            {**common, 'id': 'shot', 'index': 2, 'type': {'name': 'Shot'}, 'timestamp': '00:01:02.000',
             'shot': {'type': {'name': 'Open Play'}, 'outcome': {'name': 'Goal'}, 'statsbomb_xg': .9}}]


class OfflineTests(unittest.TestCase):
    def test_context_ignores_shot_outcome_and_commercial_xg(self):
        a = events(); b = copy.deepcopy(a)
        b[1]['shot']['outcome']['name'] = 'Saved'; b[1]['shot']['statsbomb_xg'] = .01
        self.assertEqual(context.pre_shot_context(a), context.pre_shot_context(b))

    def test_future_pass_does_not_enter_shot_context(self):
        a = events(); expected = context.pre_shot_context(a)
        future = copy.deepcopy(a[0]); future.update(index=3, timestamp='00:01:04.000'); future['pass']['cross'] = False
        self.assertEqual(expected, context.pre_shot_context(a+[future]))

    def test_other_team_or_old_pass_cannot_supply_context(self):
        for edit in ['other_team', 'old']:
            a = events()
            if edit == 'other_team':
                a[0]['team'] = {'id': 9}
            else:
                a[0]['timestamp'] = '00:00:00.000'
            self.assertEqual(context.pre_shot_context(a)[0]['prior_pass'], 0)

    def test_pixel_affine_mapping_and_roles_on_unoccluded_generic_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); transform = pixels.calibration(folder)
            row = pd.Series({'x': 108., 'y': 42., 'ff': [
                {'teammate': False, 'location': [119, 40], 'position': {'name': 'Goalkeeper'}},
                {'teammate': False, 'location': [113, 35], 'position': {'name': 'Center Back'}}]})
            path = folder/'test.png'; pixels.base.render_grid(row, path)
            a = pixels.read_pixels(path, transform)
            np.testing.assert_allclose(list(a['shooter'].values()), [2, 12], atol=.12)
            np.testing.assert_allclose(list(a['goalkeeper'].values()), [0, 1], atol=.12)
            self.assertEqual(len(a['defenders']), 1)
            np.testing.assert_allclose(list(a['defenders'][0].values()), [-5, 7], atol=.12)


if __name__ == '__main__':
    unittest.main()
