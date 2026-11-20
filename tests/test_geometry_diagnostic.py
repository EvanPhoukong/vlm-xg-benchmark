"""Offline checks for coordinate truth, API isolation, and resumable geometry tests."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import httpx
from openai import OpenAI
import numpy as np
import pandas as pd

import geometry_diagnostic as geo


def answer(**changes):
    values = {'shooter_lateral': -10, 'shooter_depth': 10, 'shot_distance': 14.1,
              'goal_angle_degrees': 24, 'goalkeeper_visible': True,
              'goalkeeper_lateral': -1, 'goalkeeper_depth': 2}
    return {**values, **changes}


def payload(prediction=None, status='completed'):
    return {'id': 'resp_geometry_test', 'object': 'response', 'created_at': 0,
            'model': 'gpt-6-luna', 'status': status,
            'output': [] if status != 'completed' else [{
                'id': 'msg_geometry_test', 'type': 'message', 'role': 'assistant',
                'status': 'completed', 'content': [{'type': 'output_text',
                'text': json.dumps(prediction or answer()), 'annotations': []}]}]}


class GeometryTests(unittest.TestCase):
    def make_client(self, requests, status='completed'):
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=payload(status=status))
        client = OpenAI(api_key='test-key', max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        self.addCleanup(client.close)
        return client

    def test_image_orientation_and_known_geometry(self):
        from mplsoccer import VerticalPitch
        import matplotlib.pyplot as plt
        pitch = VerticalPitch(half=True, pitch_type='statsbomb')
        figure, axis = pitch.draw()
        self.addCleanup(plt.close, figure)
        offsets = pitch.scatter([110], [30], ax=axis).get_offsets()[0]
        self.assertEqual(list(offsets), [30, 110])
        self.assertLess(axis.get_xlim()[0], axis.get_xlim()[1])
        self.assertLess(axis.get_ylim()[0], axis.get_ylim()[1])
        self.assertEqual(geo.relative_coordinates(110, 30), (-10, 10))
        distance, angle = geo.geometry_from_relative(0, 12)
        self.assertEqual(distance, 12)
        self.assertAlmostEqual(angle, math.degrees(2 * math.atan(4 / 12)))

    def test_angle_wraparound_and_missing_measurements(self):
        self.assertEqual(geo.geometry_from_relative(0, 0), (0, 180))
        self.assertEqual(geo.geometry_from_relative(10, 0), (10, 0))
        self.assertTrue(np.isnan(geo.geometry_from_relative(None, 12)[0]))
        self.assertTrue(np.isnan(geo.geometry_from_relative(4, 0)[1]))

    def test_selection_does_not_depend_on_goals_or_model_errors(self):
        first = pd.DataFrame({'shot_id': [f'shot-{i}' for i in range(20)],
                              'goal': np.arange(20) % 2, 'vlm_xg': np.arange(20) / 20})
        changed = first.sample(frac=1, random_state=99).copy()
        changed['goal'] = 1 - changed.goal
        changed['vlm_xg'] = 1 - changed.vlm_xg
        self.assertEqual(geo.select_shots(first, 8, 7).shot_id.tolist(),
                         geo.select_shots(changed, 8, 7).shot_id.tolist())

    def test_response_validation_handles_abstention_and_rejects_bad_numbers(self):
        self.assertFalse(geo.validate_response(answer(goalkeeper_visible=False,
                         goalkeeper_lateral=None, goalkeeper_depth=None))['goalkeeper_visible'])
        for bad in [answer(shot_distance=float('nan')), answer(goal_angle_degrees=181),
                    answer(goalkeeper_visible=False), answer(shooter_depth=True),
                    answer(goalkeeper_depth=None)]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                geo.validate_response(bad)

    def test_image_and_fixed_prompt_are_the_only_model_input(self):
        requests = []
        result, info = geo.ask_geometry(self.make_client(requests), b'original-image-bytes', 'gpt-6-luna')
        self.assertEqual(result, answer())
        body = requests[0]
        self.assertEqual(len(body['input']), 1)
        content = body['input'][0]['content']
        self.assertEqual(content[1], {'type': 'input_text', 'text': geo.PROMPT})
        self.assertEqual(content[0]['detail'], 'high')
        self.assertEqual(body['reasoning'], {'effort': 'low'})
        self.assertFalse(body['store'])
        self.assertEqual(body['text']['format']['schema'], geo.SCHEMA)
        self.assertTrue(body['text']['format']['strict'])
        self.assertEqual(info['returned_model'], 'gpt-6-luna')

    def test_incomplete_response_is_not_scored(self):
        with self.assertRaisesRegex(RuntimeError, 'No complete geometry response'):
            geo.ask_geometry(self.make_client([], status='incomplete'), b'image', 'gpt-6-luna')

    def test_cache_resumes_without_duplicate_requests_and_rejects_changed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image = output / 'shot-a.png'
            image.write_bytes(b'unchanged-original-image')
            reference = pd.DataFrame({'shot_id': ['shot-a']})
            requests = []
            client = self.make_client(requests)
            with contextlib.redirect_stdout(io.StringIO()):
                first = geo.infer_geometry(reference, output, output, 'gpt-6-luna', client)
                second = geo.infer_geometry(reference, output, output, 'gpt-6-luna', client)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len((output / 'geometry_cache.jsonl').read_text().splitlines()), 1)
            image.write_bytes(b'changed-image')
            with self.assertRaisesRegex(ValueError, 'Image changed'):
                geo.infer_geometry(reference, output, output, 'gpt-6-luna', client)
            with self.assertRaisesRegex(ValueError, 'another prompt, model, or settings'):
                geo.load_cache(output / 'geometry_cache.jsonl', 'other-signature')

    def test_errors_separate_geometry_calculation_and_location_and_count_abstentions(self):
        reference = pd.DataFrame([
            {'shot_id': 'a', 'ff': [], 'true_shooter_lateral': 0, 'true_shooter_depth': 12,
             'true_distance': 12, 'true_angle_degrees': geo.geometry_from_relative(0, 12)[1],
             'true_goalkeeper_lateral': 0, 'true_goalkeeper_depth': 2, 'goalkeeper_in_view': True},
            {'shot_id': 'b', 'ff': [], 'true_shooter_lateral': 0, 'true_shooter_depth': 12,
             'true_distance': 12, 'true_angle_degrees': geo.geometry_from_relative(0, 12)[1],
             'true_goalkeeper_lateral': 0, 'true_goalkeeper_depth': 2, 'goalkeeper_in_view': True},
        ])
        predictions = pd.DataFrame([
            {'shot_id': 'a', **answer(shooter_lateral=0, shooter_depth=12, shot_distance=20,
                 goal_angle_degrees=10, goalkeeper_lateral=0, goalkeeper_depth=2)},
            {'shot_id': 'b', **answer(shooter_lateral=0, shooter_depth=12, shot_distance=20,
                 goal_angle_degrees=10, goalkeeper_visible=False,
                 goalkeeper_lateral=None, goalkeeper_depth=None)},
        ])
        frame, metrics = geo.error_tables(reference, predictions)
        metrics = metrics.set_index('column')
        self.assertEqual(metrics.loc['shot_distance', 'mean_absolute_error'], 8)
        self.assertEqual(metrics.loc['computed_distance', 'mean_absolute_error'], 0)
        self.assertEqual(metrics.loc['shooter_position_error', 'mean_absolute_error'], 0)
        self.assertEqual(metrics.loc['goalkeeper_position_error', 'answered'], 1)
        self.assertEqual(metrics.loc['goalkeeper_position_error', 'eligible'], 2)
        self.assertEqual(metrics.loc['goalkeeper_position_error', 'within_tolerance_fraction'], .5)


if __name__ == '__main__':
    unittest.main()
