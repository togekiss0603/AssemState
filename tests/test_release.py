import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce_tables import audit_run, reproduce
from prepare_data import check


class RecordAuditTests(unittest.TestCase):
    def test_all_archived_runs_and_context_alignment(self):
        report = reproduce(ROOT / 'artifacts/pose')
        self.assertEqual(sum(r['operations'] for r in report['runs']), 1701)
        self.assertEqual([r['accepted'] for r in report['runs']], [0, 0, 13, 2, 11, 28, 1])
        self.assertEqual([r['stable'] for r in report['runs']], [40, 90, 96, 30, 82, 82, 109])
        self.assertEqual(report['context_matched_operations']['gpt_structured__gpt_selfcheck'], 160)

    def test_corrupt_release_label_is_rejected(self):
        source = ROOT / 'artifacts/pose/gpt_selfcheck'
        pose = json.loads((source / 'pose.json').read_text())
        release = json.loads((source / 'release.json').read_text())
        release['operations'][0]['release_stable'] = not release['operations'][0]['release_stable']
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'gpt_selfcheck'
            dest.mkdir()
            (dest / 'pose.json').write_text(json.dumps(pose))
            (dest / 'release.json').write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, 'release threshold'):
                audit_run(Path(tmp), 'gpt_selfcheck')

    def test_duplicate_operation_is_rejected(self):
        source = ROOT / 'artifacts/pose/gpt_selfcheck'
        pose = json.loads((source / 'pose.json').read_text())
        pose['operations'][1] = copy.deepcopy(pose['operations'][0])
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'gpt_selfcheck'
            dest.mkdir()
            (dest / 'pose.json').write_text(json.dumps(pose))
            (dest / 'release.json').write_bytes((source / 'release.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'pose operation count'):
                audit_run(Path(tmp), 'gpt_selfcheck')

    @unittest.skipUnless((ROOT / 'examples/data/main_data.json').is_file(), 'Optional data fixture is not publicly distributed')
    def test_demo_asset_coverage(self):
        root = ROOT / 'examples/data'
        self.assertEqual(check(root, json.loads((root / 'main_data.json').read_text())), [])

    def test_dataset_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid product path'):
            check(ROOT, [{'category': '..', 'name': 'outside'}])


class AgentBoundaryTests(unittest.TestCase):
    def test_missing_key_never_silently_runs_mock(self):
        from pose_estimation.visual_agent.pipeline import VisualAssemblyAgent, VisualAssemblyAgentConfig
        agent = VisualAssemblyAgent.__new__(VisualAssemblyAgent)
        agent.config = VisualAssemblyAgentConfig()
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'OPENAI_API_KEY'):
                agent._make_client()
        agent.config.dry_run = True
        self.assertTrue(agent._make_client().is_mock)

    @unittest.skipUnless((ROOT / 'examples/data/main_data.json').is_file() and (ROOT / 'examples/stage_outputs/Bench/applaro/stage1_output.json').is_file(), 'Optional data fixture is not publicly distributed')
    def test_default_target_free_agent_does_not_load_target_meshes(self):
        from pose_estimation.visual_agent import pipeline
        config = pipeline.VisualAssemblyAgentConfig(dry_run=True)
        with patch.object(pipeline, 'load_part_meshes', wraps=pipeline.load_part_meshes) as load:
            agent = pipeline.VisualAssemblyAgent(config)
        self.assertEqual(agent.target_parts, {})
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.args[0], agent.inputs.scene_parts_dir)

    def test_nested_tree_preserves_hierarchy_and_ignores_sibling_order(self):
        sys.path.insert(0, str(ROOT / 'tree/vlm_planner/inference'))
        from run_anchor import are_nested_lists_equal
        self.assertTrue(are_nested_lists_equal([[0, 1], 2], [2, [1, 0]]))
        self.assertFalse(are_nested_lists_equal([[0, 1], 2], [0, [1, 2]]))


if __name__ == '__main__':
    unittest.main()
