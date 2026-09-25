"""Focused regression checks for the APD viewer and its optical models."""

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def load_embedded_helpers():
    notebook_path = Path(__file__).with_name("Unit_sphere_viewer.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    names = {
        "fourkas_ABC", "fourkas_ABC_fresnel", "r_max_of", "load_tdms",
        "fit_gains_tmatrix", "block_channel_means", "apply_gains", "apd_tmatrix",
        "apply_tmatrix", "correct_apd_channels", "theta_from_r",
        "extract_radial_excursions", "radial_crossing_profiles",
        "fit_origin_separator", "fit_origin_core_radius", "ellipse_region_clearances",
        "fit_equal_area_ellipses", "radial_region_gaps", "fit_radial_equal_area_ellipses",
        "classify_region_cores", "region_residence_statistics",
        "region_transition_counts",
    }
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            definitions.extend(node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in names)
    if len(definitions) != len(names) or {node.name for node in definitions} != names:
        raise AssertionError("Every shared helper must be defined exactly once in the notebook")
    namespace = {"np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook_path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


ha = load_embedded_helpers()


def notebook_functions(*names):
    notebook_path = Path(__file__).with_name("Unit_sphere_viewer.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            definitions.extend(node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in names)
    if {node.name for node in definitions} != set(names):
        raise AssertionError("Requested notebook helper was not found")
    namespace = {**vars(ha), "np": np, "ha": ha, "minimize": minimize, "RIM_BINS": 90,
                 "RIM_EXTREME_POINTS": 3, "RIM_SECTOR_DEG": (130., 230.)}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook_path), "exec"), namespace)
    return namespace


class FolderScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = Path(__file__).with_name("anisotropy_folder_scan.ipynb")
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cls.helpers = {}
        with patch.dict("sys.modules", {"hinge_analysis": None}):
            for cell in cls.notebook["cells"]:
                if cell["cell_type"] != "code":
                    continue
                tree = ast.parse("".join(cell["source"]))
                tree.body = [node for node in tree.body if not (
                    isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SCAN_RESULTS"
                                                        for target in node.targets))]
                exec(compile(tree, str(notebook_path), "exec"), cls.helpers)
        cls.coefficients = {"full": cls.helpers["full_pupil_coefficients"](1.3, 1.33),
                            "fresnel": cls.helpers["fresnel_coefficients"](1.3, 1.33, 1.515, .38)}

    def test_optics_gains_and_block_data_match_original(self):
        np.testing.assert_allclose(self.coefficients["full"], ha.fourkas_ABC(), rtol=1e-13)
        np.testing.assert_allclose(self.coefficients["fresnel"], ha.fourkas_ABC_fresnel(), rtol=1e-13)
        radii = np.array([0., .1, .7, .9, 1.2, np.nan, -.1])
        for coefficients in self.coefficients.values():
            np.testing.assert_allclose(self.helpers["theta_from_radius"](radii, coefficients),
                                       ha.theta_from_r(radii, *coefficients), equal_nan=True)
        channels = np.random.default_rng(3).uniform(10., 30., (203, 4))
        channels[20, 2] = np.nan
        original = channels.copy()
        recording = {name: channels[:, index] for index, name in enumerate(("c0", "c90", "c45", "c135"))}
        recording.update(t=np.arange(len(channels)) / 250000., fps=250000., n_frames=len(channels))
        corrected, gains = ha.correct_apd_channels(recording)
        actual = self.helpers["prepare_recording"](channels, 250000., 80., self.coefficients)
        np.testing.assert_allclose(actual["gains"], [gains[name] for name in ("0", "90", "45", "135")], rtol=1e-13)
        for name, source, model in (("raw", recording, "full"), ("corrected", corrected, "full"),
                                     ("fresnel", corrected, "fresnel")):
            x_values, y_values, times, rate, flags = ha.block_channel_means(source, 20, return_flags=True)
            stage = actual["stages"][name]
            np.testing.assert_allclose(stage["ax"], x_values, atol=1e-14, equal_nan=True)
            np.testing.assert_allclose(stage["ay"], y_values, atol=1e-14, equal_nan=True)
            np.testing.assert_array_equal(stage["photometry_valid"], flags["valid"])
            expected = np.where(flags["valid"], ha.theta_from_r(np.hypot(x_values, y_values),
                                                              *self.coefficients[model]), np.nan)
            np.testing.assert_allclose(stage["theta"], expected, equal_nan=True, atol=1e-11)
            np.testing.assert_allclose(actual["times"], times, atol=1e-14)
            self.assertEqual(actual["rate"], rate)
        self.assertEqual(actual["trailing_samples"], 3)
        np.testing.assert_array_equal(channels, original)

    def test_anisotropy_plots_use_fixed_square_and_drop_outliers(self):
        points = np.array([[-1., -1.], [1., 1.], [0., 0.], [1., 0.], [0., -1.],
                           [-1.01, 0.], [0., 1.01], [100., 0.], [0., -100.], [np.nan, 0.], [np.inf, 0.]])
        original = points.copy()
        times = np.arange(len(points)) / 12500.
        stages = {name: dict(ax=points[:, 0].copy(), ay=points[:, 1].copy(),
                             photometry_valid=np.all(np.isfinite(points), axis=1),
                             theta=np.full(len(points), 30.), valid_angle=np.ones(len(points), dtype=bool))
                  for name in ("raw", "corrected", "fresnel")}
        outside_circle = dict(center=np.array([4., 5.]), radius=3., edges=np.array([[4., 8.], [7., 5.]]))
        with patch.dict(self.helpers, fit_circle=unittest.mock.Mock(return_value=outside_circle)):
            batch_figures = self.helpers["make_figures"](
                dict(stages=stages, times=times, rate=12500., rmax=.928), "outlier regression")
        for figure in batch_figures:
            self.addCleanup(plt.close, figure)
        notebook_path = Path(__file__).with_name("Unit_sphere_viewer.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        plot_source = next("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code" and "ANISOTROPY_HISTOGRAMS = {}" in "".join(cell["source"]))
        namespace = dict(np=np, plt=plt,
                         STAGES={name: (stage["ax"], stage["ay"], times, 12500.) for name, stage in stages.items()},
                         STAGE_TITLES={name: name for name in stages}, R_MAX_BY_STAGE=dict.fromkeys(stages, .928),
                         rf=dict(center=outside_circle["center"], R=outside_circle["radius"], edge=outside_circle["edges"]))
        with patch("matplotlib.pyplot.show"):
            exec(compile(ast.parse(plot_source), str(notebook_path), "exec"), namespace)
        self.addCleanup(plt.close, namespace["fig"])
        expected, _, _ = np.histogram2d(points[:5, 0], points[:5, 1], bins=140, range=[[-1, 1], [-1, 1]])
        for name, figure in (("batch", batch_figures[0]), ("viewer", namespace["fig"])):
            with self.subTest(notebook=name):
                figure.canvas.draw()
                for axis in figure.axes[:2]:
                    np.testing.assert_array_equal(axis.get_xlim(), [-1., 1.])
                    np.testing.assert_array_equal(axis.get_ylim(), [-1., 1.])
                    np.testing.assert_array_equal(axis.images[0].get_extent(), [-1., 1., -1., 1.])
                    np.testing.assert_array_equal(axis.images[0].get_array(), expected.T)
                    self.assertEqual(float(axis.images[0].get_array().sum()), 5.)
        for stage in stages.values():
            np.testing.assert_array_equal(np.column_stack((stage["ax"], stage["ay"])), original)
            np.testing.assert_array_equal(stage["theta"], np.full(len(points), 30.))

    def test_loader_uses_native_order_and_rejects_mismatched_channels(self):
        from nptdms import ChannelObject, TdmsWriter

        with tempfile.TemporaryDirectory() as directory:
            recording_path = Path(directory) / "recording.tdms"
            for lengths, interval in (((6, 6, 6, 6), 4e-6), ((6, 5, 6, 6), 4e-6), ((6, 6, 6, 6), -1.)):
                objects = [ChannelObject("APD", name, np.full(length, index + 1.), properties={"wf_increment": interval})
                           for index, (name, length) in enumerate(zip(("90", "45", "135", "0"), lengths))]
                with TdmsWriter(recording_path) as writer:
                    writer.write_segment(objects)
                if lengths == (6, 6, 6, 6) and interval > 0:
                    channels, rate = self.helpers["load_channels"](recording_path)
                    np.testing.assert_array_equal(channels, np.tile([4., 1., 2., 3.], (6, 1)))
                    self.assertAlmostEqual(rate, 250000.)
                else:
                    with self.assertRaises(ValueError):
                        self.helpers["load_channels"](recording_path)

    def test_failed_circle_still_renders_exactly_two_figures(self):
        channels = np.random.default_rng(9).uniform(10., 30., (4000, 4))
        data = self.helpers["prepare_recording"](channels, 250000., 80., self.coefficients)
        original_theta = data["stages"]["fresnel"]["theta"].copy()
        with patch.dict(self.helpers, fit_circle=unittest.mock.Mock(side_effect=RuntimeError("forced fit failure"))), \
                patch("builtins.print") as printed:
            figures = self.helpers["make_figures"](data, "circle-failure check")
        self.assertTrue(any("circle overlay omitted" in str(call) for call in printed.call_args_list))
        self.assertEqual(len(figures), 2)
        self.assertEqual([len(figure.axes) for figure in figures], [3, 6])
        for figure in figures:
            self.addCleanup(plt.close, figure)
            figure.canvas.draw()
            self.assertGreater(np.asarray(figure.canvas.buffer_rgba()).std(), 10.)
        self.assertEqual(len(figures[0].axes[1].patches), 1)
        np.testing.assert_array_equal(data["stages"]["fresnel"]["theta"], original_theta)
        for stage in data["stages"].values():
            stage["theta"][:] = np.nan
            stage["valid_angle"][:] = False
        for figure in self.helpers["make_figures"](data, "no valid angles", fit_rim=False):
            self.addCleanup(plt.close, figure)
            figure.canvas.draw()

    def test_scan_saves_two_plots_per_file_and_continues_after_failure(self):
        channels = np.random.default_rng(5).uniform(10., 30., (1000, 4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            for relative in ("one.tdms", "broken.TDMS", "one.tdms_index", "nested/one.tdms", "nested/ignore.tdmsz"):
                (root / relative).touch()

            def load(path):
                if path.stem == "broken":
                    raise ValueError("test read failure")
                return channels.copy(), 250000.

            previous_figures = set(plt.get_fignums())
            with patch.dict(self.helpers, load_channels=load), patch("builtins.print"):
                results = self.helpers["scan_folder"](root, root / "plots", show_plots=False)
            self.assertEqual([result["file"] for result in results], ["broken.TDMS", "nested/one.tdms", "one.tdms"])
            self.assertEqual([result["status"] for result in results], ["error", "ok", "ok"])
            self.assertEqual({path.relative_to(root / "plots").as_posix() for path in (root / "plots").rglob("*.png")},
                             {"one_anisotropy.png", "one_theta.png", "nested/one_anisotropy.png", "nested/one_theta.png"})
            self.assertEqual(set(plt.get_fignums()), previous_figures)
            with patch.dict(self.helpers, process_file=unittest.mock.Mock(return_value={"status": "ok"})) as namespace, \
                    patch("builtins.print"):
                results = self.helpers["scan_folder"](root, root / "plots", recursive=False, show_plots=False)
                self.assertEqual(namespace["process_file"].call_count, 2)

    def test_unbounded_circle_is_rejected_before_rendering(self):
        angles = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
        points = np.column_stack((.7 * np.cos(angles), .7 * np.sin(angles)))
        unbounded_fit = SimpleNamespace(success=True, x=np.array([24832053., -188167., 24832767.]))
        with patch.dict(self.helpers, minimize=unittest.mock.Mock(return_value=unbounded_fit)):
            with self.assertRaisesRegex(ValueError, "effectively unbounded"):
                self.helpers["fit_circle"](points[:, 0], points[:, 1])
        circle = self.helpers["fit_circle"](points[:, 0], points[:, 1])
        self.assertAlmostEqual(circle["radius"], .7, places=3)

    def test_camera_mosaic_all_parities_and_odd_edges(self):
        for origin_x in (0, 1):
            for origin_y in (0, 1):
                rows, columns = np.indices((5, 7))
                mosaic = np.array([[90, 45], [135, 0]])[(rows + origin_y) % 2, (columns + origin_x) % 2]
                frame = np.zeros((5, 7), dtype=np.uint16)
                for analyzer, value in ((0, 40), (90, 10), (45, 30), (135, 20)):
                    frame[mosaic == analyzer] = value
                frame[-1, :] = 65535
                frame[:, -1] = 65535
                frames = np.stack((frame, frame))
                actual = self.helpers["camera_channel_means"](frames, (origin_x, origin_y), chunk_frames=1)
                np.testing.assert_array_equal(actual, [[40, 10, 30, 20]] * 2)
                np.testing.assert_array_equal(frames[0], frame)

    def test_camera_sidecar_crop_marker_and_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spotrec_example.npy"
            rows, columns = np.indices((6, 8))
            mosaic = np.array([[90, 45], [135, 0]])[(rows + 3) % 2, (columns + 5) % 2]
            frame = np.zeros((6, 8), dtype=np.uint16)
            for analyzer, value in ((0, 40), (90, 10), (45, 30), (135, 20)):
                frame[mosaic == analyzer] = value
            marker = np.zeros_like(frame)
            marker[1, 1] = 1
            frames = np.stack((frame, frame, marker))
            np.save(path, frames)
            metadata = dict(actual=dict(fps=1526.4, frames=2, phase_marker_appended=True),
                            requested=dict(fps=2000.), roi=dict(x=5, y=3, w=8, h=6, phase_x=1, phase_y=1, win_raw=4),
                            rod_location=dict(center_px=dict(x=8., y=6.)),
                            background=dict(background_subtracted=True))
            path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            channels, rate, info = self.helpers["load_camera"](path)
            self.assertEqual(rate, 1526.4)
            self.assertEqual(info["crop_xywh"], (6, 4, 4, 4))
            self.assertTrue(info["marker_removed"])
            self.assertTrue(info["background_subtracted"])
            np.testing.assert_array_equal(channels, [[40, 10, 30, 20]] * 2)
            np.testing.assert_array_equal(np.load(path), frames)
            channels, _, info = self.helpers["load_camera"](path, crop_xywh=(7, 3, 4, 6))
            self.assertEqual(info["crop_xywh"], (7, 3, 4, 6))
            np.testing.assert_array_equal(channels, [[40, 10, 30, 20]] * 2)
            with self.assertRaises(ValueError):
                self.helpers["load_camera"](path, crop_xywh=(0, 0, 4, 4))
            metadata["roi"]["phase_x"] = 0
            path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                self.helpers["load_camera"](path)

    def test_camera_full_frame_schema_and_metadata_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame_stack_example.npy"
            frames = np.full((3, 4, 4), 100, dtype=np.uint16)
            frames[-1] = 0
            frames[-1, 0, 0] = 1
            np.save(path, frames)
            metadata = dict(actual=dict(fps=72.175, frames=2, phase_marker_appended=True),
                            roi=dict(OffsetX=0., OffsetY=0., Width=4., Height=4.), rod_location={})
            sidecar = path.with_suffix(".json")
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertWarnsRegex(RuntimeWarning, "whole saved ROI"):
                channels, rate, info = self.helpers["load_camera"](path)
            self.assertEqual(channels.shape, (2, 4))
            self.assertEqual(rate, 72.175)
            self.assertEqual(info["crop_source"], "whole saved ROI aggregate")
            self.assertEqual(info["crop_xywh"], (0, 0, 4, 4))
            metadata["actual"]["frames"] = 3
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frame count"):
                self.helpers["load_camera"](path)
            metadata["actual"]["phase_marker_appended"] = False
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertWarns(RuntimeWarning):
                channels, _, info = self.helpers["load_camera"](path)
            self.assertEqual(len(channels), 3)
            self.assertFalse(info["marker_removed"])
            metadata["actual"]["phase_marker_appended"] = True
            sidecar.write_text(json.dumps(metadata), encoding="utf-8")
            np.save(path, np.full((3, 4, 4), 100, dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "not a one-pixel marker"):
                self.helpers["load_camera"](path)
            sidecar.write_text("invalid JSON", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                self.helpers["load_camera"](path)

    def test_camera_missing_json_requires_explicit_fps_and_warns_on_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spotrec_example.npy"
            frames = np.full((3, 4, 4), 100, dtype=np.uint16)
            frames[-1] = 0
            frames[-1, 0, 0] = 1
            np.save(path, frames)
            with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(ValueError, "CAMERA_FPS_FALLBACK"):
                self.helpers["load_camera"](path)
            with self.assertWarns(RuntimeWarning):
                channels, rate, info = self.helpers["load_camera"](path, fps_fallback=1000., origin_fallback=(1, 1))
            self.assertEqual(rate, 1000.)
            self.assertEqual(len(channels), 2)
            self.assertIsNone(info["sidecar"])
            self.assertEqual(info["roi_origin_xy"], (1, 1))
            self.assertTrue(info["marker_removed"])

    def test_camera_skips_gains_and_matrix_but_applies_optical_models(self):
        channels = np.array([[40., 10., 30., 20.], [80., 20., 60., 40.]])
        with patch("numpy.linalg.lstsq", side_effect=AssertionError("Camera gain fit forbidden")), \
                patch.dict(self.helpers, INVERSE_T=np.full((4, 4), np.nan)):
            data = self.helpers["prepare_recording"](channels, 1000., 80., self.coefficients, correct_apd=False)
        self.assertIsNone(data["gains"])
        self.assertEqual(tuple(data["stages"]), ("raw", "fresnel"))
        np.testing.assert_array_equal(data["times"], [0., .001])
        for name, stage in data["stages"].items():
            np.testing.assert_allclose(stage["ax"], .6)
            np.testing.assert_allclose(stage["ay"], .2)
            expected = self.helpers["theta_from_radius"]([np.hypot(.6, .2)] * 2,
                                                         self.coefficients["full" if name == "raw" else "fresnel"])
            np.testing.assert_allclose(stage["theta"], expected)
        figures = self.helpers["make_figures"](data, "camera test", fit_rim=False)
        for figure in figures:
            self.addCleanup(plt.close, figure)
            figure.canvas.draw()
        self.assertEqual([len(figure.axes) for figure in figures], [2, 4])
        self.assertIn("No gain or matrix correction", figures[0].axes[0].get_title())
        self.assertIn("Hole + Fresnel", figures[1].axes[2].get_title())

    def test_mixed_scan_ignores_previews_and_saves_distinct_npy_outputs(self):
        channels = np.random.default_rng(12).uniform(10., 30., (1000, 4))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "spotrec_shared.npy", np.full((4, 4, 4), 100, dtype=np.uint16))
            metadata = dict(actual=dict(fps=1000., frames=4, phase_marker_appended=False),
                            roi=dict(x=0, y=0, w=4, h=4, win_raw=4), rod_location=dict(center_px=dict(x=2, y=2)))
            (root / "spotrec_shared.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "spotrec_shared.tdms").touch()
            (root / "spotrec_preview_example.npy").touch()
            (root / "spotrec_shared.stop").touch()
            np.save(root / "not_a_camera.npy", np.zeros(4))
            (root / "not_a_camera.json").write_text(json.dumps(metadata), encoding="utf-8")
            with patch.dict(self.helpers, load_channels=lambda _: (channels, 250000.)), patch("builtins.print"):
                results = self.helpers["scan_folder"](root, root / "plots", show_plots=False)
            self.assertEqual([result["file"] for result in results],
                             ["not_a_camera.npy", "spotrec_shared.npy", "spotrec_shared.tdms"])
            self.assertEqual([result["status"] for result in results], ["error", "ok", "ok"])
            self.assertEqual({path.name for path in (root / "plots").glob("*.png")},
                             {"spotrec_shared.npy_anisotropy.png", "spotrec_shared.npy_theta.png",
                              "spotrec_shared_anisotropy.png", "spotrec_shared_theta.png"})
            self.assertEqual(results[1]["camera"]["frames"], 4)
            self.assertEqual(results[1]["source"], "camera")
            self.assertEqual(results[2]["source"], "tdms")

    def test_standalone_empty_folder_and_missing_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = {}
            with patch.dict("sys.modules", {"hinge_analysis": None}), patch("builtins.print"):
                for cell in self.notebook["cells"]:
                    if cell["cell_type"] == "code":
                        source = "".join(cell["source"])
                        self.assertNotIn("Unit_sphere_viewer.ipynb", source)
                        exec(compile(ast.parse(source), "standalone batch notebook", "exec"), namespace)
                        if "DATA_DIR" in source:
                            namespace.update(DATA_DIR=root, OUTPUT_DIR=root / "plots", SHOW_PLOTS=False)
            self.assertEqual(namespace["SCAN_RESULTS"], [])
            self.assertFalse((root / "plots").exists())
            with self.assertRaises(FileNotFoundError):
                namespace["scan_folder"](root / "missing", root / "plots")


class AnnularRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = notebook_functions(
            "classify_annular_zones", "count_annular_routes",
            "annular_duration_samples", "annular_route_rows",
        )

    def count(self, codes, *, endpoint=2, origin=1, times=None):
        labels = np.asarray(codes, dtype=np.int8)
        if times is None:
            times = np.arange(len(labels)) / 250000.
        return self.helpers["count_annular_routes"](
            labels, times, 250000., min_endpoint_samples=endpoint, min_o_samples=origin,
        )

    def test_radial_and_angular_boundaries_and_full_circle_coverage(self):
        classify = self.helpers["classify_annular_zones"]
        angles = np.radians([0, 90, 130, 140, 150, 160, 170, 180, 190, 200, 359, -1])
        zones = classify(.8 * np.cos(angles), .8 * np.sin(angles), .8, 140.)
        np.testing.assert_array_equal(zones["labels"], [1, 1, 4, 4, 4, 2, 5, 5, 5, 3, 3, 3])
        radial = classify([0., .2, .4, .6, 1., 1.01, np.nan], np.zeros(7), .8, 140.)
        np.testing.assert_array_equal(radial["labels"], [0, 0, -1, 1, 1, -3, -2])
        angles = np.radians(np.arange(0, 360, .1))
        complete = classify(.8 * np.cos(angles), .8 * np.sin(angles), .8, 141.25)
        self.assertEqual(set(complete["labels"]), {1, 2, 3, 4, 5})
        invalid = classify([0., .8], [0., 0.], .8, 140., valid=[False, False])
        np.testing.assert_array_equal(invalid["labels"], [-2, -2])

    def test_geometry_parameter_validation(self):
        classify = self.helpers["classify_annular_zones"]
        for radius, separator, options in (
            (0., 140., {}), (.8, np.nan, {}), (.8, 170., {}),
            (.8, 140., {"outer_radius": .5}), (.8, 140., {"gate_width_deg": 0}),
            (.8, 140., {"core_fraction": .8}), (.8, 140., {"valid": [True, False]}),
        ):
            with self.subTest(radius=radius, separator=separator, options=options):
                with self.assertRaises(ValueError):
                    classify([0.], [0.], radius, separator, **options)

    def test_gate_and_origin_routes_in_both_directions(self):
        cases = (
            ([1, 1, 4, 4, 2, 2], "A->B", "via_gate"),
            ([1, 1, -1, 0, 0, -1, 2, 2], "A->B", "via_O"),
            ([2, 2, 5, 3, 3], "B->C", "via_gate"),
            ([2, 2, -1, 0, -1, 3, 3], "B->C", "via_O"),
        )
        for codes, direction, route in cases:
            for reverse in (False, True):
                ordered = codes[::-1] if reverse else codes
                expected_direction = "->".join(direction.split("->")[::-1]) if reverse else direction
                with self.subTest(direction=expected_direction, route=route):
                    result = self.count(ordered)
                    self.assertEqual(len(result["events"]), 1)
                    event = result["events"][0]
                    self.assertEqual((event["direction"], event["route"], event["exclusion"]),
                                     (expected_direction, route, None))
                    self.assertEqual(result["counts"][expected_direction][route], 1)
                    self.assertAlmostEqual(event["duration_s"],
                                           (event["end_idx"] - event["start_idx"]) / 250000.)

    def test_mixed_wrong_wedge_third_sector_and_unobserved_routes_are_excluded(self):
        cases = (
            ([1, 1, 4, 0, 2, 2], "both_wedge_and_O"),
            ([1, 1, 5, 0, 2, 2], "other_transition_wedge"),
            ([1, 1, 3, 4, 2, 2], "third_outer_sector"),
            ([1, 1, 4, -1, 2, 2], "wedge_route_left_annulus"),
            ([1, 1, 2, 2], "neither_observed_gate_nor_O"),
            ([1, 1, -1, 2, 2], "neither_observed_gate_nor_O"),
        )
        for codes, reason in cases:
            with self.subTest(reason=reason, codes=codes):
                result = self.count(codes)
                self.assertIsNone(result["events"][0]["route"])
                self.assertEqual(result["events"][0]["exclusion"], reason)
                self.assertEqual(result["excluded"]["A->B"], {reason: 1})
                self.assertEqual(sum(result["counts"]["A->B"].values()), 0)

    def test_origin_commitment_is_consecutive_and_short_contacts_are_not_hidden(self):
        result = self.count([1, 1, -1, 0, -1, 0, -1, 2, 2], origin=2)
        event = result["events"][0]
        self.assertEqual(event["exclusion"], "O_contact_too_short")
        self.assertEqual((event["o_samples"], event["max_o_run_samples"]), (2, 1))
        mixed = self.count([1, 1, 0, 4, 2, 2], origin=2)
        self.assertEqual(mixed["events"][0]["exclusion"], "both_wedge_and_O")
        self.assertEqual(self.count([1, 1, 0, 0, 2, 2], origin=2)["counts"]["A->B"]["via_O"], 1)

    def test_invalid_outside_and_time_gaps_reset_the_chain(self):
        for barrier, reason in ((-2, "invalid"), (-3, "outside")):
            result = self.count([1, 1, 4, barrier, 2, 2])
            self.assertFalse(result["events"])
            self.assertEqual(result["chain_breaks"][reason], 1)
        times = np.array([0, 1, 2, 5, 6, 7]) / 250000.
        result = self.count([1, 1, 4, 4, 2, 2], times=times)
        self.assertFalse(result["events"])
        self.assertEqual(result["chain_breaks"]["time_gap"], 1)

    def test_same_source_return_resets_route_start_and_censoring(self):
        result = self.count([1, 1, 4, 1, 1, -1, 0, -1, 2, 2, -1])
        self.assertEqual(result["same_sector_returns"], 1)
        self.assertEqual(result["counts"]["A->B"]["via_O"], 1)
        self.assertEqual(result["events"][0]["start_idx"], 4)
        self.assertEqual(result["chain_breaks"]["record_end"], 1)
        self.assertFalse(self.count([1, 4, 2, 2])["events"])
        self.assertEqual(self.count([1, 1, -1, 3, 3])["unsupported_pairs"], 1)

    def test_empty_denominators_and_time_validation(self):
        rows = self.helpers["annular_route_rows"](self.count([]))
        self.assertTrue(all(np.isnan(row["gate_fraction"]) for row in rows))
        self.assertTrue(all(np.isnan(row["valid_fraction"]) for row in rows))
        for codes, times in (([1, 2], [0., 0.]), ([1, 2], [0., 5e-6]),
                             ([1, 2], [0., np.nan]), ([6, 2], [0., 4e-6])):
            with self.assertRaises(ValueError):
                self.count(codes, times=times)
        for duration, expected in ((4., 1), (12., 3), (20., 5), (40., 10), (5., 2)):
            self.assertEqual(self.helpers["annular_duration_samples"](duration, 250000.), expected)


class SphereWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = notebook_functions("set_sphere_view", "draw_sphere_frame", "create_sphere_widget",
                                         "unwrap_sphere_azimuth", "alternate_sphere_azimuth")
        cls.helpers.update(plt=plt, Normalize=matplotlib.colors.Normalize)

    def test_alternate_fold_moves_only_negative_azimuth_quadrant(self):
        phi = np.radians([-90., -45., -1., 0., 45., 90., np.nan])
        original = phi.copy()
        alternate = self.helpers["alternate_sphere_azimuth"](phi)
        np.testing.assert_allclose(np.degrees(alternate), [90., 135., 179., 0., 45., 90., np.nan], equal_nan=True)
        np.testing.assert_allclose(np.exp(2j * alternate[:-1]), np.exp(2j * phi[:-1]), atol=1e-14)
        np.testing.assert_allclose(np.cos(alternate[:3]), -np.cos(phi[:3]), atol=1e-14)
        np.testing.assert_allclose(np.sin(alternate[:3]), -np.sin(phi[:3]), atol=1e-14)
        np.testing.assert_array_equal(alternate[3:6], phi[3:6])
        np.testing.assert_array_equal(phi, original)

    def test_native_flag_changes_only_sphere_sampling_and_preserves_invalids(self):
        notebook_path = Path(__file__).with_name("Unit_sphere_viewer.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cell_source = next("".join(cell["source"]) for cell in notebook["cells"]
                           if cell["cell_type"] == "code" and "def alternate_sphere_azimuth(" in "".join(cell["source"]))
        tree = ast.parse(cell_source)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"SPHERE_USE_NATIVE", "SPHERE_BACKGROUND_MAX_POINTS"}
            for target in node.targets))]
        compiled = compile(tree, str(notebook_path), "exec")
        recording = {"c0": np.full(200, 2.), "c90": np.ones(200),
                     "c45": np.full(200, 1.5), "c135": np.ones(200),
                     "t": np.arange(200) / 250000., "fps": 250000.}
        corrected = {name: values.copy() if isinstance(values, np.ndarray) else values
                     for name, values in recording.items()}
        corrected["c0"] *= 1.1
        corrected["c0"][60] = -.5
        raw_overview = ha.block_channel_means(recording, 20, return_flags=True)
        corrected_overview = ha.block_channel_means(corrected, 20, return_flags=True)
        stages = {"raw": raw_overview[:4], "corrected": corrected_overview[:4], "fresnel": corrected_overview[:4]}
        channel_flags = {"raw": raw_overview[4], "corrected": corrected_overview[4], "fresnel": corrected_overview[4]}
        coefficients = {"full": ha.fourkas_ABC(), "fresnel": ha.fourkas_ABC_fresnel()}
        stage_models = {"raw": "full", "corrected": "full", "fresnel": "fresnel"}
        original_corrected = corrected["c0"].copy()
        for stage in stages:
            for native in (True, False):
                with self.subTest(stage=stage, native=native):
                    namespace = dict(np=np, ha=ha, SPHERE_USE_NATIVE=native, SPHERE_BACKGROUND_MAX_POINTS=7,
                                     SPHERE_STAGE=stage, SHOW_EVERY=3, rec=recording, r1=corrected,
                                     STAGES=stages, CHANNEL_FLAGS=channel_flags, OPTICAL_COEFFICIENTS=coefficients,
                                     STAGE_MODELS=stage_models, MODEL_LABELS={"full": "full", "fresnel": "fresnel"},
                                     R_MAX_BY_STAGE={name: ha.r_max_of(*coefficients[model]) for name, model in stage_models.items()})
                    with patch("builtins.print"):
                        exec(compiled, namespace)
                    expected = ha.block_channel_means(recording if stage == "raw" else corrected,
                                                      1 if native else 20, return_flags=True)
                    np.testing.assert_allclose(namespace["Xs"], expected[0])
                    np.testing.assert_array_equal(namespace["ts"], expected[2])
                    self.assertEqual(namespace["fps_s"], 250000. if native else 12500.)
                    self.assertEqual(len(namespace["ts"]), 200 if native else 10)
                    self.assertLessEqual(len(namespace["sphere_display"]), 7)
                    if native and stage != "raw":
                        self.assertFalse(namespace["sphere_valid"][60])
                        self.assertTrue(np.isnan(namespace["SPHERE_ALTERNATE_COORDINATES"][60]).all())
                    np.testing.assert_array_equal(corrected["c0"], original_corrected)
                    np.testing.assert_array_equal(stages["fresnel"][2], corrected_overview[2])

    def test_azimuth_unwrap_preserves_double_phase_and_theta(self):
        phi = np.radians([85., 89., -89., -85., -80.])
        original = phi.copy()
        unwrapped = self.helpers["unwrap_sphere_azimuth"](phi, np.arange(len(phi)) / 12500., 12500.)
        np.testing.assert_allclose(np.degrees(unwrapped), [85., 89., 91., 95., 100.])
        np.testing.assert_allclose(np.exp(2j * unwrapped), np.exp(2j * phi), atol=1e-14)
        theta = np.radians(np.linspace(10., 70., len(phi)))
        folded = np.column_stack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
        trial = np.column_stack((np.sin(theta) * np.cos(unwrapped), np.sin(theta) * np.sin(unwrapped), np.cos(theta)))
        np.testing.assert_allclose(trial[:2], folded[:2])
        np.testing.assert_allclose(trial[2:, :2], -folded[2:, :2])
        np.testing.assert_array_equal(trial[:, 2], folded[:, 2])
        np.testing.assert_allclose(np.linalg.norm(trial, axis=1), 1.)
        np.testing.assert_array_equal(phi, original)

    def test_azimuth_unwrap_restarts_after_invalid_poles_and_time_gaps(self):
        unwrap = self.helpers["unwrap_sphere_azimuth"]
        phi = np.radians([89., -89., np.nan, -85., -80., 0., -70., -65.])
        valid = np.ones(len(phi), dtype=bool)
        valid[5] = False
        result = unwrap(phi, np.arange(len(phi)) / 12500., 12500., valid)
        np.testing.assert_allclose(np.degrees(result), [89., 91., np.nan, -85., -80., np.nan, -70., -65.],
                                   equal_nan=True)
        result = unwrap(np.radians([89., -89., -85., -80.]), np.array([0., 1., 4., 5.]) / 12500., 12500.)
        np.testing.assert_allclose(np.degrees(result), [89., 91., -85., -80.])
        self.assertEqual(unwrap([], [], 12500.).size, 0)
        self.assertTrue(np.isnan(unwrap([np.nan], [0.], 12500.))[0])
        for angles, times, rate, mask in (([0., 1.], [0.], 12500., None),
                                          ([0., 1.], [0., 0.], 12500., None),
                                          ([0.], [0.], 0., None), ([0.], [0.], 12500., [True, False])):
            with self.assertRaises(ValueError):
                unwrap(angles, times, rate, mask)

    def test_fixed_views_keep_equal_geometry_and_hide_collapsed_ticks(self):
        figure = plt.figure(figsize=(6, 6))
        self.addCleanup(plt.close, figure)
        views = ((18., 55.), (90., -90.), (0., -90.), (0., 0.))
        for number, view in enumerate(views, 1):
            axis = figure.add_subplot(2, 2, number, projection="3d")
            self.helpers["draw_sphere_frame"](axis, view)
            aspect = axis.get_box_aspect()
            np.testing.assert_allclose(aspect, aspect[0])
            for limits in (axis.get_xlim(), axis.get_ylim(), axis.get_zlim()):
                np.testing.assert_array_equal(limits, (-1., 1.))
            tick_counts = tuple(len(ticks) for ticks in (axis.get_xticks(), axis.get_yticks(), axis.get_zticks()))
            self.assertEqual(tick_counts, ((3, 3, 3), (3, 3, 0), (3, 0, 3), (0, 3, 3))[number - 1])
        figure.canvas.draw()
        self.assertGreater(np.asarray(figure.canvas.buffer_rgba()).std(), 10.)

    @unittest.skipUnless(importlib.util.find_spec("ipympl"), "Optional ipympl is not installed")
    def test_mouse_rotation_and_time_update_preserve_camera_and_observations(self):
        from matplotlib.backend_bases import MouseEvent

        times = np.arange(200) / 1000.
        theta_deg = np.linspace(10., 70., len(times))
        theta = np.radians(theta_deg)
        azimuth = np.linspace(-1., 1., len(times))
        coordinates = np.column_stack((np.sin(theta) * np.cos(azimuth),
                                       np.sin(theta) * np.sin(azimuth), np.cos(theta)))
        original_coordinates = coordinates.copy()
        valid = np.ones(len(times), dtype=bool)
        valid[[25, 68]] = False
        backend_before = matplotlib.get_backend()
        fallback_before = matplotlib.rcParams["backend_fallback"]
        existing_figure = plt.figure()
        self.addCleanup(plt.close, existing_figure)
        existing_canvas = existing_figure.canvas
        viewer = self.helpers["create_sphere_widget"](
            coordinates, theta_deg, times, valid, np.arange(0, len(times), 3),
            {"Oblique": (18., 55.), "Top (xy)": (90., -90.)},
            time_step=.001, window_s=.05, max_context_points=20,
        )
        self.addCleanup(viewer["widget"].close)
        self.addCleanup(viewer["manager"].toolbar.close)
        self.addCleanup(viewer["canvas"].close)
        self.addCleanup(viewer["time_link"].unlink)
        self.assertEqual(matplotlib.get_backend(), backend_before)
        self.assertEqual(matplotlib.rcParams["backend_fallback"], fallback_before)
        self.assertTrue(plt.fignum_exists(existing_figure.number))
        self.assertIs(existing_figure.canvas, existing_canvas)
        self.assertLessEqual(viewer["background_count"], 20)
        axis, canvas = viewer["axis"], viewer["canvas"]
        camera_before = np.array([axis.elev, axis.azim, axis.roll])
        horizontal = axis.bbox.x0 + .5 * axis.bbox.width
        vertical = axis.bbox.y0 + .5 * axis.bbox.height
        for name, location in (("button_press_event", (horizontal, vertical)),
                               ("motion_notify_event", (horizontal + 45, vertical + 25)),
                               ("button_release_event", (horizontal + 45, vertical + 25))):
            canvas.callbacks.process(name, MouseEvent(name, canvas, *location, button=1))
        camera_after_drag = np.array([axis.elev, axis.azim, axis.roll])
        self.assertFalse(np.allclose(camera_before, camera_after_drag))
        viewer["time_control"].value = .05
        np.testing.assert_allclose([axis.elev, axis.azim, axis.roll], camera_after_drag)
        selected = np.flatnonzero(valid & (times >= .05) & (times < .1))
        np.testing.assert_allclose(np.column_stack(viewer["window_points"]._offsets3d), coordinates[selected])
        np.testing.assert_allclose(np.column_stack(viewer["recent_points"]._offsets3d), coordinates[selected[-20:]])
        np.testing.assert_array_equal(coordinates, original_coordinates)
        viewer["view_control"].value = "Top (xy)"
        self.assertEqual(axis.elev, 90.)
        self.assertEqual(len(axis.get_zticks()), 0)
        viewer["context_control"].value = False
        self.assertFalse(viewer["background"].get_visible())
        axis.set_xlim(.2, 1.2)
        viewer["reset_control"].click()
        self.assertEqual((axis.elev, axis.azim), (18., 55.))
        np.testing.assert_allclose(axis.get_xlim(), (-1., 1.))
        canvas.draw()
        self.assertGreater(np.asarray(canvas.buffer_rgba()).std(), 10.)

    @unittest.skipUnless(importlib.util.find_spec("ipympl"), "Optional ipympl is not installed")
    def test_two_ms_window_exact_time_input_and_mode_changes_preserve_camera(self):
        times = 11.8 + np.arange(200) / 12500.
        phi = np.radians((np.linspace(80., 240., len(times)) + 90.) % 180. - 90.)
        unwrapped = self.helpers["unwrap_sphere_azimuth"](phi, times, 12500.)
        theta_deg = np.full(len(times), 45.)
        amplitude = np.sqrt(.5)
        folded = np.column_stack((amplitude * np.cos(phi), amplitude * np.sin(phi), np.full(len(phi), amplitude)))
        trial = np.column_stack((amplitude * np.cos(unwrapped), amplitude * np.sin(unwrapped), np.full(len(phi), amplitude)))
        valid = np.ones(len(times), dtype=bool)
        valid[68] = False
        viewer = self.helpers["create_sphere_widget"](
            folded, theta_deg, times, valid, np.arange(0, len(times), 3),
            {"Oblique": (18., 55.)}, window_s=.002, unwrapped_coordinates=trial,
        )
        self.addCleanup(viewer["widget"].close)
        self.addCleanup(viewer["manager"].toolbar.close)
        self.addCleanup(viewer["canvas"].close)
        self.addCleanup(viewer["time_link"].unlink)
        self.assertEqual(viewer["mode_control"].value, "Unwrapped")
        self.assertAlmostEqual(viewer["time_control"].step, 80e-6)
        self.assertAlmostEqual(viewer["time_input"].step, 80e-6)
        viewer["axis"].view_init(elev=35., azim=123.)
        exact_time = 11.802003
        viewer["time_input"].value = exact_time
        self.assertEqual(viewer["time_control"].value, exact_time)
        selected = np.flatnonzero(valid & (times >= exact_time) & (times < exact_time + .002))
        self.assertEqual(len(selected), 25)
        np.testing.assert_allclose(np.column_stack(viewer["window_points"]._offsets3d), trial[selected])
        viewer["mode_control"].value = "Folded"
        np.testing.assert_allclose(np.column_stack(viewer["window_points"]._offsets3d), folded[selected])
        np.testing.assert_allclose(np.column_stack(viewer["background"]._offsets3d), folded[np.arange(0, len(times), 3)])
        self.assertEqual((viewer["axis"].elev, viewer["axis"].azim), (35., 123.))
        viewer["time_control"].value = 11.804123
        self.assertEqual(viewer["time_input"].value, 11.804123)
        selected = np.flatnonzero(valid & (times >= 11.804123) & (times < 11.806123))
        np.testing.assert_allclose(np.column_stack(viewer["window_points"]._offsets3d), folded[selected])
        self.assertNotIn(68, selected)
        viewer["time_input"].value = times[-1] + 10.
        self.assertEqual(viewer["time_input"].value, viewer["time_control"].max)
        viewer["time_input"].value = times[0] - 10.
        self.assertEqual(viewer["time_control"].value, times[0])
        self.assertIn("2 ms", viewer["axis"].get_title())

    @unittest.skipUnless(importlib.util.find_spec("ipympl"), "Optional ipympl is not installed")
    def test_native_200_us_editable_bounds_and_ordered_lines_in_all_modes(self):
        times = 11.8 + np.arange(200) / 250000.
        phi = np.radians((np.linspace(80., 120., len(times)) + 90.) % 180. - 90.)
        angles_by_mode = {"Folded": phi,
                          "Unwrapped": self.helpers["unwrap_sphere_azimuth"](phi, times, 250000.),
                          "Alternate fold": self.helpers["alternate_sphere_azimuth"](phi)}
        amplitude = np.sqrt(.5)
        coordinates = {mode: np.column_stack((amplitude * np.cos(angles), amplitude * np.sin(angles),
                                               np.full(len(times), amplitude)))
                       for mode, angles in angles_by_mode.items()}
        viewer = self.helpers["create_sphere_widget"](
            coordinates["Folded"], np.full(len(times), 45.), times, np.ones(len(times), dtype=bool),
            np.arange(0, len(times), 7), {"Oblique": (18., 55.)},
            unwrapped_coordinates=coordinates["Unwrapped"], alternate_coordinates=coordinates["Alternate fold"],
        )
        self.addCleanup(viewer["widget"].close)
        self.addCleanup(viewer["manager"].toolbar.close)
        self.addCleanup(viewer["canvas"].close)
        self.addCleanup(viewer["time_link"].unlink)
        self.assertEqual(len(viewer["window_points"]._offsets3d[0]), 50)
        self.assertEqual(len(viewer["window_path"]._segments3d), 49)
        self.assertAlmostEqual(viewer["time_control"].step, 4e-6)
        self.assertFalse(viewer["background"].get_visible())
        viewer["axis"].view_init(elev=35., azim=120.)
        viewer["start_input"].value = 11.8002
        self.assertAlmostEqual(viewer["end_input"].value, 11.8004)
        viewer["end_input"].value = 11.8005
        selected = np.arange(50, 125)
        for mode, points in coordinates.items():
            with self.subTest(mode=mode):
                viewer["mode_control"].value = mode
                np.testing.assert_allclose(np.column_stack(viewer["window_points"]._offsets3d), points[selected])
                np.testing.assert_allclose(viewer["window_path"]._segments3d,
                                           np.stack((points[selected[:-1]], points[selected[1:]]), axis=1))
                np.testing.assert_allclose(np.column_stack(viewer["start_point"]._offsets3d), points[selected[:1]])
                np.testing.assert_allclose(np.column_stack(viewer["end_point"]._offsets3d), points[selected[-1:]])
                self.assertEqual((viewer["axis"].elev, viewer["axis"].azim), (35., 120.))
        viewer["time_control"].value = 11.8003
        self.assertAlmostEqual(viewer["end_input"].value, 11.8006)
        self.assertEqual(len(viewer["window_points"]._offsets3d[0]), 75)
        viewer["connection_control"].value = False
        self.assertFalse(viewer["window_path"].get_visible())
        viewer["connection_control"].value = True
        viewer["context_control"].value = True
        self.assertTrue(viewer["window_path"].get_visible())
        self.assertTrue(viewer["background"].get_visible())
        viewer["end_input"].value = viewer["start_input"].value
        self.assertAlmostEqual(viewer["end_input"].value - viewer["start_input"].value, 4e-6)
        self.assertEqual(len(viewer["window_points"]._offsets3d[0]), 1)
        self.assertEqual(len(viewer["window_path"]._segments3d), 0)

    @unittest.skipUnless(importlib.util.find_spec("ipympl"), "Optional ipympl is not installed")
    def test_connections_break_at_invalid_samples_gaps_and_empty_windows(self):
        times = np.array([0., 1., 2., 3., 6., 7., 8., 9.]) / 250000.
        azimuth = np.linspace(-.5, .5, len(times))
        coordinates = np.column_stack((np.sqrt(.5) * np.cos(azimuth), np.sqrt(.5) * np.sin(azimuth),
                                       np.full(len(times), np.sqrt(.5))))
        valid = np.ones(len(times), dtype=bool)
        valid[2] = False
        trial = coordinates.copy()
        trial[5:] = np.nan
        viewer = self.helpers["create_sphere_widget"](
            coordinates, np.full(len(times), 45.), times, valid, np.arange(len(times)),
            {"Oblique": (18., 55.)}, unwrapped_coordinates=trial, alternate_coordinates=coordinates,
            t_start=0., t_end=40e-6,
        )
        self.addCleanup(viewer["widget"].close)
        self.addCleanup(viewer["manager"].toolbar.close)
        self.addCleanup(viewer["canvas"].close)
        self.addCleanup(viewer["time_link"].unlink)
        for mode in ("Folded", "Alternate fold"):
            viewer["mode_control"].value = mode
            np.testing.assert_allclose(viewer["window_path"]._segments3d,
                                       coordinates[np.array([[0, 1], [4, 5], [5, 6], [6, 7]])])
        viewer["mode_control"].value = "Unwrapped"
        np.testing.assert_allclose(viewer["window_path"]._segments3d, coordinates[np.array([[0, 1]])])
        viewer["end_input"].value = 8e-6
        viewer["start_input"].value = 28e-6
        self.assertEqual(len(viewer["window_points"]._offsets3d[0]), 0)
        self.assertEqual(len(viewer["window_path"]._segments3d), 0)
        self.assertEqual(len(viewer["start_point"]._offsets3d[0]), 0)
        self.assertEqual(len(viewer["end_point"]._offsets3d[0]), 0)
        viewer["canvas"].draw()


class OpticalTests(unittest.TestCase):
    def test_round_trip_and_physical_bounds(self):
        angles = np.linspace(0, 90, 181)
        for coefficients in (ha.fourkas_ABC(), ha.fourkas_ABC(NA_in=.38),
                             ha.fourkas_ABC_fresnel()):
            coeff_a, coeff_b, coeff_c = coefficients
            sine_squared = np.sin(np.radians(angles))**2
            radius = coeff_c * sine_squared / (coeff_a + coeff_b * sine_squared)
            recovered, flags = ha.theta_from_r(radius, *coefficients, return_flags=True)
            np.testing.assert_allclose(recovered, angles, atol=1e-6)
            self.assertTrue(flags["valid"].all())
        self.assertAlmostEqual(ha.r_max_of(*ha.fourkas_ABC(1.3, 1.3)), .875)
        with self.assertRaises(ValueError):
            ha.fourkas_ABC(1.45, 1.33)

    def test_fresnel_equal_media_and_convergence(self):
        for inner_na in (0., .38):
            expected = ha.fourkas_ABC(1.3, 1.33, inner_na)
            result = ha.fourkas_ABC_fresnel(1.3, 1.33, 1.33, inner_na)
            np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(ha.fourkas_ABC_fresnel(quadrature_order=128),
                                   ha.fourkas_ABC_fresnel(quadrature_order=256), rtol=1e-12)

    def test_fresnel_matches_direct_analyser_integration(self):
        nodes, weights = np.polynomial.legendre.leggauss(128)
        pupil_radius = .38 + (nodes + 1) * (1.3 - .38) / 2
        sin_water = pupil_radius / 1.33
        cos_water = np.sqrt(1 - sin_water**2)
        cos_oil = np.sqrt(1 - (pupil_radius / 1.515)**2)
        transmission_s = 2 * 1.33 * cos_water / (1.33 * cos_water + 1.515 * cos_oil)
        transmission_p = 2 * 1.33 * cos_water / (1.515 * cos_water + 1.33 * cos_oil)
        field_weight = np.sqrt(cos_oil) / cos_water
        measure = weights * (1.3 - .38) / 2 * pupil_radius * field_weight**2 / 1.33**2
        pupil_azimuth = np.linspace(0, 2 * np.pi, 256, endpoint=False)
        analysers = np.radians([0, 90, 45, 135])
        coeff_a, coeff_b, coeff_c = ha.fourkas_ABC_fresnel()
        for theta_deg, phi_deg in ((20, 15), (55, 120), (90, 45)):
            theta, phi = np.radians([theta_deg, phi_deg])
            dipole = np.array([np.sin(theta) * np.cos(phi),
                               np.sin(theta) * np.sin(phi), np.cos(theta)])
            radial = transmission_p[:, None] * (
                cos_water[:, None] * (dipole[0] * np.cos(pupil_azimuth)
                                     + dipole[1] * np.sin(pupil_azimuth))
                - sin_water[:, None] * dipole[2])
            tangential = transmission_s[:, None] * (
                -dipole[0] * np.sin(pupil_azimuth) + dipole[1] * np.cos(pupil_azimuth))
            measured = []
            for analyser in analysers:
                field = (radial * np.cos(pupil_azimuth - analyser)
                         - tangential * np.sin(pupil_azimuth - analyser))
                measured.append(.5 * np.sum(measure * np.mean(field**2, axis=1)))
            expected = (coeff_a + coeff_b * np.sin(theta)**2
                        + coeff_c * np.sin(theta)**2 * np.cos(2 * (phi - analysers)))
            np.testing.assert_allclose(measured, expected, rtol=1e-12, atol=1e-14)

    def test_invalid_radii_stay_flagged(self):
        coefficients = ha.fourkas_ABC()
        values = [np.nan, np.inf, -.1, 1.2, 0.]
        angles, flags = ha.theta_from_r(values, *coefficients, return_flags=True)
        self.assertTrue(np.isnan(angles[:4]).all())
        self.assertEqual(angles[-1], 0.)
        np.testing.assert_array_equal(flags["valid"], [False, False, False, False, True])
        self.assertTrue(flags["over_range"][3])
        self.assertTrue(flags["negative_radius"][2])
        clipped, flags = ha.theta_from_r(values, *coefficients, clip_r=True, return_flags=True)
        self.assertEqual(clipped[3], 90.)
        self.assertFalse(flags["valid"][3])


class PhotometryTests(unittest.TestCase):
    def test_gain_before_matrix_round_trip(self):
        phase = np.linspace(0, 8 * np.pi, 2000)
        coeff_a, coeff_b, coeff_c = ha.fourkas_ABC()
        sine_squared = np.sin(np.radians(45 + 25 * np.sin(.37 * phase)))**2
        base = coeff_a + coeff_b * sine_squared
        mod_x = coeff_c * sine_squared * np.cos(2 * phase)
        mod_y = coeff_c * sine_squared * np.sin(2 * phase)
        ideal = np.column_stack((base + mod_x, base - mod_x, base + mod_y, base - mod_y))
        known_gains = np.array([1., 1.3, .8, 1.1])
        raw = (ideal @ np.linalg.inv(ha.apd_tmatrix()).T) / known_gains
        names = ("0", "90", "45", "135")
        recording = {f"c{name}": raw[:, index].copy() for index, name in enumerate(names)}
        recording.update(t=np.arange(len(raw)) / 250000., fps=250000., n_frames=len(raw))
        corrected, gains = ha.correct_apd_channels(recording)
        recovered = np.column_stack([corrected[f"c{name}"] for name in names])
        np.testing.assert_allclose(recovered, ideal, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose([gains[name] for name in names], known_gains, rtol=1e-12)
        np.testing.assert_array_equal(recording["c0"], raw[:, 0])
        recording["c45"][3] = np.nan
        corrected_with_gap, finite_gains = ha.correct_apd_channels(recording)
        np.testing.assert_allclose([finite_gains[name] for name in names], known_gains, rtol=1e-12)
        _, _, _, _, flags = ha.block_channel_means(corrected_with_gap, 1, return_flags=True)
        self.assertTrue(flags["nonfinite"][3])
        self.assertFalse(flags["valid"][3])
        recording["c0"][0] = -1.
        unchanged, _ = ha.correct_apd_channels(recording, dict.fromkeys(names, 1.), np.eye(4))
        self.assertEqual(unchanged["c0"][0], -1.)

    def test_all_blocks_and_ratio_of_means(self):
        recording = {"c0": np.arange(1., 9.), "c90": np.ones(8),
                     "c45": np.full(8, 2.), "c135": np.ones(8),
                     "t": np.arange(8) / 250000., "fps": 250000.}
        ax_values, ay_values, times, rate = ha.block_channel_means(recording, 2)
        expected_mean = np.array([1.5, 3.5, 5.5, 7.5])
        np.testing.assert_allclose(ax_values, (expected_mean - 1) / (expected_mean + 1))
        np.testing.assert_allclose(ay_values, 1 / 3)
        np.testing.assert_allclose(np.diff(times), 8e-6)
        self.assertEqual(len(times), 4)
        self.assertEqual(rate, 125000.)
        with self.assertRaises(ValueError):
            ha.block_channel_means(recording, 0)

    def test_photometry_flags(self):
        recording = {"c0": np.array([1., 0., -2., -1., np.nan]),
                     "c90": np.array([1., 0., 1., 2., 1.]),
                     "c45": np.ones(5), "c135": np.ones(5),
                     "t": np.arange(5) / 1000., "fps": 1000.}
        ax_values, _, _, _, flags = ha.block_channel_means(recording, 1, return_flags=True)
        self.assertTrue(np.isnan(ax_values[[1, 2, 4]]).all())
        self.assertEqual(ax_values[3], -3.)
        self.assertTrue(flags["negative_channel"][3])
        np.testing.assert_array_equal(flags["valid"], [True, False, False, False, False])


class ExcursionTests(unittest.TestCase):
    def test_first_passage_needs_confirmed_core_and_outer_visits(self):
        radius = np.array([.9, .9, 0., 0., .3, .5, .81, .82, .85, .4,
                           .9, 0., 0., .4, .82, .5, 0., 0., .4, .85, .9])
        result = ha.extract_radial_excursions(
            -radius, np.zeros_like(radius), np.arange(len(radius)) * .001, (0., 0.), 1.,
            min_core_samples=2, min_outer_samples=2,
        )
        self.assertEqual([(path["start_idx"], path["end_idx"], path["confirm_idx"])
                          for path in result["paths"]], [(3, 6, 7), (17, 19, 20)])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["aborted"], 1)
        self.assertEqual(result["censored"], 0)
        self.assertAlmostEqual(result["paths"][0]["duration_s"], .003)

    def test_invalid_samples_and_time_gaps_break_paths(self):
        for interrupted_by in ("mask", "nan", "time_gap"):
            with self.subTest(interrupted_by=interrupted_by):
                radius = np.array([0., 0., .3, .5, .7, .85, .9, 0., 0., .4])
                times = np.arange(len(radius)) * .001
                valid = np.ones(len(radius), dtype=bool)
                if interrupted_by == "mask":
                    valid[3] = False
                elif interrupted_by == "nan":
                    radius[3] = np.nan
                else:
                    times[3:] += .005
                result = ha.extract_radial_excursions(
                    -radius, np.zeros_like(radius), times, (0., 0.), 1.,
                    valid=valid, min_core_samples=2, min_outer_samples=2,
                )
                self.assertEqual(result["paths"], [])
                self.assertEqual(result["censored"], 2)

    def test_circle_coordinates_do_not_transform_data(self):
        circle_center = np.array([-.04, .025])
        core_center = circle_center + np.array([.04, -.025])
        points = np.array([core_center, core_center, [-.3, .2], [-.5, .45], [-.55, .5]])
        original = points.copy()
        result = ha.extract_radial_excursions(
            points[:, 0], points[:, 1], np.arange(len(points)) * .001, circle_center, .74,
            core_center=core_center, min_core_samples=2, min_outer_samples=2,
        )
        self.assertEqual(len(result["paths"]), 1)
        self.assertEqual(result["paths"][0]["end_idx"], 3)
        np.testing.assert_array_equal(points, original)
        with self.assertRaises(ValueError):
            ha.extract_radial_excursions([], [], [], (0, 0), 1, core_center=(.79, 0))

    def test_crossing_profiles_separate_directions_without_classifying_them(self):
        radii = np.array([0., 0., .25, .45, .65, .85, .9])
        center = np.array([-.04, .025])
        points = []
        for angle in (120., 180., 350.):
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            points.extend(center + radii[:, None] * direction)
        points = np.asarray(points)
        times = np.arange(len(points)) * .001
        events = ha.extract_radial_excursions(
            points[:, 0], points[:, 1], times, center, 1.,
            min_core_samples=2, min_outer_samples=2,
        )
        profiles = ha.radial_crossing_profiles(
            points[:, 0], points[:, 1], times, events, center, 1., [.4, .6, .8],
        )
        np.testing.assert_allclose(profiles["bearing_deg"], [[120.] * 3, [180.] * 3, [350.] * 3])
        np.testing.assert_allclose(np.linalg.norm(profiles["points"] - center, axis=2),
                                   np.tile([.4, .6, .8], (3, 1)))
        self.assertTrue(np.all(np.diff(profiles["times"], axis=1) > 0))

    def test_target_profile_uses_confirmed_crossing_after_a_probe(self):
        points = np.array([[0., 0.], [0., 0.], [-.4, 0.], [-.85, 0.],
                           [-.5, .1], [0., .85], [0., .9]])
        times = np.arange(len(points)) * .001
        events = ha.extract_radial_excursions(
            points[:, 0], points[:, 1], times, (0., 0.), 1.,
            min_core_samples=2, min_outer_samples=2,
        )
        profiles = ha.radial_crossing_profiles(
            points[:, 0], points[:, 1], times, events, (0., 0.), 1., [.4, .8],
        )
        self.assertGreater(profiles["times"][0, -1], times[4])
        self.assertLess(profiles["bearing_deg"][0, -1], 100)
        self.assertAlmostEqual(np.linalg.norm(profiles["points"][0, -1]), .8)

    def test_smaller_target_keeps_a_shorter_upper_excursion(self):
        points = np.array([[0., 0.], [0., 0.], [0., .3], [0., .55], [0., .6],
                           [0., .3], [0., 0.], [0., 0.], [-.3, 0.], [-.6, 0.],
                           [-.7, 0.], [-.85, 0.], [-.9, 0.]])
        times = np.arange(len(points)) * .001
        counts = []
        for target in (.5, .8):
            result = ha.extract_radial_excursions(
                points[:, 0], points[:, 1], times, (0., 0.), 1., outer_fraction=target,
                min_core_samples=2, min_outer_samples=2,
            )
            counts.append(len(result["paths"]))
        self.assertEqual(counts, [2, 1])


class ConstrainedRegionTests(unittest.TestCase):
    def test_joint_radial_fit_has_identical_axes_area_and_clearances(self):
        rng = np.random.default_rng(8)
        points = np.vstack((rng.normal((-.5, .4), (.045, .025), size=(1800, 2)),
                            rng.normal((-.4, -.4), (.035, .03), size=(1500, 2))))
        original = points.copy()
        result = ha.fit_radial_equal_area_ellipses(
            points[:, 0], points[:, 1], .12, (0., 1.), histogram_step=.015,
            min_aspect=2., max_aspect=4., maxiter=120, popsize=8, seed=5,
        )
        np.testing.assert_array_equal(result["regions"]["B"]["semiaxes"], result["regions"]["C"]["semiaxes"])
        self.assertAlmostEqual(np.linalg.norm(result["regions"]["B"]["center"]),
                               np.linalg.norm(result["regions"]["C"]["center"]))
        self.assertAlmostEqual(result["pair_clearance"], .12)
        for edge_gap in result["pair_gaps"].values():
            self.assertGreaterEqual(edge_gap, .12 - 1e-10)
        for name, side in (("B", 1), ("C", -1)):
            region = result["regions"][name]
            diagnostics = result["diagnostics"][name]
            self.assertTrue(2 <= diagnostics["aspect_ratio"] <= 4)
            self.assertAlmostEqual(np.pi * np.prod(region["semiaxes"]), result["area"])
            direction = np.array([np.cos(np.radians(region["angle_deg"])), np.sin(np.radians(region["angle_deg"]))])
            self.assertAlmostEqual(region["center"][0] * direction[1] - region["center"][1] * direction[0], 0.)
            gaps = ha.ellipse_region_clearances(region, .12, (0., 1.), side)
            self.assertGreaterEqual(min(gaps.values()), .005 - 1e-10)
            self.assertGreater(diagnostics["histogram_count"], 1300)
            np.testing.assert_allclose(diagnostics["radial_interval"],
                                       diagnostics["center_radius"] + np.array([-1, 1]) * region["semiaxes"][0])
        np.testing.assert_array_equal(points, original)
        with self.assertRaises(ValueError):
            ha.fit_radial_equal_area_ellipses(points[:, 0], points[:, 1], .12, (0., 1.), min_aspect=1.)
        with self.assertRaises(ValueError):
            ha.fit_radial_equal_area_ellipses(points[:, 0], points[:, 1], .12, (0., 1.), pair_clearance=.05)

    def test_radial_pair_gap_matches_boundary_distance_and_is_rotation_invariant(self):
        from scipy.spatial import cKDTree

        regions = {"A": dict(center=(0., 0.), semiaxes=(.15, .15))}
        boundaries = []
        phase = np.linspace(0, 2 * np.pi, 16000, endpoint=False)
        for name, bearing in (("B", 175.), ("C", 120.)):
            angle = np.radians(bearing)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            center = .7 * rotation[:, 0]
            regions[name] = dict(center=center, semiaxes=(.25, .09), angle_deg=bearing)
            boundaries.append(np.column_stack((.25 * np.cos(phase), .09 * np.sin(phase))) @ rotation.T + center)
        gaps = ha.radial_region_gaps(regions)
        measured = float(cKDTree(boundaries[0]).query(boundaries[1])[0].min())
        self.assertAlmostEqual(gaps["A-B"], .3)
        self.assertAlmostEqual(gaps["A-C"], .3)
        self.assertAlmostEqual(gaps["B-C"], measured, delta=2e-6)
        rotate = notebook_functions("rotate_region_frame")["rotate_region_frame"]
        rotated = rotate(regions, (-.5, -.5), 71.)
        np.testing.assert_allclose(list(ha.radial_region_gaps(rotated["regions"]).values()), list(gaps.values()), atol=1e-12)
        inconsistent = {**regions, "C": {**regions["C"], "center": regions["C"]["center"] * 1.1}}
        with self.assertRaises(ValueError):
            ha.radial_region_gaps(inconsistent)

    def test_separator_finds_interior_gap_in_supplied_annulus(self):
        angles = np.radians(np.concatenate((np.linspace(119, 137, 1000), np.linspace(167, 191, 2000))))
        x_values, y_values = .6 * np.cos(angles), .6 * np.sin(angles)
        result = ha.fit_origin_separator(x_values, y_values, angle_bounds_deg=(130., 170.))
        self.assertTrue(result["interior_minimum"])
        self.assertTrue(145 < result["angle_deg"] < 160)
        self.assertAlmostEqual(np.linalg.norm(result["normal"]), 1.)
        direction = np.array([np.cos(np.radians(result["angle_deg"])), np.sin(np.radians(result["angle_deg"]))])
        self.assertAlmostEqual(float(direction @ result["normal"]), 0.)
        self.assertEqual(result["annulus_samples"], 3000)
        with self.assertRaises(ValueError):
            ha.fit_origin_separator([0.], [0.])

    def test_A_radius_uses_right_side_without_changing_input(self):
        x_values = np.array([.01, .02, .03, .04, -.7, .8, np.nan])
        y_values = np.zeros_like(x_values)
        original = x_values.copy()
        result = ha.fit_origin_core_radius(x_values, y_values, quantile=.75, right_radius_limit=.3)
        self.assertAlmostEqual(result["radius"], .0325)
        self.assertEqual(result["selected_samples"], 4)
        np.testing.assert_array_equal(x_values, original)

    def test_ellipse_clearance_uses_full_boundary(self):
        region = {"center": (1., .4), "semiaxes": (.3, .15), "angle_deg": 32.}
        result = ha.ellipse_region_clearances(region, .2, (0., 1.), 1)
        phase = np.linspace(0, 2 * np.pi, 200000)
        angle = np.radians(region["angle_deg"])
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        boundary = (np.column_stack((.3 * np.cos(phase), .15 * np.sin(phase))) @ rotation.T
                    + region["center"])
        self.assertAlmostEqual(result["disk_gap"], np.linalg.norm(boundary, axis=1).min() - .2, places=8)
        self.assertAlmostEqual(result["line_gap"], boundary[:, 1].min(), places=8)
        overlapping = {"center": (0., 0.), "semiaxes": (.4, .2)}
        self.assertEqual(ha.ellipse_region_clearances(overlapping, .2, (1., 0.), 1)["disk_gap"], -.2)

    def test_equal_area_fit_enforces_both_clearances(self):
        rng = np.random.default_rng(5)
        points = np.vstack((rng.normal((.5, .45), (.04, .03), size=(1200, 2)),
                            rng.normal((-.5, -.45), (.04, .03), size=(1000, 2))))
        fitted = ha.fit_equal_area_ellipses(points[:, 0], points[:, 1], .12, (0., 1.),
                                           histogram_step=.015, maxiter=80, popsize=8, clearance=.01)
        for name, side in (("B", 1), ("C", -1)):
            region = fitted["regions"][name]
            self.assertAlmostEqual(np.pi * np.prod(region["semiaxes"]), fitted["area"])
            gaps = ha.ellipse_region_clearances(region, .12, (0., 1.), side)
            self.assertGreaterEqual(gaps["line_gap"], .01 - 1e-10)
            self.assertGreaterEqual(gaps["disk_gap"], .01 - 1e-10)
            self.assertGreater(fitted["diagnostics"][name]["histogram_count"], 800)


class RegionResidenceTests(unittest.TestCase):
    def test_core_labels_keep_gaps_and_invalid_samples(self):
        regions = {
            "A": {"center": (0., 0.), "semiaxes": (.2, .1)},
            "B": {"center": (-1., 0.), "semiaxes": (.1, .2)},
            "C": {"center": (0., 1.), "semiaxes": (.3, .05), "angle_deg": 90.},
        }
        x_values = np.array([0., -1., 0., -.5, np.nan, 0., .15, .25])
        y_values = np.array([0., 0., 1.2, 0., 0., 0., 0., 0.])
        original = x_values.copy()
        valid = np.array([True, True, True, True, True, False, True, True])
        result = ha.classify_region_cores(x_values, y_values, regions, valid=valid)
        np.testing.assert_array_equal(result["labels"], [0, 1, 2, -1, -2, -2, 0, -1])
        self.assertEqual(result["region_names"], ("A", "B", "C"))
        np.testing.assert_array_equal(x_values, original)
        smaller = ha.classify_region_cores(x_values, y_values, regions, valid=valid, scale=.5)
        self.assertEqual(smaller["labels"][6], -1)

    def test_overlapping_and_malformed_cores_are_rejected(self):
        ellipse = {"center": (0., 0.), "semiaxes": (1., 1.)}
        with self.assertRaisesRegex(ValueError, "overlaps"):
            ha.classify_region_cores([0.], [0.], {"A": ellipse, "B": ellipse})
        with self.assertRaises(ValueError):
            ha.classify_region_cores([0.], [0.], {"A": {"center": (0., 0.), "semiaxes": (-1., 1.)}})
        with self.assertRaises(ValueError):
            ha.classify_region_cores([0.], [0.], {"A": ellipse}, valid=[True, True])

    def test_occupancy_is_independent_of_commitment_and_gaps_are_not_filled(self):
        labels = np.array([-1, 0, 0, -1, 0, -1, 1, 1, 1, -2, 1, 1, -1, 2, 2, 2])
        original = labels.copy()
        times = np.arange(len(labels)) / 1000.
        unfiltered = ha.region_residence_statistics(labels, times, 1000., ("A", "B", "C"))
        committed = ha.region_residence_statistics(labels, times, 1000., ("A", "B", "C"), min_visit_samples=2)
        for name in ("A", "B", "C", "unassigned", "invalid"):
            self.assertEqual(unfiltered["summary"][name]["time_s"], committed["summary"][name]["time_s"])
        self.assertAlmostEqual(committed["summary"]["A"]["fraction_valid"], 3 / 15)
        self.assertEqual(committed["summary"]["unassigned"]["samples"], 4)
        self.assertEqual(committed["summary"]["invalid"]["samples"], 1)
        self.assertEqual(committed["summary"]["A"]["n_runs"], 2)
        self.assertEqual(committed["summary"]["A"]["n_short_runs"], 1)
        np.testing.assert_allclose(committed["summary"]["A"]["complete_durations_s"], [.002])
        self.assertEqual(committed["summary"]["B"]["n_censored_runs"], 2)
        self.assertEqual(committed["summary"]["B"]["n_retained_complete_runs"], 0)
        self.assertEqual(committed["summary"]["C"]["n_censored_runs"], 1)
        self.assertAlmostEqual(sum(row["time_s"] for row in committed["summary"].values()), .016)
        np.testing.assert_array_equal(labels, original)

    def test_time_gaps_censor_visits_without_counting_missing_time(self):
        labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        times = np.arange(8) / 1000.
        times[3:] += .005
        result = ha.region_residence_statistics(labels, times, 1000., ("A", "B"))
        np.testing.assert_array_equal(result["runs"]["start_idx"], [0, 3, 6])
        np.testing.assert_array_equal(result["runs"]["left_censored"], [True, True, False])
        np.testing.assert_array_equal(result["runs"]["right_censored"], [True, False, True])
        self.assertAlmostEqual(result["observed_time_s"], .008)
        self.assertAlmostEqual(result["span_s"], .013)
        self.assertEqual(result["time_gaps"], 1)

    def test_empty_and_invalid_recordings_have_no_complete_visits(self):
        for labels in (np.array([], dtype=int), np.array([-2, -2], dtype=int)):
            result = ha.region_residence_statistics(labels, np.arange(len(labels)) / 1000., 1000., ("A",))
            self.assertEqual(result["valid_time_s"], 0.)
            self.assertEqual(result["summary"]["A"]["n_retained_complete_runs"], 0)
            self.assertTrue(np.isnan(result["summary"]["A"]["fraction_valid"]))


class RegionTransitionTests(unittest.TestCase):
    def test_BC_passages_keep_brief_A_contacts(self):
        labels = np.array([-1, 1, 1, 1, -1, 2, 2, 2, -1, 0, 0, 0, -1,
                           1, 1, 1, -1, 0, -1, 2, 2, 2, -2, 1, 1, 1])
        times = np.arange(len(labels)) / 1000.
        residence = ha.region_residence_statistics(labels, times, 1000., ("A", "B", "C"), min_visit_samples=3)
        result = ha.region_transition_counts(residence)
        self.assertEqual([(event["from"], event["to"], event["category"])
                          for event in result["bc_passages"]],
                         [("B", "C", "no_A_sample"), ("C", "B", "committed_A"), ("B", "C", "brief_A")])
        self.assertEqual(result["bc_passages"][-1]["a_samples"], 1)
        self.assertEqual(result["bc_counts"]["B->C"]["no_A_sample"], 1)
        self.assertEqual(result["counts"][1, 2], 2)
        shorter = ha.region_transition_counts(ha.region_residence_statistics(labels, times, 1000., ("A", "B", "C")))
        self.assertEqual(shorter["bc_counts"]["B->C"]["no_A_sample"], 1)
        self.assertEqual(shorter["bc_counts"]["B->C"]["committed_A"], 1)

    def test_BC_passages_do_not_cross_time_gaps(self):
        labels = np.array([1, 1, 1, -1, 2, 2, 2])
        times = np.arange(len(labels)) / 1000.
        times[4:] += .01
        result = ha.region_transition_counts(ha.region_residence_statistics(labels, times, 1000., ("A", "B", "C"), min_visit_samples=2))
        self.assertEqual(result["bc_passages"], [])
        self.assertEqual(result["counts"].sum(), 0)

    def test_last_same_endpoint_visit_restarts_passage(self):
        labels = np.array([1, 1, 0, 0, 1, 1, -1, 2, 2])
        residence = ha.region_residence_statistics(labels, np.arange(len(labels)) / 1000., 1000., ("A", "B", "C"), min_visit_samples=2)
        passages = ha.region_transition_counts(residence)["bc_passages"]
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["start_idx"], 5)
        self.assertEqual(passages[0]["category"], "no_A_sample")


class RotationControlTests(unittest.TestCase):
    def test_rigid_rotation_preserves_shape_clearances_and_A(self):
        helpers = notebook_functions("rotate_region_frame", "choose_opposite_side_rotation")
        regions = {
            "A": {"center": (0., 0.), "semiaxes": (.2, .2), "angle_deg": 0.},
            "B": {"center": (-.5, .4), "semiaxes": (.3, .04 / .3), "angle_deg": 30.},
            "C": {"center": (-.5, -.4), "semiaxes": (.25, .16), "angle_deg": -20.},
        }
        original = json.loads(json.dumps(regions))
        control = helpers["choose_opposite_side_rotation"](regions, (0., 1.), clearance=.01, margin_deg=2.)
        self.assertGreater(control["old_line_c_gap"], .01)
        self.assertAlmostEqual(control["ccw_deg"], control["minimum_ccw_deg"] + 2.)
        for name, side in (("B", 1), ("C", -1)):
            rotated = control["regions"][name]
            np.testing.assert_array_equal(rotated["semiaxes"], regions[name]["semiaxes"])
            self.assertAlmostEqual(np.linalg.norm(rotated["center"]), np.linalg.norm(regions[name]["center"]))
            before = ha.ellipse_region_clearances(regions[name], .2, (0., 1.), side)
            after = ha.ellipse_region_clearances(rotated, .2, control["normal"], side)
            np.testing.assert_allclose(list(after.values()), list(before.values()), atol=1e-10)
        np.testing.assert_array_equal(control["regions"]["A"]["center"], regions["A"]["center"])
        np.testing.assert_array_equal(control["regions"]["A"]["semiaxes"], regions["A"]["semiaxes"])
        for name in regions:
            np.testing.assert_array_equal(regions[name]["center"], original[name]["center"])
            np.testing.assert_array_equal(regions[name]["semiaxes"], original[name]["semiaxes"])
        too_small = helpers["rotate_region_frame"](regions, (0., 1.), control["minimum_ccw_deg"] - .1)
        self.assertLess(ha.ellipse_region_clearances(too_small["regions"]["C"], .2, (0., 1.), 1)["line_gap"], .01)
        with self.assertRaises(ValueError):
            helpers["choose_opposite_side_rotation"](regions, (0., 1.), clearance=.01, ccw_override=1.)

    def test_control_normalization_keeps_unobserved_fraction_undefined(self):
        helper = notebook_functions("compare_region_rotations")["compare_region_rotations"]
        labels = np.array([-1, 1, 1, 1, -1, 2, 2, 2, -1, 0, 0, 0, -1, 1, 1, 1, -1, 0, -1, 2, 2, 2])
        times = np.arange(len(labels)) / 1000.
        result = helper({"original": labels, "same": labels.copy()}, times, 1000., (1000., 3000.))
        original_rows = [row for row in result["passages"] if row["condition"] == "original"]
        same_rows = [row for row in result["passages"] if row["condition"] == "same"]
        for original_row, same_row in zip(original_rows, same_rows):
            for key in original_row.keys() - {"condition"}:
                self.assertEqual(original_row[key], same_row[key])
        selected = next(row for row in original_rows if row["minimum_us"] == 3000 and row["direction"] == "B->C")
        self.assertEqual(selected["total_passages"], 2)
        self.assertEqual(selected["no_A"], 1)
        self.assertEqual(selected["brief_A"], 1)
        self.assertEqual(selected["no_A_fraction"], .5)
        self.assertEqual(selected["source_samples"], 6)
        self.assertAlmostEqual(selected["no_A_per_source_s"], 1 / .006)
        self.assertAlmostEqual(selected["no_A_per_1000_source_visits"], 500.)
        no_endpoint = helper({"empty_B": np.zeros(10, dtype=int)}, np.arange(10) / 1000., 1000., (1000.,))
        for row in no_endpoint["passages"]:
            self.assertEqual(row["total_passages"], 0)
            self.assertTrue(np.isnan(row["no_A_fraction"]))
            self.assertTrue(np.isnan(row["no_A_per_source_s"]))
            self.assertTrue(np.isnan(row["no_A_per_1000_source_visits"]))


class NotebookTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_historical_BC_examples_are_fixed_not_reselected(self):
        notebook_path = Path(__file__).with_name("Unit_sphere_viewer.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        assignments = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                assignments.extend(node for node in tree.body if isinstance(node, ast.Assign)
                                   and any(isinstance(target, ast.Name) and target.id == "PRESERVED_BC_ARCHIVE"
                                           for target in node.targets))
        self.assertEqual(len(assignments), 1)
        namespace = {}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(notebook_path), "exec"), namespace)
        archive = namespace["PRESERVED_BC_ARCHIVE"]
        self.assertEqual(archive["historical_commitment_us"], 40.)
        expected = [(2950730, 2950742, "B->C"), (2950751, 2950756, "C->B")]
        for event, (start, end, direction) in zip(archive["events"], expected):
            self.assertEqual((event["start_idx"], event["end_idx"], event["direction"]), (start, end, direction))
            self.assertEqual(len(event["reference_ax"]), end - start + 1)
            labels = ha.classify_region_cores(event["reference_ax"], event["reference_ay"], archive["regions"])["labels"]
            self.assertFalse(np.any(labels == 0))
            self.assertEqual(labels[0], 1 if direction == "B->C" else 2)
            self.assertEqual(labels[-1], 2 if direction == "B->C" else 1)

    def test_preserved_BC_recovery_verifies_native_trace(self):
        helper = notebook_functions("recover_preserved_bc_events")["recover_preserved_bc_events"]
        recording = {"c0": np.linspace(1., 3., 100), "c90": np.ones(100),
                     "c45": np.full(100, 2.), "c135": np.ones(100),
                     "t": np.arange(100) / 250000., "fps": 250000.}
        native = ha.block_channel_means(recording, 1)
        event = {"direction": "B->C", "start_idx": 20, "end_idx": 32,
                 "reference_ax": native[0][20:33].tolist(), "reference_ay": native[1][20:33].tolist()}
        archive = {"rate_hz": 250000., "events": [event]}
        recovered = helper(recording, archive, context_us=8.)[0]
        np.testing.assert_array_equal(recovered["native_indices"], np.arange(18, 35))
        np.testing.assert_array_equal(recovered["ax"][recovered["path_slice"]], native[0][20:33])
        np.testing.assert_array_equal(recovered["t"][recovered["path_slice"]], recording["t"][20:33])
        with self.assertRaises(AssertionError):
            helper(recording, {**archive, "events": [{**event, "reference_ax": np.zeros(13).tolist()}]})
        with self.assertRaises(ValueError):
            helper(recording, {**archive, "rate_hz": 1000.})

    def test_notebook_has_no_external_helper_import_or_reload(self):
        notebook_path = Path(__file__).with_name("Unit_sphere_viewer.ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            self.assertNotIn("hinge_analysis", source)
            self.assertNotIn("importlib.reload", source)
        with patch.dict("sys.modules", {"hinge_analysis": None}):
            helpers = load_embedded_helpers()
            self.assertAlmostEqual(helpers.r_max_of(*helpers.fourkas_ABC()), .9268984950974503)

    def test_BC_screen_flags_possible_between_sample_A_crossing(self):
        helper = notebook_functions("screen_bc_passage")["screen_bc_passage"]
        passage = {"start_idx": 0, "end_idx": 1, "category": "no_A_sample"}
        screened = helper(passage, np.array([-.5, .5]), np.zeros(2), .2)
        self.assertEqual(screened["min_sample_radius"], .5)
        self.assertEqual(screened["min_connector_radius"], 0.)
        self.assertTrue(screened["connector_intersects_A"])
        self.assertTrue(screened["adjacent_core_samples"])
        around = helper({**passage, "end_idx": 2}, np.array([-.5, -.5, 0.]), np.array([0., .5, .5]), .2)
        self.assertFalse(around["connector_intersects_A"])
        self.assertFalse(around["adjacent_core_samples"])
        self.assertNotIn("path_samples", passage)

    def test_time_spent_map_conserves_valid_exposure_and_marginals(self):
        helper = notebook_functions("anisotropy_time_map")["anisotropy_time_map"]
        x_values = np.array([-.1, -.1, .1, .1, .1, np.nan, .1])
        y_values = np.array([-.1, -.1, .1, .1, .1, 0., .1])
        valid = np.array([True, True, True, True, True, True, False])
        result = helper(x_values, y_values, 250000., bin_width=.02, valid=valid)
        self.assertEqual(result["valid_samples"], 5)
        self.assertEqual(result["excluded_samples"], 2)
        self.assertAlmostEqual(result["total_time_s"], 20e-6)
        self.assertAlmostEqual(result["excluded_time_s"], 8e-6)
        np.testing.assert_allclose(result["seconds"], result["counts"] * 4e-6)
        np.testing.assert_allclose(result["time_by_x"], result["seconds"].sum(axis=1))
        np.testing.assert_allclose(result["time_by_y"], result["seconds"].sum(axis=0))
        np.testing.assert_allclose(np.diff(result["x_edges"]), .02)
        np.testing.assert_allclose(np.diff(result["y_edges"]), .02)
        coarse = helper(x_values, y_values, 250000., bin_width=.2, valid=valid)
        self.assertAlmostEqual(coarse["seconds"].sum(), result["seconds"].sum())

    def test_time_spent_map_handles_a_single_location_and_no_valid_data(self):
        helper = notebook_functions("anisotropy_time_map")["anisotropy_time_map"]
        result = helper([0., 0., 0.], [0., 0., 0.], 1000.)
        np.testing.assert_allclose(result["seconds"], [[.003]])
        with self.assertRaises(ValueError):
            helper([np.nan], [0.], 1000.)
        with self.assertRaises(ValueError):
            helper([0.], [0.], 1000., bin_width=0.)

    def test_pinned_native_event_preserves_all_samples_and_alignment(self):
        helper = notebook_functions("recover_native_event")["recover_native_event"]
        recording = {"c0": np.linspace(1., 5., 1000), "c90": np.ones(1000),
                     "c45": np.linspace(2., 4., 1000), "c135": np.ones(1000),
                     "t": np.arange(1000) / 250000., "fps": 250000.}
        original = ha.block_channel_means(recording, 20)
        anchor = {"native_rate_hz": 250000., "block_size": 20,
                  "start_native": 300, "end_native_exclusive": 420,
                  "confirm_native_exclusive": 460,
                  "reference_ax": original[0][15:21].copy(),
                  "reference_ay": original[1][15:21].copy()}
        recovered = helper(recording, anchor, context_s=40e-6)
        np.testing.assert_array_equal(recovered["native_indices"], np.arange(280, 480))
        np.testing.assert_array_equal(recovered["native"][2], recording["t"][280:480])
        np.testing.assert_allclose(np.diff(recovered["native"][2]), 4e-6, atol=1e-15)
        np.testing.assert_allclose(recovered["native"][0],
                                   (recording["c0"][280:480] - 1) / (recording["c0"][280:480] + 1))
        self.assertEqual(recovered["event_mask"].sum(), 120)
        self.assertEqual(recovered["confirmation_mask"].sum(), 40)
        np.testing.assert_allclose(recovered["averaged"][0][recovered["original_slice"]], anchor["reference_ax"])
        with self.assertRaises(ValueError):
            helper(recording, {**anchor, "native_rate_hz": 1000.})
        with self.assertRaises(ValueError):
            helper(recording, {**anchor, "start_native": 301})
        with self.assertRaises(AssertionError):
            helper(recording, {**anchor, "reference_ax": np.zeros(6)})

    def test_focused_core_contacts_are_counted_at_native_rate(self):
        helper = notebook_functions("focused_core_returns")["focused_core_returns"]
        distances = np.array([.05, .04, .5, .08, .07, .4, .2, .18, .5])
        times = np.arange(len(distances)) / 250000.
        valid = np.ones(len(distances), dtype=bool)
        valid[6] = False
        returns = helper(distances, times, valid, np.ones(len(times), dtype=bool), [.12, .30], 250000.)
        self.assertEqual([(visit["fraction"], visit["start_idx"], visit["stop_idx"])
                          for visit in returns], [(.12, 3, 5), (.30, 3, 5), (.30, 7, 8)])
        self.assertAlmostEqual(returns[0]["duration_s"], 8e-6)
        self.assertAlmostEqual(returns[-1]["duration_s"], 4e-6)
        self.assertAlmostEqual(returns[0]["exit_s"], times[5])

    def test_stage_specific_optics_preserve_measured_anisotropy(self):
        helper = notebook_functions("reconstruct_stages")["reconstruct_stages"]
        coefficients = {"full": ha.fourkas_ABC(), "fresnel": ha.fourkas_ABC_fresnel()}
        input_angles = np.array([0., 20., 55., 75., 89., 90.])
        sine_squared = np.sin(np.radians(input_angles))**2
        coeff_a, coeff_b, coeff_c = coefficients["fresnel"]
        radii = coeff_c * sine_squared / (coeff_a + coeff_b * sine_squared)
        x_values = radii * np.cos(.6)
        y_values = radii * np.sin(.6)
        x_before, y_before = x_values.copy(), y_values.copy()
        stages = {stage: (x_values, y_values, np.arange(6) / 1000., 1000.)
                  for stage in ("raw", "corrected", "fresnel")}
        flags = {stage: {"valid": np.ones(6, dtype=bool)} for stage in stages}
        models = {"raw": "full", "corrected": "full", "fresnel": "fresnel"}
        theta, validity, rmax = helper(stages, flags, models, coefficients)
        np.testing.assert_allclose(theta["fresnel"], input_angles, atol=1e-6)
        self.assertTrue(validity["fresnel"]["valid_angle"].all())
        self.assertFalse(validity["corrected"]["valid_angle"][-1])
        self.assertTrue(np.isnan(theta["corrected"][-1]))
        self.assertGreater(abs(theta["fresnel"][2] - theta["corrected"][2]), .5)
        for stage in stages:
            self.assertEqual(rmax[stage], ha.r_max_of(*coefficients[models[stage]]))
        np.testing.assert_array_equal(x_values, x_before)
        np.testing.assert_array_equal(y_values, y_before)

    def test_rejected_peak_fits_remain_missing(self):
        helper = notebook_functions("two_peak_trace")["two_peak_trace"]

        class RejectedMixture:
            def __init__(self, *args, **kwargs):
                self.means_ = np.array([[20.], [60.]])
                self.weights_ = np.array([.99, .01])
                self.converged_ = True

            def fit(self, values):
                return self

        _, axis = plt.subplots()
        angles = np.tile([20., 60.], 100)
        with patch("sklearn.mixture.GaussianMixture", RejectedMixture):
            result = helper(angles, np.arange(len(angles)) / 1000., 1000., axis, "test")
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["peaks"]).all())
        self.assertEqual(len(axis.lines), 0)

    def test_off_origin_circle_is_diagnostic(self):
        helpers = notebook_functions("envelope", "rim_fit")
        phase = np.linspace(0, 2 * np.pi, 18000, endpoint=False)
        x_values = -.04 + .74 * np.cos(phase)
        y_values = .025 + .74 * np.sin(phase)
        x_before, y_before = x_values.copy(), y_values.copy()
        result = helpers["rim_fit"](x_values, y_values)
        np.testing.assert_allclose(result["center"], [-.04, .025], atol=1e-3)
        self.assertAlmostEqual(result["R"], .74, delta=1e-3)
        np.testing.assert_array_equal(x_values, x_before)
        np.testing.assert_array_equal(y_values, y_before)
        with self.assertRaises(ValueError):
            helpers["rim_fit"](np.array([]), np.array([]))


if __name__ == "__main__":
    unittest.main()