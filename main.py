#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' is importable even in atypical setups
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.pipeline import BuildingFootprintPipeline
from src.benchmark import PerformanceBenchmark, ExperimentRunner


def main():
	parser = argparse.ArgumentParser(description="Building Footprint Pipeline")
	parser.add_argument("--mode", choices=["single", "experiment", "benchmark", "demo", "multistate", "table", "compare6", "proof3d", "state-eval"], default="single")
	parser.add_argument("--real-table", action="store_true", help="Generate real-results summary table from multi-state data")
	# new multi-state mode
 
 
	parser.add_argument("--multistate", action="store_true", help="Run multi-state real-data RL fusion demo")
	parser.add_argument("--n-states", type=int, default=10, help="Number of states to include for multistate mode")
	parser.add_argument("--patches-per-state", type=int, default=10, help="Max patches per state for multistate mode")
	parser.add_argument("--state", type=str, default=None, help="State folder name under data dir")
	parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
	parser.add_argument("--samples", type=int, default=20, help="Demo synthetic samples")
	parser.add_argument("--iters", type=int, default=50, help="RL training iterations (demo)")
	parser.add_argument("--allow-cpu-slow-train", action="store_true", help="If set, do not limit epochs on CPU")
	parser.add_argument("--max-states", type=int, default=10, help="Max number of states to process in multistate mode")
	parser.add_argument("--cap-per-state", type=int, default=12, help="Cap of patches per state in multistate mode")
	args = parser.parse_args()

	cfg = Config()
	cfg.NUM_EPOCHS = args.epochs
	cfg.ALLOW_SLOW_TRAIN_ON_CPU = args.allow_cpu_slow_train
	pipeline = BuildingFootprintPipeline(cfg)

	if args.mode == "demo" and not args.multistate:
		results, metrics = pipeline.run_demo(n_samples=args.samples, rl_iters=args.iters)
		print("\nFinal Metrics (aggregated):")
		for k, v in metrics.items():
			if not k.endswith("_std"):
				print(f"  {k}: {v:.4f}")
		print(f"\nTotal polygons: {sum(len(r.get('polygons', [])) for r in results)}")
		return

	if args.mode == "single" and not args.multistate:
		# Full end-to-end pipeline with real data (if available)
		results, metrics = pipeline.run_complete_pipeline(state_name=args.state, download_data=False)
		print("\nFinal Metrics (aggregated):")
		for k, v in metrics.items():
			if not k.endswith("_std"):
				print(f"  {k}: {v:.4f}")
		print(f"\nTotal polygons: {sum(len(r.get('polygons', [])) for r in results)}")
		return

	if args.mode == "multistate":
		from src.global_vis import save_global_visualizations, plot_us_improvement_map
		summary = pipeline.run_multistate_real_demo(n_states=args.n_states, patches_per_state=args.patches_per_state)
		if summary and isinstance(summary, dict) and "table" in summary:
			# Save CSV-like table to a pandas CSV as well
			import pandas as pd
			df = pd.DataFrame(summary["table"]).sort_values("improvement", ascending=False)
			csv_path = pipeline.config.LOGS_DIR / "multistate_rl_summary.csv"
			df.to_csv(csv_path, index=False)
			# Global visuals
			plot_us_improvement_map(summary["table"], pipeline.config.FIGURES_DIR / "us_improvement_map.png")
			save_global_visualizations(csv_path, pipeline.config.FIGURES_DIR)
			print(df.to_string(index=False, float_format='%.4f'))
		return

	if args.mode == "table":
		if args.real_table:
			res = pipeline.run_real_results_summary_table(n_states=args.n_states, patches_per_state=args.patches_per_state)
			if isinstance(res, dict) and res.get("table") is not None:
				print("Saved real-results table:")
				print(res["table"].to_string(index=False, float_format='%.4f'))
				print(f"CSV: {res['csv']}\nPNG: {res['png']}")
		else:
			from src.reports import create_demo_result_table
			df, csv_path, png_path = create_demo_result_table(pipeline, n_samples=args.samples)
			print("Saved demo table:")
			print(df.to_string(index=False, float_format='%.4f'))
			print(f"CSV: {csv_path}\nPNG: {png_path}")
		return

	if args.mode == "compare6":
		res = pipeline.run_six_method_real_comparison(n_states=args.n_states, patches_per_state=args.patches_per_state)
		if isinstance(res, dict) and res.get("table") is not None:
			print("Saved six-method comparison table:")
			print(res["table"].to_string(index=False, float_format='%.4f'))
			print(f"CSV: {res['csv']}\nPNG: {res['png']}\nPARAMS: {res['params']}")
		return

	if args.mode == "proof3d":
		from src.visual_proofs import save_input_mask_3d_proofs
		paths = save_input_mask_3d_proofs(cfg, n_states=args.n_states, patches_per_state=args.patches_per_state)
		if paths:
			print("Saved 3D proof figures:")
			for p in paths:
				print(p)
		else:
			print("No figures generated (no states or patches found).")
		return

	if args.mode == "state-eval":
		state = args.state or "RhodeIsland"
		res = pipeline.run_single_state_real_eval(state_name=state, patches_per_state=args.patches_per_state, rl_iters=args.iters)
		if isinstance(res, dict) and res.get("table") is not None:
			print(res["table"].to_string(index=False, float_format='%.4f'))
			print(f"CSV: {res['csv']}\nPNG: {res['png']}")
			print("Raw summary:")
			print(res["raw"]) 
		return
	if args.mode == "benchmark" and not args.multistate:
		# Prepare some data via demo generator for benchmarking consistency
		rough, gts = pipeline.create_synthetic_data(20)
		bench = PerformanceBenchmark(cfg)
		baseline_masks = {
			"threshold": [bench.simple_threshold_baseline(p) for p in rough],
			"morphology": [bench.morphology_baseline(p) for p in rough],
			"watershed": [bench.watershed_baseline(p) for p in rough],
		}
		for name, masks in baseline_masks.items():
			m = bench.evaluate_baseline(masks, gts)
			print(f"{name:12} IoU={m['iou']:.4f} Precision={m['precision']:.4f} Recall={m['recall']:.4f} F1={m['f1_score']:.4f}")
		# our pipeline
		reg = pipeline.step4_hybrid_regularization(rough, gts)
		fused, _ = pipeline.step5_adaptive_fusion(reg, training_iterations=30)
		our_masks = [r["fused"] for r in fused]
		our_m = bench.evaluate_baseline(our_masks, gts)
		print(f"{'our':12} IoU={our_m['iou']:.4f} Precision={our_m['precision']:.4f} Recall={our_m['recall']:.4f} F1={our_m['f1_score']:.4f}")
		return

	if args.mode == "experiment" and not args.multistate:
		runner = ExperimentRunner()
		# run same demo with varied seeds/configs (placeholder)
		runner.run_experiment("balanced_demo", cfg, run_callable=lambda: pipeline.run_demo(n_samples=20, rl_iters=50))
		df = runner.compare_experiments()
		if df is not None:
			print(df.to_string(index=False, float_format='%.4f'))
		return


if __name__ == "__main__":
	main()

