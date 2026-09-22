"""
Main orchestration script for training the credit card fraud detection model.

MENTOR NOTE:
An orchestrator ties all independent modules together into a reproducible pipeline.
In production, this might be replaced or wrapped by tools like Airflow, Kubeflow,
or MLflow Recipes. A good orchestration script handles arguments, tracks timing,
logs progress, and saves artifacts for deployment.

Usage:
    python scripts/train.py                    # Full pipeline
    python scripts/train.py --skip-eda         # Skip EDA plots
    python scripts/train.py --tune             # Include hyperparameter tuning (slow)
    python scripts/train.py --skip-mlflow      # Skip MLflow logging
"""

import argparse
import time
import sys
import logging
import joblib
import numpy as np
from pathlib import Path

# ============================================================
# IMPORTANT: Add the project root to sys.path so we can import src.*
# This is needed when running: python scripts/train.py
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Fix Windows console encoding to handle emoji/unicode characters
# Windows defaults to cp1252 which can't render emoji. UTF-8 fixes this.
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Phase imports (from our src modules) ---
from src.config import (
    MODELS_DIR, PLOTS_DIR, LOGS_DIR, RANDOM_SEED,
    MODEL_FILENAME, SCALER_FILENAME, THRESHOLD_FILENAME,
    COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE,
    FEATURE_COLUMNS, MLFLOW_EXPERIMENT_NAME
)
from src.data_loader import load_data, get_data_summary
from src.eda import run_full_eda
from src.preprocessing import prepare_data
from src.resampling import get_resampled_datasets
from src.models import get_model_pipeline, train_model
from src.evaluation import evaluate_model, compare_models, find_optimal_threshold
from src.explainability import (
    compute_shap_values, plot_global_feature_importance,
    plot_single_prediction, explain_prediction_for_stakeholder
)
from src.experiment_tracking import setup_mlflow, log_experiment


def print_banner():
    """Prints a banner for the training script."""
    print("=" * 65)
    print("  💳  CREDIT CARD FRAUD DETECTION — TRAINING PIPELINE")
    print("=" * 65)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Train Credit Card Fraud Detection Models"
    )
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip Exploratory Data Analysis (saves ~30s)")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning (much slower)")
    parser.add_argument("--skip-mlflow", action="store_true",
                        help="Skip MLflow experiment logging")
    args = parser.parse_args()

    print_banner()
    start_time_total = time.time()

    # Ensure output directories exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # =============================================================
        # PHASE 2: Load Data & EDA
        # =============================================================
        print("\n" + "=" * 65)
        print("  📊 PHASE 2: Data Loading & EDA")
        print("=" * 65)
        t0 = time.time()
        df = load_data()
        summary = get_data_summary(df)
        print(f"  ✅ Loaded {summary['shape'][0]:,} rows × {summary['shape'][1]} cols "
              f"in {time.time() - t0:.2f}s")
        print(f"  📌 Class distribution: {summary['class_counts']}")
        print(f"  📌 Fraud rate: {summary['class_distribution'].get(1, 0):.4%}")

        if not args.skip_eda:
            logger.info("Running EDA visualizations...")
            t0 = time.time()
            run_full_eda(df)
            print(f"  ✅ EDA plots saved to {PLOTS_DIR} in {time.time() - t0:.2f}s")
        else:
            print("  ⏭️  Skipping EDA as requested.")

        # =============================================================
        # PHASE 3: Data Preprocessing
        # =============================================================
        print("\n" + "=" * 65)
        print("  🔧 PHASE 3: Data Preprocessing")
        print("=" * 65)
        t0 = time.time()
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
        print(f"  ✅ Train: {X_train.shape}, Test: {X_test.shape} in {time.time() - t0:.2f}s")
        print(f"  📌 Train fraud rate: {y_train.mean():.4%}")
        print(f"  📌 Test fraud rate:  {y_test.mean():.4%}")

        # =============================================================
        # PHASE 4: Resampling
        # =============================================================
        print("\n" + "=" * 65)
        print("  ⚖️  PHASE 4: Handling Imbalanced Data")
        print("=" * 65)
        t0 = time.time()
        resampled = get_resampled_datasets(X_train, y_train)
        for strategy, (X_rs, y_rs) in resampled.items():
            fraud_count = int(y_rs.sum())
            print(f"  📌 {strategy:15s}: {len(X_rs):>8,} samples, {fraud_count:>6,} frauds "
                  f"({y_rs.mean():.2%})")
        print(f"  ✅ Resampling complete in {time.time() - t0:.2f}s")

        # =============================================================
        # PHASE 5 & 6: Train and Evaluate All Models
        # =============================================================
        print("\n" + "=" * 65)
        print("  🤖 PHASE 5 & 6: Model Training & Evaluation")
        print("=" * 65)

        results = {}
        trained_models = {}
        models_to_train = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
        resample_strategies = ['original', 'smote']

        for model_name in models_to_train:
            for strategy in resample_strategies:
                run_name = f'{model_name}_{strategy}'
                print(f"\n  🔄 Training {run_name}...")
                t_train = time.time()

                try:
                    pipeline = get_model_pipeline(model_name)
                    X_rs, y_rs = resampled[strategy]

                    # Train
                    fitted_pipeline = train_model(pipeline, X_rs, y_rs, run_name)

                    # Evaluate
                    metrics = evaluate_model(
                        fitted_pipeline, X_test, y_test, run_name,
                        save_dir=str(PLOTS_DIR)
                    )

                    # Store results
                    results[run_name] = metrics
                    trained_models[run_name] = fitted_pipeline

                    elapsed = time.time() - t_train
                    print(f"  ✅ {run_name}: PR-AUC={metrics['pr_auc']:.4f}, "
                          f"Recall={metrics['recall']:.4f}, "
                          f"F1={metrics['f1']:.4f} ({elapsed:.2f}s)")

                except Exception as e:
                    logger.error(f"Failed to train {run_name}: {e}")
                    print(f"  ❌ {run_name} failed: {e}")

        # =============================================================
        # Compare Models & Select Best
        # =============================================================
        if not results:
            print("\n❌ No models trained successfully. Exiting.")
            sys.exit(1)

        print("\n" + "-" * 65)
        print("  📊 Model Comparison (sorted by PR-AUC)")
        print("-" * 65)
        comparison_df = compare_models(results)
        # Print only key columns
        display_cols = [c for c in ['pr_auc', 'recall', 'precision', 'f1', 'roc_auc']
                        if c in comparison_df.columns]
        print(comparison_df[display_cols].to_string())

        # Select best model by PR-AUC
        best_name = max(results, key=lambda k: results[k]['pr_auc'])
        best_model = trained_models[best_name]
        best_metrics = results[best_name]
        print(f"\n  🏆 Best model: {best_name}")
        print(f"     PR-AUC:    {best_metrics['pr_auc']:.4f}")
        print(f"     Recall:    {best_metrics['recall']:.4f}")
        print(f"     Precision: {best_metrics['precision']:.4f}")

        # =============================================================
        # Threshold Tuning
        # =============================================================
        print("\n" + "=" * 65)
        print("  🎯 PHASE 6 (cont.): Business-Cost Threshold Tuning")
        print("=" * 65)
        y_probs = best_model.predict_proba(X_test)[:, 1]
        optimal_threshold, costs = find_optimal_threshold(
            y_test, y_probs,
            cost_fn=COST_FALSE_NEGATIVE,
            cost_fp=COST_FALSE_POSITIVE,
            save_dir=str(PLOTS_DIR),
            model_name=best_name
        )
        print(f"  ✅ Optimal threshold: {optimal_threshold:.4f}")
        print(f"     (FN cost=${COST_FALSE_NEGATIVE}, FP cost=${COST_FALSE_POSITIVE})")

        # =============================================================
        # PHASE 7: Hyperparameter Tuning (optional)
        # =============================================================
        if args.tune:
            print("\n" + "=" * 65)
            print("  🔬 PHASE 7: Hyperparameter Tuning")
            print("=" * 65)
            try:
                from src.tuning import run_hyperparameter_tuning
                from sklearn.model_selection import train_test_split

                # Create validation set from training data
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size=0.15,
                    random_state=RANDOM_SEED, stratify=y_train
                )
                tuning_results = run_hyperparameter_tuning(X_tr, y_tr, X_val, y_val)

                for name, result in tuning_results.items():
                    print(f"  📌 {name}: Best PR-AUC = {result['best_score']:.4f}")
                    print(f"     Params: {result['best_params']}")
            except Exception as e:
                logger.error(f"Hyperparameter tuning failed: {e}")
                print(f"  ⚠️  Tuning failed: {e}. Continuing with default params.")

        # =============================================================
        # PHASE 8: Explainability (SHAP)
        # =============================================================
        print("\n" + "=" * 65)
        print("  🔍 PHASE 8: Model Explainability (SHAP)")
        print("=" * 65)
        try:
            # Use a small sample for SHAP (it's computationally expensive)
            X_test_sample = X_test.head(500)
            model_type = 'tree' if 'logistic' not in best_name else 'linear'

            explainer, shap_values = compute_shap_values(
                best_model, X_test_sample, model_name=model_type
            )
            plot_global_feature_importance(shap_values, X_test_sample, str(PLOTS_DIR))
            print(f"  ✅ SHAP plots saved to {PLOTS_DIR}")

            # Explain a fraud prediction for a non-technical stakeholder
            fraud_indices = y_test[y_test == 1].index.tolist()
            if fraud_indices:
                # Find a fraud case in our sample
                sample_indices = X_test_sample.index.tolist()
                fraud_in_sample = [i for i in fraud_indices if i in sample_indices]
                if fraud_in_sample:
                    idx = sample_indices.index(fraud_in_sample[0])
                    plot_single_prediction(
                        explainer, shap_values, X_test_sample, idx, str(PLOTS_DIR)
                    )
                    explanation = explain_prediction_for_stakeholder(
                        shap_values, X_test_sample, idx, FEATURE_COLUMNS
                    )
                    print(f"\n  💬 Stakeholder Explanation (sample fraud):")
                    print(f"     {explanation}")

        except Exception as e:
            logger.error(f"SHAP explainability failed: {e}")
            print(f"  ⚠️  SHAP analysis failed: {e}. Continuing.")

        # =============================================================
        # PHASE 9: MLflow Experiment Tracking
        # =============================================================
        if not args.skip_mlflow:
            print("\n" + "=" * 65)
            print("  📝 PHASE 9: MLflow Experiment Tracking")
            print("=" * 65)
            try:
                setup_mlflow()
                for run_name, metrics in results.items():
                    model_obj = trained_models[run_name]
                    # Extract classifier params from pipeline
                    try:
                        params = model_obj.named_steps['classifier'].get_params()
                        # Filter out non-serializable params
                        params = {k: str(v) for k, v in params.items()
                                  if not callable(v)}
                    except Exception:
                        params = {'model_name': run_name}

                    # Filter metrics to only numeric values
                    numeric_metrics = {k: v for k, v in metrics.items()
                                       if isinstance(v, (int, float))}

                    log_experiment(
                        model_name=run_name,
                        model=model_obj,
                        params=params,
                        metrics=numeric_metrics,
                        artifacts_dir=str(PLOTS_DIR)
                    )
                print(f"  ✅ Logged {len(results)} experiments to MLflow")
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")
                print(f"  ⚠️  MLflow logging failed: {e}. Continuing.")
        else:
            print("\n  ⏭️  Skipping MLflow logging as requested.")

        # =============================================================
        # PHASE 10: Save Best Model for Deployment
        # =============================================================
        print("\n" + "=" * 65)
        print("  💾 PHASE 10: Saving Deployment Artifacts")
        print("=" * 65)

        model_path = MODELS_DIR / MODEL_FILENAME
        scaler_path = MODELS_DIR / SCALER_FILENAME
        threshold_path = MODELS_DIR / THRESHOLD_FILENAME

        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(optimal_threshold, threshold_path)

        print(f"  ✅ Model saved:     {model_path}")
        print(f"  ✅ Scaler saved:    {scaler_path}")
        print(f"  ✅ Threshold saved: {threshold_path}")

        # =============================================================
        # FINAL SUMMARY
        # =============================================================
        total_time = time.time() - start_time_total
        print("\n" + "=" * 65)
        print("  🎉 TRAINING PIPELINE COMPLETE!")
        print("=" * 65)
        print(f"  Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"  Best model: {best_name}")
        print(f"  PR-AUC:     {best_metrics['pr_auc']:.4f}")
        print(f"  Threshold:  {optimal_threshold:.4f}")
        print()
        print("  Next steps:")
        print("  1. Start the API:  uvicorn api.app:app --reload")
        print("  2. Test with curl:")
        print('     curl -X POST http://localhost:8000/predict \\')
        print('       -H "Content-Type: application/json" \\')
        print('       -d @sample_transaction.json')
        print("  3. Run simulation:  python scripts/simulate_stream.py")
        print("  4. Run dashboard:   streamlit run scripts/run_dashboard.py")
        print()

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
