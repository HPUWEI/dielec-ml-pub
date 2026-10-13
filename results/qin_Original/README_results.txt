DielectroNet Nested Repeated 5-fold CV Results
===============================================

1. repeated_5fold_cv_summary.csv / repeated_5fold_cv_results.xlsx
   Six-model Mean ± SD from 25 outer-fold evaluations.

2. repeated_5fold_cv_all_results.csv
   R2, RMSE and MAE for all 25 outer folds of all six models.

3. all_models_25_outer_fold_predictions.csv
   Raw outer-fold predictions for every sample, repeat and fold.

4. <model>_averaged_oof_predictions.csv
   For each sample: five repeat-level OOF predictions, their mean, and the sample-level prediction SD across the five repeats.

5. all_models_averaged_oof_predictions.xlsx
   The six model-level averaged OOF tables plus OOF metrics.

6. <model>_averaged_oof_observed_vs_predicted.png/.svg
   Observed vs predicted plots based on averaged OOF predictions.
   The dashed diagonal is the ideal prediction line.

7. all_models_averaged_oof_metrics.csv
   R2/RMSE/MAE calculated from the 252 averaged OOF predictions for each model, plus summary statistics of sample-level prediction SD.

8. <best_model>_pipeline.pkl / *_features.pkl / *_best_params.txt
   Final full-data models and their selected parameters for platform use.

9. Feature-analysis files, *_meta.pkl and *_deepseek_*.txt
   Generated for the model selected by the highest mean outer-fold R2.
