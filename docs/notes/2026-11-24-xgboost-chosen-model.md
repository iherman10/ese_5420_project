# XGBoost chosen model
Created: 2026-11-24
Scope: The setting and tree count chosen from XGBoost tuning, and the edge check from `2026-11-13-xgboost.md`.

## Choice

Best mean CV AUC: `max_depth = 6`, `min_child_weight = 1`, `subsample = colsample_bytree = 0.8`, 1,198 trees. CV AUC 0.7900 (SD 0.0023), Brier 0.1398.

## Edge check

- **Trees:** 1,198 is well under the 2,000 cap, so the cap stays.
- **`max_depth`:** 6 is inside {4, 6, 8}, so no change.
- **`min_child_weight`:** 1 is the bottom of {1, 10}. The plan only says how to extend it upward, and its lead over 10 at the same depth and subsampling is 0.0004 AUC, far below the fold SD of 0.0023. The range isn't extended downward, and 1 is kept.
