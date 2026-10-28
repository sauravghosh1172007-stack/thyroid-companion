# Thyroid Companion — Streamlit Community Cloud deployment

## Files required
- `app.py`
- `thyroid_symptom_model.joblib`
- `requirements.txt`
- `results/.gitkeep`

The supplied model was serialized with scikit-learn 1.9.1, so the requirements pin that version.

## Deploy
1. Create a GitHub repository, for example `thyroid-companion`.
2. Upload the *contents* of this folder to the repository root (not this folder nested inside another folder).
3. Confirm `app.py`, `requirements.txt`, and `thyroid_symptom_model.joblib` are visible at the root on GitHub.
4. Visit https://share.streamlit.io/ and sign in using GitHub.
5. Select **Create app**, choose the repository, branch `main`, and main file path `app.py`.
6. Under **Advanced settings**, choose Python 3.12 if available. Deploy.
7. Once the deployment finishes, copy the generated `*.streamlit.app` URL to share the site.

## Notes
- Do not remove or rename `thyroid_symptom_model.joblib`; it must be beside `app.py`.
- The app can run without training scripts or the original CSV because it loads the saved model.
- The `results/` evaluation reports were not included in the supplied files, so the Model Dashboard will show a missing-report note until those reports are added. This does not prevent the symptom checker from running.
- The app is an educational prototype trained on synthetic labels, not a clinical diagnostic tool. Do not enter real personal or health information into a public demo.
- Feedback saved to the app's local filesystem may not persist permanently across redeployments or restarts.
