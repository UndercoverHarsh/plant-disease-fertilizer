# Plant Disease Detection & Fertilizer Recommendation — MVP

## Included
- Streamlit web interface
- Leaf image upload and preview
- Demo/manual disease-label workflow
- Rule-based disease-management guidance
- Soil pH and nitrogen-input form
- Optional TensorFlow model inference pathway

## Run
Install Python 3.10 or 3.11, then in this folder:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important limitation
Automatic AI diagnosis is **not active until you add a trained model**. Demo/manual mode demonstrates the recommendation workflow but does not analyze the image. Never present demo output as a real prediction.

To enable inference, put a trained Keras model at `models/plant_disease_model.keras`. It must accept RGB images resized to 224x224 and output scores in the same order as the `DISEASES` list in `app.py`. Validate on held-out data and report accuracy, precision, recall, F1-score, and a confusion matrix before claiming performance.

## Suggested 50% milestone demonstration
1. Upload a leaf image.
2. Demonstrate the manual-label recommendation workflow.
3. Enter soil pH / N test information.
4. Explain that trained-model integration and validation are the next milestones.

Educational prototype only; not a validated agricultural diagnostic or fertilizer-prescription tool.
