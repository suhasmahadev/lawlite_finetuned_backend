from django.test import TestCase



# Create your tests here.
# test_local_gen.py
import os, sys
from summarizer.utils import ml
model_dir = os.getenv("HF_LOCAL_MODEL_DIR")
if not model_dir:
    print("Set HF_LOCAL_MODEL_DIR and retry.")
    sys.exit(1)

txt = "This is a short contract. Party A agrees to deliver X on date Y. Payment terms are Z. Liability limited..."
try:
    out = ml._summarize_with_local_model(txt, local_dir=model_dir)
    print("LOCAL SUMMARY:\n", out)
except Exception as e:
    print("LOCAL MODEL ERROR:", e)
