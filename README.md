

**CRISPR‑AI: Deep Learning for CRISPR‑Cas9 Off‑Target Prediction**

This repo contains the code and result files used in my MSc thesis on CRISPR‑Cas9 off‑target prediction with deep learning and genetic‑algorithm hyperparameter optimisation. It focuses on three architectures (GRU, BiLSTM, CNN‑BiLSTM) trained on the DeepCRISPR dataset using 4‑channel one‑hot encoding and evaluated mainly with AUPRC under severe class imbalance.
​

**1. Data and Pre‑processing**
load_deepcrispr_data.py – Loads the DeepCRISPR dataset, applies basic cleaning and splits into train/validation/test sets.

step1_data_preprocessing.py – Full preprocessing pipeline: handling N bases, class‑imbalance weights, and saving processed tensors.

step2_encoding.py – Implements the 4‑channel sgRNA/target encoding and logical‑OR integration used as model input.
​

2. Model Building and Training
step3_build_models.py – Defines all neural architectures: GRU, BiLSTM, CNN‑only, CNN‑BiLSTM and CNN‑BiLSTM‑Attention.

step4c_cnn_bilstm_4ch_optimized.py – Final CNN‑BiLSTM training script with GA‑optimised hyperparameters (best model in the thesis).

step4c_bilstm_memory_optimized.py – BiLSTM training with GA‑constrained hidden size and batch size for memory efficiency.

step4c_true_ga_4MODELS.py – Genetic‑algorithm driver: evolves hyperparameters for GRU, BiLSTM and CNN‑BiLSTM using validation AUPRC as fitness.

step6_random_search_test.py – Random‑search baseline over the same architectures and hyperparameter ranges, used to compare with GA.
​

3. Final Evaluation Scripts
Step5-final-test-optionb.py – Loads the best GA models, retrains on train+val, and evaluates on the held‑out DeepCRISPR test set (AUPRC, AUROC, precision, recall, F1). This corresponds to the “test result option B” configuration in the thesis.

step5_final_save_run.py – Utility script to save final model weights, logits, and metrics for later analysis and plotting.

find-model-weights.py – Small helper to locate and load the correct checkpoint files for each architecture.

4. Results are in two csv files. Obtiom b is for test test evaluated genetic algorithm optimized model, and the matrix one is for text set evaluated random serach models.​
​

5. Figures
Key figures used in Chapters 4 and 5 (saved as .jpg):

fig1_ig_heatmap_enhanced.jpg – Integrated Gradients position‑importance heatmaps.

fig4_ism_sensitivity.jpg – In Silico Mutagenesis positional sensitivity curve.

fig6_cnn_filters.jpg – CNN filter analysis: information content and filter strength, including NGG‑PAM motif filter.

fig7_bio_alignment.jpg – Biological alignment scores and position heatmap (seed/PAM vs distal).
and some other images too related to the pre stages of project. ​


Marking-grades-criteria.docx – University marking rubric used to align the work with MSc assessment criteria.
