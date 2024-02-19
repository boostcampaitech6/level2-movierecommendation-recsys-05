MODELS=("Pop" "ItemKNN" "BPR" "NeuMF" "ConvNCF" "DMF" "FISM" "NAIS" "SpectralCF" "GCMC" "NGCF" "LightGCN" "DGCF" "LINE" "MultiVAE" "MultiDAE" "MacridVAE" "CDAE" "ENMF" "NNCF" "RaCT" "RecVAE" "EASE" "SLIMElastic" "SGL" "ADMMSLIM" "NCEPLRec" "SimpleX" "NCL" "Random" "DiffRec" "LDiffRec")
for MODEL in "${MODELS[@]}"; do
  python /data/ephemeral/level2-movierecommendation-recsys-05/RecBole/run_recbole.py --model $MODEL --dataset movierec --config_files recbole/properties/dataset/movierec.yaml
done
