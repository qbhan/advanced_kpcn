python test.py \
  --mode kpcn \
  --diffuse_model trained_model/test_kpcn_relL2_1/diff_e6.pt\
  --specular_model trained_model/test_kpcn_relL2_1/spec_e6.pt \
  --data_dir '/root/kpcn_data/kpcn_data/data' \
  --save_dir 'test/check_cython_test_kpcn_relL2_1'