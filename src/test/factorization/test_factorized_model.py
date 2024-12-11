import copy

import transformers

from neuroflex.factorization.factorized_model import LocalSVDModel, GlobalBaseModel

if __name__ == "__main__":
    model_id = "EleutherAI/gpt-neo-1.3B"
    #model_id = "bert-base-uncased"
    model = transformers.AutoModel.from_pretrained(model_id)
    #print(model)
    #prev_mat = model.encoder.layer[0].attention.self.query.weight.detach().clone()

    blocks = model.config.num_layers
    hidden_size = model.config.hidden_size
    r_svd = 512
    r_gbm_attention = int(r_svd * blocks * 2 * hidden_size / (blocks * hidden_size + hidden_size))
    r_gbm_mlp_up = int(r_svd * blocks * 5 * hidden_size / (blocks * 4 * hidden_size + hidden_size))
    r_gbm_mlp_down = int(r_svd * blocks * 5 * hidden_size / (blocks * hidden_size + 4 * hidden_size))

    print(f"r_svd: {r_svd}")
    print(f"r_gbm_attention: {r_gbm_attention}")
    print(f"r_gbm_mlp_up: {r_gbm_mlp_up}")
    print(f"r_gbm_mlp_down: {r_gbm_mlp_down}")

    svd_attention_bert_target_layers = {"query": {"rank": r_svd},
                                        "key": {"rank": r_svd},
                                        "value": {"rank": r_svd}}
    svd_attention_gpt_target_layers = {"q_proj": {"rank": r_svd},
                                       "k_proj": {"rank": r_svd},
                                       "v_proj": {"rank": r_svd}}
    gbm_attention_bert_target_layers = {"query": {"rank": r_gbm_attention},
                                        "key": {"rank": r_gbm_attention},
                                        "value": {"rank": r_gbm_attention}}
    gbm_attention_gpt_target_layers = {"q_proj": {"rank": r_gbm_attention},
                                       "k_proj": {"rank": r_gbm_attention},
                                       "v_proj": {"rank": r_gbm_attention}}

    svd_mlp_bert_target_layers_0 = {"intermediate": {"rank": r_svd},
                                    "output": {"rank": r_svd}}
    svd_mlp_gpt_target_layers_0 = {"c_fc": {"rank": r_svd},
                                   "c_proj": {"rank": r_svd}}
    gbm_mlp_bert_target_layers_0 = {"intermediate": {"rank": r_gbm_mlp_up},
                                    "output": {"rank": r_gbm_mlp_down}}
    gbm_mlp_gpt_target_layers_0 = {"c_fc": {"rank": r_gbm_mlp_up},
                                   "c_proj": {"rank": r_gbm_mlp_down}}
    gbm_mlp_bert_target_layers_1 = {"intermediate": {"rank": r_gbm_attention},
                                    "output": {"rank": r_gbm_attention}}
    gbm_mlp_gpt_target_layers_1 = {"c_fc": {"rank": r_gbm_attention},
                                   "c_proj": {"rank": r_gbm_attention}}

    print()
    print("SVD FACTORIZATION:")
    #"""
    svd_model_wrapper = LocalSVDModel(
        copy.deepcopy(model),
        target_layers=svd_attention_gpt_target_layers,
        use_names_as_keys=True
    )
    #"""

    print()
    print("GBM FACTORIZATION:")
    gbm_model_wrapper = GlobalBaseModel(
        copy.deepcopy(model),
        target_layers=gbm_attention_gpt_target_layers,
        use_names_as_keys=True,
        average_svd_initialization="svd_of_average_matrix",
        post_init_train=True
    )

    #new_mat = model_wrapper.model.encoder.layer[0].attention.self.query.weight.detach().clone()
