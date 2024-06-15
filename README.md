# Flash Attention with Top 1 Probability

There is no point for Flash Attention to return the full probability -- the whole point of Flash Attention is not storing the full probability to improve IO efficiency.

However, in many applications, for example, the analysis of retrieval head, we do not need the full probability distribution, but may need topK probability, particularly the top 1, because we want to understand which exact token the head is looking at.

In this repo, we implement a triton kernel that supports the topK (currently top 1) probability of each attention head, and use it to visualize the effect of retrieval heads.

Currently kernel at `triton_flash_with_p_bf16.py` and tested at `triton_flash_attn_with_prob.ipynb`

TODO: 
* Tuning parameters, compare efficiency
* Integrate to transformer inference
* Support topK

