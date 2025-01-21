use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // define closure to get Tensor<f32>
        let get_tensor = |name: &str| -> Tensor<f32> {
            let view = safetensor
                .tensor(name)
                .unwrap_or_else(|_| panic!("Tensor {} not found", name));
            let tensor = view
                .data()
                .chunks(4)
                .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("convert to f32 failed")))
                .collect::<Vec<f32>>();
            Tensor::new(tensor, view.shape())
        };
        // get layer number
        let num_layers = config.num_hidden_layers;
        // define closure to get Vec<Tensor<f32>>
        let get_tensors = |w_name: &str| -> Vec<Tensor<f32>> {
            (0..num_layers)
                .map(|i| get_tensor(&format!("model.layers.{}.{}.weight", i, w_name)))
                .collect()
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_tensors("input_layernorm"),
            wq: get_tensors("self_attn.q_proj"),
            wk: get_tensors("self_attn.k_proj"),
            wv: get_tensors("self_attn.v_proj"),
            wo: get_tensors("self_attn.o_proj"),
            rms_ffn_w: get_tensors("post_attention_layernorm"),
            w_up: get_tensors("mlp.up_proj"),
            w_gate: get_tensors("mlp.gate_proj"),
            w_down: get_tensors("mlp.down_proj"),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
