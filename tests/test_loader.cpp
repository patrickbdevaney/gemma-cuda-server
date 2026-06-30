// test_loader.cpp — Phase 3.1 loader gate: open the real Gemma-4 NVFP4 checkpoint, validate structure.
#include "safetensors.h"
#include <cstdio>
#include <string>

static void show(const st::SafeTensors& m, const std::string& n) {
    if (!m.has(n)) { printf("  MISSING  %s\n", n.c_str()); return; }
    const auto& t = m.get(n);
    printf("  %-8s [", t.dtype.c_str());
    for (size_t i = 0; i < t.shape.size(); ++i) printf("%s%ld", i ? "," : "", t.shape[i]);
    printf("]  %.2f MB  %s\n", t.nbytes / 1e6, n.c_str());
}

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1]
        : std::string(getenv("HOME")) + "/models/gemma-4-26B-A4B-it-NVFP4/model.safetensors";
    st::SafeTensors m(path);
    printf("opened %s\n  total tensors: %zu (expect 47648)\n", path.c_str(), m.count());

    printf("\n[embedding / lm_head / final norm]\n");
    show(m, "model.language_model.embed_tokens.weight");
    show(m, "model.language_model.norm.weight");

    printf("\n[layer 0 = SLIDING attn: q/k/v_proj NVFP4 packed + scales]\n");
    for (auto s : {"q_proj","k_proj","v_proj","o_proj"}) {
        std::string b = std::string("model.language_model.layers.0.self_attn.") + s;
        show(m, b + ".weight_packed"); show(m, b + ".weight_scale"); show(m, b + ".weight_global_scale");
    }
    printf("  q/k/v norms:\n");
    for (auto s : {"q_norm","k_norm","v_norm"}) show(m, std::string("model.language_model.layers.0.self_attn.") + s + ".weight");

    printf("\n[layer 5 = FULL attn: attention_k_eq_v => v_proj should be ABSENT]\n");
    for (auto s : {"q_proj","k_proj","v_proj","o_proj"}) {
        std::string b = std::string("model.language_model.layers.5.self_attn.") + s;
        show(m, b + ".weight_packed");
    }

    printf("\n[layer 0 MoE: router (bf16, NOT quantized) + expert 0 + dense mlp]\n");
    show(m, "model.language_model.layers.0.router.proj.weight");
    for (auto s : {"gate_proj","up_proj","down_proj"}) {
        std::string b = std::string("model.language_model.layers.0.experts.0.") + s;
        show(m, b + ".weight_packed"); show(m, b + ".weight_scale");
    }
    printf("  [decoder norms]\n");
    for (auto s : {"input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"})
        show(m, std::string("model.language_model.layers.0.") + s + ".weight");

    // sanity counts: how many layers have v_proj packed (sliding) vs not (full)?
    int sliding = 0, full = 0;
    for (int L = 0; L < 30; ++L) {
        std::string vp = "model.language_model.layers." + std::to_string(L) + ".self_attn.v_proj.weight_packed";
        (m.has(vp) ? sliding : full)++;
    }
    printf("\nlayers with v_proj (sliding-style): %d, without (k_eq_v full): %d  (expect 25 / 5)\n", sliding, full);
    return 0;
}
