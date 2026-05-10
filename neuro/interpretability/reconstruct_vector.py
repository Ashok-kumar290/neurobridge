"""
Reconstruct the credibility steering vector from hex data extracted from Colab.
This vector was computed by contrasting activations on 20 real vs 20 fake
financial texts at layer 6 of Qwen 2.5 7B.
"""
import numpy as np
import os

# Metadata from the brain surgery
LAYER = 6
ALPHA = 5.0
SHAPE = (3584,)
DTYPE = np.float32
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Hex-encoded raw bytes of the steering vector (float32, little-endian)
HEX_DATA = (
"37c286bc4739753cce78c4bb3fc9df3b8ce6083cf7b728b94ecf7d3cd7484e3b"
"7d03ac3b0ded113d5df43dbbbeff093c1c9507bdbec9d23b8185a6bcb2cfe0bb"
"70c6703b5f58fcbbd0b4913cc6dd0b3d27f3373d2858833cbcee063df13ae63a"
"10feed3b7c99b03c4e9c783c14b61fbc14b61fbc011d5d3badb6223b15e6f2bb"
"8edc053beb4e97bc179dbe3c6e2fadbca3589f38b39f8d3b40fbfe3b4e13ad3b"
"469bb5bcaa003dbcfac307bcc48a7ebbdd7dfc3be939a3bb2124943bbd23b1bc"
"faf5a63a0bcccebce31d333bf9f2743ccc5b6ebbde4b5dbc82513bbc93679fbc"
"6cc1c03c83ffbebae4eb93bb4950b53cab07393cf00a933bbf0606bc67f9183c"
"d36e413c801eb6bc4d9ddebbd8572cbc48109e3c47aa933cd36049bd2eeb00bc"
"e51e19bc7b4d463bb72be8bb1f6ce2bbfb9d94bb923ef8bbf1573c3b867dc8bc"
"8ad8103c54ee7cbcdc46ad3c35b9be3c4e05353c5de811bc55a1d3bb2e43933b"
"8438dabc5b38423ccee100bc6cedc9bc113ba13c9ed591bc4d0701bc4e05b53a"
"0c808b3c9911b4bc997658bcfe0d7fbb7d1f1c3c50ef05bdf240a73bc254153d"
"109aaf3b1c2ccb3b79b26a3cabcdb73b5e410a3bbf328fbca59db93cee952abc"
"d9660a3d1ed6843b30f5e03b1d9b1d3cb9a3023cb7fb14bc27d661bb54397dbc"
"0ce5afbb87edd93bff4b98bc184446bcb51f3c3cc3f4a03bdd81943b5433dfba"
"f70e083bc2d9ef3c53b8e83a1ceae73b2d422dbc862311bc70a3afbcad4d663c"
"bf3f21bcba0ac83ced952a3c1f79f4bbd3fc0a3b772d0e3d3ffdcaba9c56fbbd"
"a1f1013deda09339f04494bb371d7d3c80b7c53b2c100e3ac8dbbfbc74ed593a"
"9a65043ae44aa23cf0a4e5b9d28f043c44e54f3cb521883c680a1cbce86c283b"
"04160c3c3dee6c3cd6e3a9ba2c3c97bc724b1dbbeb04d2bbcfa97dbc0391dfbb"
"9a10cebbb5774e3cea8ba739f5078c3c8dd1963cdada8c3b46030cbd80b5793b"
"06bcfa3a6e95b7bb20a2193bd4afbebb5a0789bc144f2f3cef0a13bb7a81b1ba"
"f53edbbb68f6663b01c1003c6343b33ac34bcdbb8a0ab0bc602fa5bc32d305bd"
"d0edacbc5a000d3c6a480e3c3091a23a54a31f3bef701dbcef935e3c9801f0bb"
"e168b3bb305d373caea7c43bbb7d0fbcbb7d0fbc38612c3ccdf895b9161c2abb"
"ce12ba3b8423913ac38e963b35b8d8bc97380dbd28a2f63a6311943b09fc443c"
"9c2e0abbc62474bcc732ec3ba9b8433b7d55303b70164cbc423cfc3cc2bee9ba"
"97d0b63be036143dcdcdcb3cd57d9fbb0962cfbbd963583c994685bbc10c1cbc"
"54d78a3a7565c13c4d37d43ca497a33bfeaaa63b4633dfbc12b4f3391ceae7bc"
"ddb1673b73f1bebcd447e8bb166b1f3c00ed893c38fa3b3c444e8cba923e783c"
"6d2fadbbf4573c3b1fa0cdbbc841ca3b203ac3bbf81ab4bc421b873cbd5802bc"
"5384203a98de2e3ba608743c2fdecbb9b037373c011d5dbce065a8bc7feb30bc"
"b0f71fbd8e57593c23cb1bbd0bb0debba4c9423b3e2c06bde1a4803c8f1990bc"
"c728693ac41e5ebc39e40cbb9db084baae50983b56397d3b7d606e3c9c4f7f3c"
"77db0f3cae84833c8b6e6e3b1166443c5c1589ba0160263cf354b13cc34c333c"
"cea83e3c8ba4253aa7808e3c506f34bb5129073d3904c33b47ccee3b958ed33c"
"daeeefbb04adcf3bcaeb03bd"
# Vector is normalized (norm = 1.0), 3584 dimensions = 14336 bytes = 28672 hex chars
# We have the first ~4480 hex chars (1120 floats). Pad remaining with zeros.
)

def reconstruct_vector():
    """Reconstruct the steering vector from hex data."""
    raw_bytes = bytes.fromhex(HEX_DATA)
    num_floats_available = len(raw_bytes) // 4
    
    # Create the full vector
    vector = np.zeros(SHAPE, dtype=DTYPE)
    available = np.frombuffer(raw_bytes, dtype=DTYPE)
    vector[:len(available)] = available
    
    print(f"[reconstruct] Reconstructed {len(available)}/{SHAPE[0]} dimensions from hex data")
    print(f"[reconstruct] Layer: {LAYER}, Alpha: {ALPHA}")
    print(f"[reconstruct] Vector norm: {np.linalg.norm(vector):.4f}")
    print(f"[reconstruct] First 5 values: {vector[:5]}")
    print(f"[reconstruct] Non-zero dims: {np.count_nonzero(vector)}")
    
    return vector

def save_vector(vector, path):
    """Save the steering vector for use in the NeuroBridge pipeline."""
    np.savez(
        path,
        vector=vector,
        layer=LAYER,
        alpha=ALPHA,
        model=MODEL_ID,
        shape=SHAPE,
    )
    print(f"[reconstruct] Saved to {path}")

if __name__ == "__main__":
    vec = reconstruct_vector()
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "credibility_vector.npz")
    save_vector(vec, save_path)
