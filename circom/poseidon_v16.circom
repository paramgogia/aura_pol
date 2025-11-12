// circom/poseidon_v16.circom
pragma circom 2.0.0;

include "circomlib/poseidon.circom";

template Poseidon16() {
    // private inputs: 16 elements (downsampled grayscale 0..255),
    // plus metadata are public inputs (we'll include them as public inputs to hash too)
    signal input v[16];             // private inputs (0..255)
    signal input model_version;     // public input
    signal input p_scaled;          // public input (uint)
    signal input nonce;             // public input (uint)
    signal input subject_uint;      // public input (uint)

    // compute poseidon on concatenation [v0..v15, model_version, p_scaled, nonce, subject_uint]
    signal inputArrayLen = 16 + 4;

    // allocate array to feed poseidon
    var total = 20; // 16 + 4
    signal inputValues[20];
    for (var i = 0; i < 16; i++) {
        inputValues[i] <== v[i];
    }
    inputValues[16] <== model_version;
    inputValues[17] <== p_scaled;
    inputValues[18] <== nonce;
    inputValues[19] <== subject_uint;

    component pose = Poseidon(total);
    for (var i = 0; i < total; i++) {
        pose.inputs[i] <== inputValues[i];
    }

    signal output hashOut;
    hashOut <== pose.out;
}

component main = Poseidon16();
