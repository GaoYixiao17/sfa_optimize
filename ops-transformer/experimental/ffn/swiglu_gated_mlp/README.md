# SwigluGatedMlp

## Product Support

| Product | Support |
|:--|:--:|
| Atlas A3 training/inference series | Yes |
| Atlas A2 training/inference series | Yes |

## Function

`SwigluGatedMlp` computes a fused gated MLP:

```text
gate_up = MatMul(x, gate_up_weight)
hidden = SwiGLU(gate_up)
y = MatMul(hidden, down_weight)
```

For an input `x` with shape `[..., hidden_size]`, `gate_up_weight` has shape
`[hidden_size, 2 * intermediate_size]`, `down_weight` has shape
`[intermediate_size, out_size]`, and the output `y` has shape `[..., out_size]`.

## Parameters

| Name | Input/Output/Attr | Description | Data Type | Format |
|:--|:--|:--|:--|:--|
| x | Input | MLP input tensor. The last dimension is `hidden_size`. | BFLOAT16, FLOAT16, FLOAT | ND |
| gate_up_weight | Input | Weight of the first matmul. The second dimension must be even. | BFLOAT16, FLOAT16, FLOAT | ND |
| down_weight | Input | Weight of the second matmul. The first dimension must equal `gate_up_weight.shape[1] / 2`. | BFLOAT16, FLOAT16, FLOAT | ND |
| y | Output | Output tensor. The leading dimensions are the same as `x`, and the last dimension is `down_weight.shape[1]`. | BFLOAT16, FLOAT16, FLOAT | ND |
| cube_math_type | Attr | MatMul compute mode used by the ACLNN wrapper. Valid values are `0` and `1`; default is `1`. | int64 | - |

## Constraints

- `x` rank must be greater than or equal to 2.
- `gate_up_weight` and `down_weight` must be 2D tensors.
- All inputs and output must use the same data type.
- `gate_up_weight.shape[0]` must equal the last dimension of `x`.
- `gate_up_weight.shape[1]` must be divisible by 2.
- `down_weight.shape[0]` must equal `gate_up_weight.shape[1] / 2`.

## Invocation

| Invocation | Example | Description |
|:--|:--|:--|
| ACLNN | [test_aclnn_swiglu_gated_mlp.cpp](./examples/test_aclnn_swiglu_gated_mlp.cpp) | Calls the operator through [aclnnSwigluGatedMlp](./docs/aclnnSwigluGatedMlp.md). |
| Graph mode | - | Builds the operator through [operator IR](./op_graph/swiglu_gated_mlp_proto.h). |
