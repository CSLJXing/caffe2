#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(Split, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(Concat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(Split)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')")
    .Arg("axis", "Which axis to split on")
    .Arg("split", "length of each output")
    .Arg("order", "Either NHWC or NCWH, will split on C axis")
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          ArgumentHelper helper(def);

          CAFFE_ENFORCE(
            helper.HasArgument("axis") ^ helper.HasArgument("order"),
            "You should either specify the dim to split, or the order "
            "in the case of 4-D images."
          );

          int axis;
          if (helper.HasArgument("axis"))
          {
            axis = helper.GetSingleArgument<int>("axis", -1);
          } else
          {
            axis = GetDimFromOrderString(
              helper.GetSingleArgument<string>("order", "")
            );
          }
          CAFFE_ENFORCE_GE(axis, 0);

          const int input_channels = in[0].dims(axis);
          vector<int> out_shape;
          for (int i = 0; i < in[0].dims_size(); ++i)
          {
             out_shape.push_back(in[0].dims(i));
          }
          vector<int> split = helper.GetRepeatedArgument<int>("split");

          if (in.size() == 2)
          {
            vector<TensorShape> out;
            for (int i = 0; i < def.output_size(); i++)
            {
              TensorShape ts;
              ts.set_unknown_shape(true);
              out.push_back(ts);
            }
            return out;
          } else if (split.size() == 0)
          {
            CAFFE_ENFORCE_GE(input_channels % split.size(), 0);
            out_shape[axis] = input_channels / split.size();
            vector<TensorShape> out;
            for (int i = 0; i < def.output_size(); ++i)
            {
              out.push_back(
                CreateTensorShape(out_shape, TensorProto::FLOAT)
              );
            }
            return out;
          } else
          {
            CAFFE_ENFORCE_GE(split.size(), def.output_size());
            vector<TensorShape> out;
            for (int i = 0; i < split.size(); ++i)
            {
              out_shape[axis] = split[i];
              out.push_back(
                CreateTensorShape(out_shape, TensorProto::FLOAT)
              );
            }
            return out;
          }
        })
    .SetDoc(R"DOC(Split a tensor into a list of tensors, along the specified
    'axis'. The lengths of the split can be specified using argument 'axis' or
    optional second input blob to the operator. Otherwise, the tensor is split
    to equal sized parts.
    )DOC");
OPERATOR_SCHEMA(Concat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .Arg("axis", "Which axis to concat on")
    .Arg("order", "Either NHWC or HCWH, will concat on C axis")
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          ArgumentHelper helper(def);

          CAFFE_ENFORCE(
            helper.HasArgument("axis") ^ helper.HasArgument("order"),
            "You should either specify the dim to split, or the order "
            "in the case of 4-D images."
          );

          int axis;
          if (helper.HasArgument("axis"))
          {
            axis = helper.GetSingleArgument<int>("axis", -1);
          } else
          {
            axis = GetDimFromOrderString(
              helper.GetSingleArgument<string>("order", "")
            );
          }
          CAFFE_ENFORCE_GE(axis, 0);

          int output_channels = 0;
          for (int i = 0; i < in.size(); i++)
          {
            output_channels += in[i].dims(axis);
          }

          vector<int> out_shape;
          for (int i = 0; i < in[0].dims_size(); ++i)
          {
            out_shape.push_back(in[0].dims(i));
          }
          out_shape[axis] = output_channels;

          vector<TensorShape> out(2);
          out[0] = CreateTensorShape(out_shape, TensorProto::FLOAT);
          out[1] = CreateTensorShape(vector<int>(1, out_shape.size()) , TensorProto::FLOAT);

          return out;
        })
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Output(0, "concat_result", "Concatenated tensor")
    .Output(1, "split_info", "The dimensions of the inputs.");

// Backward compatibility names.
REGISTER_CPU_OPERATOR(DepthSplit, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(DepthConcat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(DepthSplit)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .SetDoc("Backward compatible operator name for Split.");
OPERATOR_SCHEMA(DepthConcat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .SetDoc("Backward compatible operator name for Concat.");

class GetSplitGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    vector<string> output_grads;
    for (int i = 0; i < def_.output_size(); ++i) {
      if (!GradOut(i).IsEmpty()) {
        output_grads.push_back(GO(i));
      }
    }
    if (output_grads.empty()) {
      return {};
    }
    return SingleGradientDef(
        "Concat", "", output_grads,
        vector<string>{GI(0), "_" + GI(0) + "_dims"});
  }
};
REGISTER_GRADIENT(Split, GetSplitGradient);
REGISTER_GRADIENT(DepthSplit, GetSplitGradient);

class GetConcatGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (GradOut(0).IsEmpty()) {
      return {};
    }
    vector<string> grads;
    for (int i = 0; i < def_.input_size(); ++i) {
      grads.push_back(GI(i));
    }
    return SingleGradientDef(
        "Split", "", vector<string>{GO(0), O(1)}, grads);
  }
};
REGISTER_GRADIENT(Concat, GetConcatGradient);
REGISTER_GRADIENT(DepthConcat, GetConcatGradient);
}  // namespace
}  // namespace caffe2
