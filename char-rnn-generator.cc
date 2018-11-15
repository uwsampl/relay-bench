
#include <iostream>
#include <tvm/api_registry.h>
#include <tvm/relay/interpreter.h>
#include <tvm/tvm.h>

using namespace tvm;
using namespace runtime;
using namespace relay;

static DLDataType dtype_f32 =
    DLDataType{.code = DLDataTypeCode::kDLFloat, .bits = 32, .lanes = 1};
static DLDataType dtype_u32 =
    DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 32, .lanes = 1};
static DLDataType dtype_u1 =
    DLDataType{.code = DLDataTypeCode::kDLUInt, .bits = 1, .lanes = 1};
static DLDataType dtype_i32 =
    DLDataType{.code = DLDataTypeCode::kDLInt, .bits = 32, .lanes = 1};
static DLContext context =
    DLContext{.device_type = DLDeviceType::kDLCPU, .device_id = 0};
bool NDToBool(const NDArray &nd) {
  DLContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  NDArray cpu_array = nd.CopyTo(cpu_ctx);
  CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
  return reinterpret_cast<uint8_t *>(cpu_array->data)[0];
}
NDArray ValueToND(const Value &v) {
  const TensorValueNode *tv = v.as<TensorValueNode>();
  CHECK(tv);
  return tv->data;
}
ConValue TagToCV(size_t tag, const tvm::Array<Value> &fields) {
  NodePtr<ConValueNode> n = make_node<ConValueNode>();
  NodePtr<ConstructorNode> con = make_node<ConstructorNode>();
  con->tag = tag;
  n->con = Constructor(con);
  n->fields = fields;
  return ConValue(n);
}
TensorValue global11;
TensorValue global13;

ConValue global0(const TensorValue &param0, const TensorValue &param1,
                 const TensorValue &param2, const TensorValue &param3,
                 const TensorValue &param4, const TensorValue &param5,
                 const TensorValue &param6, const TensorValue &param7,
                 const TensorValue &param8, const TensorValue &param9) {
  auto local_x_1 = [=](const TensorValue &param0, const TensorValue &param1,
                       const TensorValue &param2) {
    const PackedFunc *pf = runtime::Registry::Get("op-8242035636470084578");
    CHECK(pf);
    TensorValue local_2 =
        TensorValueNode::make(NDArray::Empty({1, 204}, dtype_f32, context));
    (*pf)(param0->data, param1->data, param2->data, local_2->data);
    return local_2;
  };
  auto local_x_3 = [=](const TensorValue &param0, const TensorValue &param1,
                       const TensorValue &param2) {
    const PackedFunc *pf = runtime::Registry::Get("op2538537375965756304");
    CHECK(pf);
    TensorValue local_4 =
        TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
    (*pf)(param0->data, param1->data, param2->data, local_4->data);
    return local_4;
  };
  auto local_x_5 = [=](const TensorValue &param0, const TensorValue &param1,
                       const TensorValue &param2) {
    const PackedFunc *pf = runtime::Registry::Get("op2590341137946825400");
    CHECK(pf);
    TensorValue local_6 =
        TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
    (*pf)(param0->data, param1->data, param2->data, local_6->data);
    return local_6;
  };
  auto local_x_7 = [=](const TensorValue &param0) {
    const PackedFunc *pf = runtime::Registry::Get("op3979369115425545227");
    CHECK(pf);
    TensorValue local_8 =
        TensorValueNode::make(NDArray::Empty({}, dtype_u1, context));
    (*pf)(param0->data, local_8->data);
    return local_8;
  };
  auto local_fwd_9 = [=](const TensorValue &param0, const TensorValue &param1,
                         const TensorValue &param2) {
    auto local_x_10 = global11;
    auto local_x_12 = global13;
    auto local_x_14 = [=](const TensorValue &param0,
                          const TensorValue &param1) {
      const PackedFunc *pf = runtime::Registry::Get("op-4746875704108702661");
      CHECK(pf);
      TensorValue local_15 =
          TensorValueNode::make(NDArray::Empty({1}, dtype_i32, context));
      (*pf)(param0->data, param1->data, local_15->data);
      return local_15;
    };
    auto local_x_16 = local_x_14(local_x_12, param1);
    auto local_x_17 = [=](const TensorValue &param0,
                          const TensorValue &param1) {
      const PackedFunc *pf = runtime::Registry::Get("op9151187201150592787");
      CHECK(pf);
      TensorValue local_18 =
          TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
      (*pf)(param0->data, param1->data, local_18->data);
      return local_18;
    };
    auto local_x_19 = local_x_17(local_x_10, local_x_16);
    auto local_x_20 = local_x_1(param0, local_x_19, param2);
    auto local_x_21 = [=](const TensorValue &param0, const TensorValue &param1,
                          const TensorValue &param2) {
      const PackedFunc *pf = runtime::Registry::Get("op2590029761362752295");
      CHECK(pf);
      TensorValue local_22 =
          TensorValueNode::make(NDArray::Empty({1, 128}, dtype_f32, context));
      (*pf)(param0->data, param1->data, param2->data, local_22->data);
      return local_22;
    };
    auto local_x_23 = local_x_21(local_x_20, param4, param5);
    auto local_x_24 = local_x_3(local_x_20, param6, param7);
    auto local_x_25 = [=](const TensorValue &param0,
                          const TensorValue &param1) {
      const PackedFunc *pf = runtime::Registry::Get("op8610143474476802891");
      CHECK(pf);
      TensorValue local_26 =
          TensorValueNode::make(NDArray::Empty({1, 186}, dtype_f32, context));
      (*pf)(param0->data, param1->data, local_26->data);
      return local_26;
    };
    auto local_x_27 = local_x_25(local_x_23, local_x_24);
    auto local_x_28 = local_x_5(local_x_27, param8, param9);
    auto local_x_29 = [=](const TensorValue &param0) {
      const PackedFunc *pf = runtime::Registry::Get("op4327758349284940724");
      CHECK(pf);
      TensorValue local_30 =
          TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
      (*pf)(param0->data, local_30->data);
      return local_30;
    };
    auto local_x_31 = local_x_29(local_x_28);
    auto local_x_32 = [=](const TensorValue &param0) {
      const PackedFunc *pf = runtime::Registry::Get("op-3194121857226101249");
      CHECK(pf);
      TensorValue local_33 =
          TensorValueNode::make(NDArray::Empty({}, dtype_i32, context));
      (*pf)(param0->data, local_33->data);
      return local_33;
    };
    auto local_x_34 = local_x_32(local_x_31);
    auto local_x_35 = local_x_7(local_x_34);
    auto local_x_36 =
        TupleValueNode::make({local_x_31, local_x_23, local_x_34, local_x_35});
    return local_x_36;
  };
  auto local_x_37 = local_fwd_9(param1, param2, param3);
  auto local_x_38 = Downcast<TensorValue>(local_x_37->fields[2]);
  auto local_x_39 = [=](const TensorValue &param0) {
    const PackedFunc *pf = runtime::Registry::Get("op-2717888097881546574");
    CHECK(pf);
    TensorValue local_40 =
        TensorValueNode::make(NDArray::Empty({}, dtype_i32, context));
    (*pf)(param0->data, local_40->data);
    return local_40;
  };
  auto local_x_41 = local_x_39(param0);
  auto local_x_42 = Downcast<TensorValue>(local_x_37->fields[2]);
  auto local_x_43 = Downcast<TensorValue>(local_x_37->fields[1]);
  auto local_x_44 = TagToCV(0, {});
  auto local_x_45 = [=](const TensorValue &param0) {
    const PackedFunc *pf = runtime::Registry::Get("op-2717858108703975279");
    CHECK(pf);
    TensorValue local_46 =
        TensorValueNode::make(NDArray::Empty({}, dtype_u1, context));
    (*pf)(param0->data, local_46->data);
    return local_46;
  };
  auto local_x_47 = local_x_45(param0);
  ConValue local_55;

  if (NDToBool(local_x_47->data)) {
    auto local_x_49 = TagToCV(0, {});

    local_55 = local_x_49;
  } else {
    auto local_x_50 = Downcast<TensorValue>(local_x_37->fields[3]);
    ConValue local_54;

    if (NDToBool(local_x_50->data)) {

      local_54 = local_x_44;
    } else {
      auto local_x_52 = global0(local_x_41, param0, local_x_42, local_x_43,
                                param4, param5, param6, param7, param8, param9);
      auto local_x_53 = TagToCV(1, {local_x_38, local_x_52});

      local_54 = local_x_53;
    }
    auto local_x_51 = local_54;

    local_55 = local_x_51;
  }
  auto local_x_48 = local_55;
  return local_x_48;
}

TVM_REGISTER_API("relay.aot.default.2")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      global11 = args[0];
      global13 = args[1];

      *ret = global0(args[2], args[3], args[4], args[5], args[6], args[7],
                     args[8], args[9], args[10], args[11]);
    });
