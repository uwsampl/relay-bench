
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
TensorValue global16;
TensorValue global18;

ConValue global0(const TensorValue &local_1, const TensorValue &local_2,
                 const TensorValue &local_3, const TensorValue &local_4,
                 const TensorValue &local_5, const TensorValue &local_6,
                 const TensorValue &local_7, const TensorValue &local_8,
                 const TensorValue &local_9, const TensorValue &local_10) {
  auto local_fwd_11 = [=](const TensorValue &local_12,
                          const TensorValue &local_13,
                          const TensorValue &local_14) {
    auto local_x_15 = global16;
    auto local_x_17 = global18;
    auto local_x_19 = [=](const TensorValue &local_20,
                          const TensorValue &local_21) {
      const PackedFunc *pf = runtime::Registry::Get("op-7034913050403622503");
      CHECK(pf);
      TensorValue local_22 =
          TensorValueNode::make(NDArray::Empty({1}, dtype_i32, context));
      (*pf)(local_20->data, local_21->data, local_22->data);
      return local_22;
    };
    auto local_x_23 = local_x_19(local_x_17, local_13);
    auto local_x_24 = [=](const TensorValue &local_25,
                          const TensorValue &local_26) {
      const PackedFunc *pf = runtime::Registry::Get("op-2426713954522055328");
      CHECK(pf);
      TensorValue local_27 =
          TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
      (*pf)(local_25->data, local_26->data, local_27->data);
      return local_27;
    };
    auto local_x_28 = local_x_24(local_x_15, local_x_23);
    auto local_x_29 = [=](const TensorValue &local_30,
                          const TensorValue &local_31,
                          const TensorValue &local_32) {
      const PackedFunc *pf = runtime::Registry::Get("op-6789492519294472756");
      CHECK(pf);
      TensorValue local_33 =
          TensorValueNode::make(NDArray::Empty({1, 204}, dtype_f32, context));
      (*pf)(local_30->data, local_31->data, local_32->data, local_33->data);
      return local_33;
    };
    auto local_x_34 = local_x_29(local_12, local_x_28, local_14);
    auto local_x_35 = [=](const TensorValue &local_36,
                          const TensorValue &local_37,
                          const TensorValue &local_38) {
      const PackedFunc *pf = runtime::Registry::Get("op-448523052254355985");
      CHECK(pf);
      TensorValue local_39 =
          TensorValueNode::make(NDArray::Empty({1, 128}, dtype_f32, context));
      (*pf)(local_36->data, local_37->data, local_38->data, local_39->data);
      return local_39;
    };
    auto local_x_40 = local_x_35(local_x_34, local_5, local_6);
    auto local_x_41 = [=](const TensorValue &local_42,
                          const TensorValue &local_43,
                          const TensorValue &local_44) {
      const PackedFunc *pf = runtime::Registry::Get("op343787827424710963");
      CHECK(pf);
      TensorValue local_45 =
          TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
      (*pf)(local_42->data, local_43->data, local_44->data, local_45->data);
      return local_45;
    };
    auto local_x_46 = local_x_41(local_x_34, local_7, local_8);
    auto local_x_47 = [=](const TensorValue &local_48,
                          const TensorValue &local_49) {
      const PackedFunc *pf = runtime::Registry::Get("op-2750334794306406931");
      CHECK(pf);
      TensorValue local_50 =
          TensorValueNode::make(NDArray::Empty({1, 186}, dtype_f32, context));
      (*pf)(local_48->data, local_49->data, local_50->data);
      return local_50;
    };
    auto local_x_51 = local_x_47(local_x_40, local_x_46);
    auto local_x_52 = [=](const TensorValue &local_53,
                          const TensorValue &local_54,
                          const TensorValue &local_55) {
      const PackedFunc *pf = runtime::Registry::Get("op343795278195122140");
      CHECK(pf);
      TensorValue local_56 =
          TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
      (*pf)(local_53->data, local_54->data, local_55->data, local_56->data);
      return local_56;
    };
    auto local_x_57 = local_x_52(local_x_51, local_9, local_10);
    auto local_x_58 = [=](const TensorValue &local_59) {
      const PackedFunc *pf = runtime::Registry::Get("op4391048907219141147");
      CHECK(pf);
      TensorValue local_60 =
          TensorValueNode::make(NDArray::Empty({1, 58}, dtype_f32, context));
      (*pf)(local_59->data, local_60->data);
      return local_60;
    };
    auto local_x_61 = local_x_58(local_x_57);
    auto local_x_62 = [=](const TensorValue &local_63) {
      const PackedFunc *pf = runtime::Registry::Get("op-3456003299060308512");
      CHECK(pf);
      TensorValue local_64 =
          TensorValueNode::make(NDArray::Empty({}, dtype_i32, context));
      (*pf)(local_63->data, local_64->data);
      return local_64;
    };
    auto local_x_65 = local_x_62(local_x_61);
    auto local_x_66 = [=](const TensorValue &local_67) {
      const PackedFunc *pf = runtime::Registry::Get("op7602301861619355963");
      CHECK(pf);
      TensorValue local_68 =
          TensorValueNode::make(NDArray::Empty({}, dtype_u1, context));
      (*pf)(local_67->data, local_68->data);
      return local_68;
    };
    auto local_x_69 = local_x_66(local_x_65);
    auto local_x_70 =
        TupleValueNode::make({local_x_40, local_x_65, local_x_69});
    return local_x_70;
  };
  auto local_x_71 = [=](const TensorValue &local_72) {
    const PackedFunc *pf = runtime::Registry::Get("op-2964815820486397095");
    CHECK(pf);
    TensorValue local_73 =
        TensorValueNode::make(NDArray::Empty({}, dtype_u1, context));
    (*pf)(local_72->data, local_73->data);
    return local_73;
  };
  auto local_x_74 = local_x_71(local_1);
  ConValue local_90;

  if (NDToBool(local_x_74->data)) {
    auto local_x_76 = TagToCV(0, {});

    local_90 = local_x_76;
  } else {
    auto local_x_77 = local_fwd_11(local_2, local_3, local_4);
    auto local_x_78 = Downcast<TensorValue>(local_x_77->fields[2]);
    ConValue local_89;

    if (NDToBool(local_x_78->data)) {
      auto local_x_80 = TagToCV(0, {});

      local_89 = local_x_80;
    } else {
      auto local_x_81 = Downcast<TensorValue>(local_x_77->fields[1]);
      auto local_x_82 = [=](const TensorValue &local_83) {
        const PackedFunc *pf = runtime::Registry::Get("op-2964911815668570852");
        CHECK(pf);
        TensorValue local_84 =
            TensorValueNode::make(NDArray::Empty({}, dtype_i32, context));
        (*pf)(local_83->data, local_84->data);
        return local_84;
      };
      auto local_x_85 = local_x_82(local_1);
      auto local_x_86 = Downcast<TensorValue>(local_x_77->fields[0]);
      auto local_x_87 =
          global0(local_x_85, local_2, local_x_81, local_x_86, local_5, local_6,
                  local_7, local_8, local_9, local_10);
      auto local_x_88 = TagToCV(1, {local_x_81, local_x_87});

      local_89 = local_x_88;
    }
    auto local_x_79 = local_89;

    local_90 = local_x_79;
  }
  auto local_x_75 = local_90;
  return local_x_75;
}

TVM_REGISTER_API("relay.aot.default.2")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      global16 = args[0];
      global18 = args[1];

      *ret = global0(args[2], args[3], args[4], args[5], args[6], args[7],
                     args[8], args[9], args[10], args[11]);
    });
