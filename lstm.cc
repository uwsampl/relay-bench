
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

ConValue global0(const ConValue &local_1, const TensorValue &local_2,
                 const TensorValue &local_3, const TensorValue &local_4,
                 const TensorValue &local_5, const TensorValue &local_6,
                 const TensorValue &local_7, const TensorValue &local_8,
                 const TensorValue &local_9, const TensorValue &local_10,
                 const TensorValue &local_11) {
  auto local_fwd_cell_12 = [=](const TensorValue &local_13,
                               const TensorValue &local_14,
                               const TensorValue &local_15) {
    auto local_x_16 = [=](const TensorValue &local_17,
                          const TensorValue &local_18) {
      const PackedFunc *pf = runtime::Registry::Get("op-8269736952946545582");
      CHECK(pf);
      TensorValue local_19 =
          TensorValueNode::make(NDArray::Empty({1, 48}, dtype_f32, context));
      (*pf)(local_17->data, local_18->data, local_19->data);
      return local_19;
    };
    auto local_x_20 = local_x_16(local_13, local_14);
    auto local_x_21 = [=](const TensorValue &local_22,
                          const TensorValue &local_23,
                          const TensorValue &local_24) {
      const PackedFunc *pf = runtime::Registry::Get("op-6806743077622327199");
      CHECK(pf);
      TensorValue local_25 =
          TensorValueNode::make(NDArray::Empty({1, 32}, dtype_f32, context));
      (*pf)(local_22->data, local_23->data, local_24->data, local_25->data);
      return local_25;
    };
    auto local_x_26 = local_x_21(local_x_20, local_8, local_9);
    auto local_x_27 = [=](const TensorValue &local_28,
                          const TensorValue &local_29,
                          const TensorValue &local_30,
                          const TensorValue &local_31) {
      const PackedFunc *pf = runtime::Registry::Get("op2749465684336401322");
      CHECK(pf);
      TensorValue local_32 =
          TensorValueNode::make(NDArray::Empty({1, 32}, dtype_f32, context));
      (*pf)(local_28->data, local_29->data, local_30->data, local_31->data,
            local_32->data);
      return local_32;
    };
    auto local_x_33 = local_x_27(local_x_20, local_6, local_7, local_x_26);
    auto local_x_34 = [=](const TensorValue &local_35,
                          const TensorValue &local_36,
                          const TensorValue &local_37,
                          const TensorValue &local_38,
                          const TensorValue &local_39) {
      const PackedFunc *pf = runtime::Registry::Get("op3954849355008429229");
      CHECK(pf);
      TensorValue local_40 =
          TensorValueNode::make(NDArray::Empty({1, 32}, dtype_f32, context));
      (*pf)(local_35->data, local_36->data, local_37->data, local_38->data,
            local_39->data, local_40->data);
      return local_40;
    };
    auto local_x_41 =
        local_x_34(local_15, local_x_20, local_4, local_5, local_x_33);
    auto local_x_42 = [=](const TensorValue &local_43,
                          const TensorValue &local_44,
                          const TensorValue &local_45,
                          const TensorValue &local_46) {
      const PackedFunc *pf = runtime::Registry::Get("op5393002233121881107");
      CHECK(pf);
      TensorValue local_47 =
          TensorValueNode::make(NDArray::Empty({1, 32}, dtype_f32, context));
      (*pf)(local_43->data, local_44->data, local_45->data, local_46->data,
            local_47->data);
      return local_47;
    };
    auto local_x_48 = local_x_42(local_x_41, local_x_20, local_10, local_11);
    auto local_x_49 = TupleValueNode::make({local_x_48, local_x_41});
    return local_x_49;
  };
  TensorValue local_51;
  ConValue local_52;
  ConValue local_53 = local_1;
  ConValue local_54;
  {

    CHECK(local_53->con->tag != -1);
    if (local_53->con->tag == 0) {
      goto label_58;
    } else {
      goto label_57;
    }
  }
  {
  label_58:
    auto local_x_56 = TagToCV(0, {});

    local_54 = local_x_56;
    goto label_55;
  }
label_57 : {

  CHECK(local_53->con->tag != -1);
  if (local_53->con->tag == 1) {
    TensorValue local_66 = Downcast<TensorValue>(local_53->fields[0]);
    ConValue local_67 = Downcast<ConValue>(local_53->fields[1]);

    local_51 = local_66;
  label_68:

    local_52 = local_67;
  label_69:
    goto label_65;
  } else {
    goto label_64;
  }
}
  {
  label_65:
    auto local_x_59 = local_fwd_cell_12(local_51, local_2, local_3);
    auto local_x_60 = Downcast<TensorValue>(local_x_59->fields[0]);
    auto local_x_61 = Downcast<TensorValue>(local_x_59->fields[1]);
    auto local_x_62 =
        global0(local_52, local_x_60, local_x_61, local_4, local_5, local_6,
                local_7, local_8, local_9, local_10, local_11);
    auto local_x_63 = TagToCV(1, {local_x_60, local_x_62});

    local_54 = local_x_63;
    goto label_55;
  }
label_64:
  CHECK(false) << "does not match any";
label_55:;
  auto local_x_50 = local_54;
  return local_x_50;
}

TVM_REGISTER_API("relay.aot.default.1")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      *ret = global0(args[0], args[1], args[2], args[3], args[4], args[5],
                     args[6], args[7], args[8], args[9], args[10]);
    });
