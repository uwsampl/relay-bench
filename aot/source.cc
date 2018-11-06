
    #include <tvm/tvm.h>
    #include <tvm/api_registry.h>

    using namespace tvm;

    
    TVM_REGISTER_API("aaa")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
        PackedFunc *pf = reinterpret_cast<PackedFunc*>(140324078892016);
        CHECK(pf);
        (*pf)(args[0], args[1], args[2]);
    });
    
    