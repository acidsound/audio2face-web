import onnx
from onnx import numpy_helper
import collections, sys

m = onnx.load("/home/opc/dlp3d/weights/unitalker_v0.4.0_base.onnx")
g = m.graph
print("opset:", [o.version for o in m.opset_import])
print("ir_version:", m.ir_version)
print("inputs:")
for i in g.input:
    t = i.type.tensor_type
    print("  ", i.name, [d.dim_value for d in t.shape.dim], t.elem_type)
print("outputs:")
for o in g.output:
    t = o.type.tensor_type
    print("  ", o.name, [d.dim_value for d in t.shape.dim], t.elem_type)
# operator histogram
cnt = collections.Counter(n.op_type for n in g.node)
print("node count:", len(g.node))
print("operator histogram (top 30):")
for op, c in cnt.most_common(30):
    print(f"  {op}: {c}")
# initializer dtypes
dt = collections.Counter()
for ini in g.initializer:
    dt[ini.data_type] += 1
print("initializer dtype counts (1=FLOAT,7=FLOAT16,11=INT32,9=INT8):", dict(dt))
# does it already contain FLOAT16? external data?
print("external_data initializers:", sum(1 for ini in g.initializer if ini.HasField('external_data') and ini.external_data))
print("total params (FLOAT):", sum(ini.dims and numpy_helper.to_array(ini).size for ini in g.initializer if ini.data_type==1))
