from pybads import BADS
import numpy as np
import json
import sys

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.float64):
            return float(obj)

        return json.JSONEncoder.default(self, obj)

def jsonify(obj):
    try:
        return json.dumps(obj, cls=NumpyEncoder)
    except Exception as e:
        logging.exception("Error converting json, falling back on string")
        return str(obj)

def target(x):
    print("REQUEST_EVALUATION", x.tolist())
    return float(input())

conf = json.loads(sys.argv[1])
bads = BADS(target, conf["x0"], conf["lower_bounds"], conf["upper_bounds"], conf["plausible_lower_bounds"], conf["plausible_upper_bounds"])
result = bads.optimize()
del result['fun']
print("FINAL_RESULT", jsonify(result))