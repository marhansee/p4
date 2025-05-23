import json
import os
import numpy as np
from utilities.data_validation import convert




def save_results(script_dir, config, results):
    results_path = os.path.join(script_dir, config["results_path"])

    with open(results_path, "w") as f:
        json.dump({"results": results}, f, indent=2, default=convert)

    for result in results:
        if isinstance(result.get("fishing_confidence"), np.floating):
            result["fishing_confidence"] = float(result["fishing_confidence"])

    print(f"Saved prediction results to {results_path}")

    return json.loads(json.dumps({"results": results}, default=convert))


