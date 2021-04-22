import os
import sys
import json
import time
import main


all_divisions = ["INTERIOR PLAINS", "PACIFIC MOUNTAIN SYSTEM", "ROCKY MOUNTAIN SYSTEM", "LAURENTIAN UPLAND",
             "INTERMONTANE PLATEAUS", "APPALACHIAN HIGHLANDS", "ATLANTIC PLAIN", "INTERIOR HIGHLANDS"]
status_file = "hydroconus_status.json"
gauges_path = os.path.join("data", "camels-gages.csv")
db_path = os.path.join("D:\git", "hydro-data", "catchment-data.sqlite3")
seed = 100


def update_status(division, lstm: bool=False, eval: bool=False, robust: bool=False, run_dir: str=None):
    create = False
    if not os.path.join(".", status_file):
        create = True
    if create:
        status = {}
    else:
        with open(os.path.join(".", status_file)) as f:
            status = json.load(f)
    if division not in status.keys():
        status[division] = {}
    if not eval and not robust:
        status[division]["ealstm" if lstm is False else "lstm"] = "Completed"
        status[division]["ealstm-dir" if lstm is False else "lstm-dir"] = str(run_dir)
    elif not robust:
        status[division]["eval-ealstm" if lstm is False else "eval-lstm"] = "Completed"
    else:
        status[division]["robust-ealstm" if lstm is False else "robust-ealstm"] = "Completed"
    with open(os.path.join(".", status_file), 'w') as f:
        json.dump(status, f)


def get_status(divisions: list, run_label):
    if os.path.join(".", status_file):
        with open(os.path.join(".", status_file)) as f:
            status = json.load(f)
        for d, s in status.items():
            if run_label in s.keys():
                if s[run_label] == "Completed":
                    del divisions[divisions.index(d)]
    return divisions


def get_run_dir(division, lstm: bool=False):

    with open(os.path.join(".", status_file)) as f:
        status = json.load(f)
    label = "ealstm-dir" if not lstm else "lstm-dir"
    run_dir = None
    if division in status.keys():
        if label in status[division].keys():
            run_dir = status[division][label]
    return run_dir


if __name__ == "__main__":
    lstm = False
    eval = False
    robust = False
    mode = "train" if not eval else "evaluate" if not robust else "eval_robustness"
    sys.argv.append(mode)

    t0 = time.time()

    run_type = "ealstm" if lstm is False else "lstm"
    run_type = "eval-" + run_type if eval else run_type
    run_type = "robust-" + run_type if robust else run_type
    divisions = get_status(all_divisions, run_type)
    config = main.get_args()
    for d in divisions:
        print(f"Stating run-type: {run_type} for division: {d}")
        config["physio_division"] = d
        if eval or robust:
            config["run_dir"] = get_run_dir(d, lstm)
        config["gauges_path"] = gauges_path
        config["db_path"] = db_path
        config["seed"] = seed
        config["mode"] = mode
        config["no_static"] = True if lstm else False
        if mode is "train":
            main.train(config)
        elif mode is "evaluate":
            main.evaluate(config)
        elif mode is "eval_robustness":
            main.eval_robustness(config)
        update_status(d, lstm, eval, robust, run_dir=config["run_dir"])
        print(f"Completed run-type: {run_type} for division: {d}")
    t1 = time.time()
    print("All divisions completed. Total runtime: {} sec".format(round(t1-t0, 4)))
