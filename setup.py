import random
import argparse
from utils import *

set_seed(123)


# filter out plans with runtime < 100
def filter_plans():
    def filter(original_plans):
        filtered_plans = []
        for plan in original_plans:
            run_time = plan[0][0][0]["Plan"]["Actual Total Time"]
            if run_time > 100:
                filtered_plans.append(plan[0][0][0])
        return filtered_plans

    for workload in workloads:
        with open(
            os.path.join(ROOT_DIR, "data/workload1/{}".format(workload)) + ".json", "r"
        ) as f:
            content = f.readlines()
        for line in content:
            plans = json.loads(line)
        filted_plans = filter(plans)
        assert len(filted_plans) >= 10000
        filted_plans = random.sample(filted_plans, 10000)
        print("workload: ", workload, "filted plans: ", len(filted_plans))
        with open(
            os.path.join(ROOT_DIR, "data/workload1/{}".format(workload))
            + "_filted.json",
            "w",
        ) as f:
            json.dump(filted_plans, f)
    # test case
    with open("data/workload1/walmart_filted.json", "r") as f:
        content = f.readlines()
    for line in content:
        plans = json.loads(line)
    print(len(plans))


def get_statistic():
    plans = read_workload_runs(
        ROOT_DIR + "data/workload1", db_names=workloads, verbose=True
    )

    # get statistics
    db_plans = {}
    db_max_node_len = {}
    runtimes = []
    cards = []
    costs = []

    # get all node types, including subplan, implement by dfs
    node_types = set()
    for plan in plans:
        plan_db_id = plan["database_id"]
        if plan_db_id not in db_plans:
            db_plans[plan_db_id] = []
            db_max_node_len[plan_db_id] = 0
        db_plans[plan_db_id].append(plan)

        plan = plan["Plan"]
        runtimes.append(plan["Actual Total Time"])
        costs.append(plan["Total Cost"])
        cards.append(plan["Plan Rows"])
        node_types.add(plan["Node Type"])
        stack = [plan]
        node_len = 1
        while len(stack) > 0:
            node = stack.pop()
            if "Plans" in node:
                for child in node["Plans"]:
                    node_len += 1
                    node_types.add(child["Node Type"])
                    stack.append(child)
                    runtimes.append(child["Actual Total Time"])
                    costs.append(child["Total Cost"])
                    cards.append(child["Plan Rows"])
        db_max_node_len[plan_db_id] = max(db_max_node_len[plan_db_id], node_len)

    runtimes, cards, costs = np.array(runtimes), np.array(cards), np.array(costs)

    # write to a json file
    node_types = list(node_types)
    statistics = {
        "Actual Total Time": {
            "type": str(FeatureType.numeric),
            "max": float(np.max(runtimes)),
            "min": float(np.min(runtimes)),
            "center": float(np.median(runtimes)),
            "scale": float(np.quantile(runtimes, 0.75))
            - float(np.quantile(runtimes, 0.25)),
        },
        "Plan Rows": {
            "type": str(FeatureType.numeric),
            "max": float(np.max(cards)),
            "min": float(np.min(cards)),
            "center": float(np.median(cards)),
            "scale": float(np.quantile(cards, 0.75)) - float(np.quantile(cards, 0.25)),
        },
        "Total Cost": {
            "type": str(FeatureType.numeric),
            "max": float(np.max(costs)),
            "min": float(np.min(costs)),
            "center": float(np.median(costs)),
            "scale": float(np.quantile(costs, 0.75)) - float(np.quantile(costs, 0.25)),
        },
        "node_types": {
            "type": str(FeatureType.categorical),
            "value_dict": {node_type: i for i, node_type in enumerate(node_types)},
        },
    }

    with open(ROOT_DIR + "data/workload1/statistics.json", "w") as f:
        json.dump(statistics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter_plans",
        action="store_true",
        help="filter plans (we filter out plans with runtime < 100))",
    )
    parser.add_argument(
        "--get_statistic", action="store_true", help="get filter plans statistic"
    )

    args = parser.parse_args()

    # transform args to configs
    configs = vars(args)

    # filter plans
    if configs["filter_plans"]:
        filter_plans()
    if configs["get_statistic"]:
        get_statistic()
