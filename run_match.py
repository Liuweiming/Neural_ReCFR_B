import re
import numpy as np
import argparse
import os
from multiprocessing import Process
from scipy.stats import t

parser = argparse.ArgumentParser(description="run matchs.")
parser.add_argument("-n", "--num_procs", dest="num_procs", type=int,
                    default=10, help="num threads.")
parser.add_argument("-m", "--matchs_per_thread", dest="num_matchs", type=int,
                    default=100000, help="matchs per thread.")
parser.add_argument("-a", "--agent_name", dest="agent", type=str,
                    default="Bob", help="file anme.")
parser.add_argument("-p", "--post_process", dest="post",
                    action="store_true", help="post process")
parser.add_argument("-w", "--match", dest="match", type=str,
                    default="deep_cfr_ossbcfr", help="which match.")
params = parser.parse_args()


def run(*pars):
    comm = " ".join(pars)
    print(comm)
    os.system(comm)


def post(file_name, agent_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    patt = re.compile("STATE.+:([-\d]+)\|([-\d]+):(\w+)\|(\w+)")
    scores = np.empty(shape=(len(lines),), dtype=np.int32)
    pos = 0
    for line in lines:
        # print(line)
        result = patt.search(line)
        if result:
            if (result.group(3) == agent_name):
                scores[pos] = int(result.group(1))
            else:
                scores[pos] = int(result.group(2))
            # print(scores[pos])
            pos += 1
    scores = scores[:pos]
    return scores


def set_play_file(play_file, init_0, init_1):
    with open(play_file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "init_strategy_0" in lines[i]:
                lines[i] = "--init_strategy_0=" + init_0 + " \\\n"
            if "init_strategy_1" in lines[i]:
                lines[i] = "--init_strategy_1=" + init_1 + " \\\n"
        with open(play_file, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    checkfiles_0 = [
        10, 24, 38, 50, 63, 73,
        84, 98,
        110,
        124,
        134,
        148,
        160,
        174,
        186,
        196,
        209,
        220,
        230,
        240
    ]
    checkfiles_1 = [
        3000, 8000, 12000, 16000, 21000, 26000,
        30000, 35000,
        39000,
        42000,
        46000,
        50000,
        55000,
        59000,
        63000,
        67000,
        71000,
        75000,
        79000,
        83000
    ]
    if params.match == "deep_cfr_ossbcfr":
        run_play_0 = "./run_play_HULH.sh"
        run_play_1 = "./run_play_HULH_ossbcfr.sh"
        play_match = "./play_match.sh"
        match_name = "match_HULH_deep_cfr_ossbcfr"
    # match_HULH_abs_deep_cfr
    elif params.match == "abs_deep_cfr":
        run_play_0 = "../poker-cfrm/run_play.sh"
        run_play_1 = "./run_play_HULH.sh"
        play_match = "./play_match.sh"
        match_name = "match_HULH_abs_deep_cfr"
    # match_HULH_abs_ossbcfr
    elif params.match == "abs_ossbcfr":
        run_play_0 = "../poker-cfrm/run_play.sh"
        run_play_1 = "./run_play_HULH_ossbcfr.sh"
        play_match = "./play_match.sh"
        match_name = "match_HULH_abs_ossbcfr"
    # match_HULH_abs_abs
    elif params.match == "abs_abs":
        run_play_0 = "../poker-cfrm/run_play_1.sh"
        run_play_1 = "../poker-cfrm/run_play.sh"
        play_match = "./play_match.sh"
        match_name = "match_HULH_abs_abs"
    # match_HULH_random_deep_cfr
    elif params.match == "random_deep_cfr":
        run_play_0 = "./third_party/project_acpc_server/example_player.limit.2p.sh"
        run_play_1 = "./run_play_HULH.sh"
        play_match = "./play_match.sh"
        match_name = "match_HULH_random_deep_cfr"
    # match_HULH_random_ossbcfr
    elif params.match == "random_ossbcfr":
        run_play_0 = "./third_party/project_acpc_server/example_player.limit.2p.sh"
        run_play_1 = "./run_play_HULH_ossbcfr.sh"
        play_match = "./play_match.sh"
        match_name = "match_HULH_random_ossbcfr"
    else:
        print("error match")
        exit()

    for index, (c_0, c_1) in enumerate(zip(checkfiles_0, checkfiles_1)):
        c_00 = "./models/checkpoint-HULH_poker_deep_cfr_policy_0_gpu" + \
            str(c_0)
        c_01 = "./models/checkpoint-HULH_poker_deep_cfr_policy_0_gpu" + \
            str(c_0)
        c_10 = "./models/checkpoint-HULH_poker_ossbcfr_policy_0_gpu" + str(c_1)
        c_11 = "./models/checkpoint-HULH_poker_ossbcfr_policy_1_gpu" + str(c_1)

        match_name_i = match_name
        if run_play_0 == "./run_play_HULH.sh":
            set_play_file(run_play_0, c_00, c_01)
            match_name_i += "_" + str(c_0)
        elif run_play_0 == "./run_play_HULH_ossbcfr.sh":
            set_play_file(run_play_0, c_10, c_11)
            match_name_i += "_" + str(c_1)
        if run_play_1 == "./run_play_HULH.sh":
            set_play_file(run_play_1, c_00, c_01)
            match_name_i += "_" + str(c_0)
        elif run_play_1 == "./run_play_HULH_ossbcfr.sh":
            set_play_file(run_play_1, c_10, c_11)
            match_name_i += "_" + str(c_1)

        record = []
        match_names = []
        for i in range(params.num_procs):
            match_name_ii = match_name_i + "_" + str(i)
            match_names.append(match_name_ii)
            if not params.post:
                process = Process(target=run, args=(
                    play_match, match_name_ii, str(params.num_matchs), run_play_0, run_play_1))
                process.start()
                record.append(process)

        for p in record:
            p.join()

        post_datas = []
        for i in range(params.num_procs):
            post_data = post(
                "./third_party/project_acpc_server/{}.log".format(match_names[i]), params.agent)
            post_datas.append(post_data)
        data = np.concatenate(post_datas)
        mean = data.mean()
        # evaluate sample variance by setting delta degrees of freedom (ddof) to
        # 1. The degree used in calculations is N - ddof
        stddev = data.std(ddof=1)
        # Get the endpoints of the range that contains 95% of the distribution
        t_bounds = t.interval(0.95, len(data) - 1)
        # sum mean to the confidence interval
        diff = [critval * stddev / np.sqrt(len(data)) for critval in t_bounds]
        ci = [mean + critval * stddev /
              np.sqrt(len(data)) for critval in t_bounds]
        print("Mean:", mean, diff,
              "Confidence Interval 95%:", ci, "Std:", stddev, )
