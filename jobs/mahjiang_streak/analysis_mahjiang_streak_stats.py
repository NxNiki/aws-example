import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.s3_utils import list_s3_files, read_files


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

REMOVE_RARE_PAYOUT_THRESHOLD = 0
REMOVE_RARE_COMPOSITION_THRESHOLD = 0
SUBSAMPLE_RATIO = 0.007
MIN_SUBSAMPLE_COUNT = 100

GET_STATS = {"BG": False, "FG": True, "Trigger": False}

# TODO: separate process each single file and combine all results to enable parallel processing.

# issue: the total number of bet rounds is not consistent using different methods (elimination_num and (bet_num, loginname))
#   the difference is small though.

# issue: For some bet, the base_account (the first bet amount in a bet round) is 0. This result in Inf when calculating
#   the stardard payout (payout/base_account * 20)

# issue: there are very rare cases where free game round is odd number.


def remove_first_and_last_n_bets(data, n=1):
    # Remove rows where bet_num is the minimum for each loginname

    data = data.sort_values(by=["loginname", "bet_num", "elimination_num"])

    min_betnum_per_login = data.groupby("loginname")["bet_num"].transform(
        lambda x: sorted(set(x))[n - 1] if len(set(x)) > n else max(x)
    )
    data = data[data["bet_num"] > min_betnum_per_login].copy()

    max_betnum_per_login = data.groupby("loginname")["bet_num"].transform(
        lambda x: sorted(set(x))[-n] if len(set(x)) > n else min(x)
    )
    data = data[data["bet_num"] < max_betnum_per_login].copy()

    return data


def sample_bet_rounds(data, ratio=SUBSAMPLE_RATIO, min_count=MIN_SUBSAMPLE_COUNT):
    """
    Randomly sample bet rounds from the DataFrame.
    All rows with the same ("loginname", "bet_num") will be either selected or not selected.
    """
    # Identify unique bet rounds by ("loginname", "bet_num")
    bet_rounds = data[["loginname", "bet_num"]].drop_duplicates()
    n_sample = max(int(len(bet_rounds) * ratio), 1)

    if n_sample < min_count:
        logger.info(f"n_sample is less than {min_count}, no sampling will be performed")
        return data

    sampled_rounds = bet_rounds.sample(n=n_sample, random_state=42)
    sampled_data = data.merge(sampled_rounds, on=["loginname", "bet_num"], how="inner")
    return sampled_data


def remove_bet_rounds_with_short_free_game(data):
    """
    Remove rows from the DataFrame where bet_num corresponds to a group
    whose max free_elimination_num (for game_type == 'FG') is less than 10.
    Returns a filtered DataFrame.
    """
    # Find max free_elimination_num for each (bet_num, loginname) where game_type == 'FG'
    free_game_rounds = (
        data.loc[data["game_type"] == "FG", ["bet_num", "loginname", "free_elimination_num"]]
        .groupby(["bet_num", "loginname"])
        .agg("max")
    )

    # Identify (bet_num, loginname) where max free_elimination_num < 10
    short_round_keys = free_game_rounds.index[free_game_rounds["free_elimination_num"] < 10].tolist()

    if short_round_keys:
        logger.warning(
            f"Removing the following (bet_num, loginname) with free game rounds less than 10: {short_round_keys}"
        )

    # Remove rows with these (bet_num, loginname) combinations
    mask = ~data.set_index(["bet_num", "loginname"]).index.isin(short_round_keys)
    filtered_data = data[mask].copy()

    return filtered_data


def add_base_account(data):

    data = data.copy()
    data["base_account"] = data.groupby(["bet_num", "loginname"])["account"].transform("max")

    # remove bet round (including BG and FG) with 0 base_account:
    mask = data["base_account"] == 0

    if any(mask):
        logger.warning(
            "remove bet rounds with 0 base_account: %s \n%d/%d, %.4f",
            data.loc[mask, ["loginname", "bet_num"]].drop_duplicates(),
            sum(mask),
            len(data),
            sum(mask) / len(data) if len(data) > 0 else 0,
        )

    return data[~mask]


def get_hit_count(data, game_type=None, group_cols=["bet_num", "loginname"]):

    if game_type:
        max_payout = data[data["game_type"] == game_type].groupby(group_cols)["payout"].max()
    else:
        max_payout = data.groupby(group_cols)["payout"].max()

    return sum(max_payout > 0)


def get_trigger_stats(data):

    stats = {
        "Total_Game_Rounds": 0,  # total number of bet rounds without considering free game.
        "Hit_Count": 0,  # the number of bet round with at least one payouts > 0
        "Hit_Count_BG": 0,
        "Hit_Count_FG": 0,
        "Free_Game_Triggered": 0,  # number of bet rounds with free game triggered.
        "Free_Trigger_Rounds": defaultdict(int),
    }

    # update Trigger_Stats:
    game_rounds_by_bet = len(data[["bet_num", "loginname"]].drop_duplicates())
    game_rounds = len(data[(data["elimination_num"] == 1) & (data["game_type"] == "BG")])

    if game_rounds != game_rounds_by_bet:
        logger.warning(
            f"inconsistent game round calculated by elimination_num {game_rounds} and (bet_num, loginname) {game_rounds_by_bet}"
        )

    stats["Total_Game_Rounds"] = game_rounds_by_bet

    stats["Hit_Count"] = get_hit_count(data)
    stats["Hit_Count_BG"] = get_hit_count(data, game_type="BG")
    stats["Hit_Count_FG"] = get_hit_count(data, game_type="FG")

    free_game_triggered = len(data.loc[data["game_type"] == "FG", ["loginname", "bet_num"]].drop_duplicates())
    stats["Free_Game_Triggered"] = free_game_triggered

    logger.info(f"stats: {stats}")

    # Group by loginname and bet_num, get max free_elimination_num for each group
    free_game_rounds = (
        data.loc[data["game_type"] == "FG", ["free_elimination_num", "loginname", "bet_num"]]
        .groupby(["loginname", "bet_num"])["free_elimination_num"]
        .max()
    )

    # Count occurrences of each max free_elimination_num value
    free_game_rounds = free_game_rounds.value_counts().sort_index()

    free_game_counts = 0
    for max_rounds, count in free_game_rounds.items():
        if max_rounds % 2 == 1:
            logger.warning(f"free game round: {max_rounds} is ODD! count: {count}")
        else:
            logger.info(f"add free game round: {max_rounds}, count: {count}")
        stats["Free_Trigger_Rounds"][str(max_rounds)] = count
        free_game_counts += count

    if free_game_counts != free_game_triggered:
        logger.critical("total number of free game triggered is not equal to sum of count for each round length!")

    return stats


def get_item_stats(data, game_type=None):

    stats = {
        "Total_Count": 0,  # Total number of payouts = zero_count + sum (payout_count)
        "Zero_Count": 0,  # total number of zero payouts
        "Nonzero_Payouts": [],
    }

    # select sub-columns to save memory space:
    if game_type == "BG":
        columns = ["bet_num", "loginname", "payout", "base_account", "game_type"]
        group_cols = ["bet_num", "loginname"]
        data = data.loc[data["game_type"] == game_type, columns].copy()
    elif game_type == "FG":
        columns = ["bet_num", "loginname", "payout", "base_account", "game_type", "free_elimination_num"]
        group_cols = ["bet_num", "loginname", "free_elimination_num"]
        data = data.loc[data["game_type"] == game_type, columns].copy()
    else:
        data = data[columns].copy()

    game_rounds = len(data[group_cols].drop_duplicates())
    stats["Total_Count"] = game_rounds

    stats["Zero_Count"] = game_rounds - get_hit_count(data)
    stats["Nonzero_Payouts"] = get_payout_stats(data, group_cols)

    logger.info(f"item stats for gametype {game_type}: {stats}")

    return stats


def get_payout_stats(data, group_cols=["bet_num", "loginname"]):

    payout_stats = []
    data = data[data["payout"] > 0].copy()
    data["standard_payout"] = (data["payout"] / data["base_account"] * 20).round().astype(int)
    data["total_payouts"] = data.groupby(group_cols)["standard_payout"].transform("sum")
    total_payouts = sorted(data["total_payouts"].unique())

    for total_payout in total_payouts:

        logger.info(f"process payouts: {total_payout}")

        data_payout = data[data["total_payouts"] == total_payout]
        p_stats = {
            "payouts": total_payout,
            "payouts_count": len(data_payout[group_cols].drop_duplicates()),
            "Compositions": get_composition_stats(data_payout, group_cols),
        }

        payout_stats.append(p_stats)

    return payout_stats


def get_composition_stats(data, group_cols=["bet_num", "loginname"]):
    """
    data with same total_payouts.
    """

    game_type = data["game_type"].unique()
    if len(game_type) != 1:
        raise ValueError("Only single game_type is valid!")

    if game_type == "BG":
        multiplier = [1, 2, 3, 5]
    else:
        multiplier = [2, 4, 6, 10]

    payouts_compositions = []
    data = data.copy()
    data["level"] = data.groupby(group_cols)["standard_payout"].transform("count")

    for level in sorted(data["level"].unique()):
        logger.info(f"process payout level: {level}")
        data_level = data.loc[data["level"] == level, group_cols + ["standard_payout"]]

        # For each unique (bet_num, loginname) in this level, get the sequence of standard_payouts (ordered as they appear)
        composition_counter = {}
        grouped = data_level.groupby(group_cols)
        prev_payout_sequence = []
        for _, group in grouped:
            payout_sequence = group["standard_payout"].tolist()  # keep as list, order matters
            if payout_sequence != prev_payout_sequence:
                logger.info(f"process payout sequence: {payout_sequence}")
                prev_payout_sequence = payout_sequence

            # Use str of list as key to preserve order and uniqueness
            key = str(payout_sequence)
            if key in composition_counter:
                composition_counter[key]["count"] += 1
            else:
                composition_counter[key] = {"count": 1, "sequence": payout_sequence}

        level_multiplier = (
            multiplier[:level]
            if level <= len(multiplier)
            else multiplier + [multiplier[-1]] * (level - len(multiplier))
        )
        for comp in composition_counter.values():
            payouts_compositions.append(
                {
                    "composition_count": comp["count"],
                    "levels": level,
                    "payout": comp["sequence"],
                    "multiplier": level_multiplier,
                    "odds": [int(p / m) for p, m in zip(comp["sequence"], level_multiplier)],
                }
            )

    return payouts_compositions


def check_existing_file(file_name):

    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            stats_f = json.load(f)

        logger.info(f"read existing stats file: {file_name}")
    else:
        stats_f = None

    return stats_f


def update_trigger_stats(stats_total: Dict, stats: Dict) -> Dict:

    if not stats:
        return {}

    stats_total["Total_Game_Rounds"] += stats["Total_Game_Rounds"]
    stats_total["Hit_Count"] += stats["Hit_Count"]
    stats_total["Hit_Count_BG"] += stats["Hit_Count_BG"]
    stats_total["Hit_Count_FG"] += stats["Hit_Count_FG"]
    stats_total["Free_Game_Triggered"] += stats["Free_Game_Triggered"]

    for key, value in stats["Free_Trigger_Rounds"].items():
        stats_total["Free_Trigger_Rounds"][key] += value

    return stats_total


def update_item_stats(stats_total, stats):

    if not stats:
        return {}

    stats_total["Total_Count"] += stats["Total_Count"]
    stats_total["Zero_Count"] += stats["Zero_Count"]

    stats_total["Nonzero_Payouts"] = update_payout_stats(stats_total["Nonzero_Payouts"], stats["Nonzero_Payouts"])


def update_payout_stats(payout_total: List[Dict], payout: List[Dict]) -> List[Dict]:

    payout_index = {d["payouts"]: i for i, d in enumerate(payout_total)}
    for d in payout:
        if d["payouts"] in payout_index:
            index = payout_index[d["payouts"]]
            payout_total[index]["payouts_count"] += d["payouts_count"]
            payout_total[index]["Compositions"] = update_payout_composition(
                payout_total[index]["Compositions"], d["Compositions"]
            )
        else:
            payout_total.append(d)

    return payout_total


def update_payout_composition(composition_total: List[Dict], composition: List[Dict]) -> List[Dict]:

    composition_index = {str(d["payout"]): i for i, d in enumerate(composition_total)}
    for d in composition:
        if str(d["payout"]) in composition_index:
            index = composition_index[str(d["payout"])]
            composition_total[index]["composition_count"] += d["composition_count"]
        else:
            composition_total.append(d)

    return composition_total


def remove_rare_payouts(stats: Dict):

    count_threshold = int(stats["Total_Count"] * REMOVE_RARE_PAYOUT_THRESHOLD)

    if count_threshold == 0:
        logger.info("count_threshold is 0, no rare payouts will be removed")
        return stats

    selected_payouts = []
    for payout in stats["Nonzero_Payouts"]:
        if payout["payouts_count"] < count_threshold:
            logger.info(
                f"remove rare payout: {payout['payouts']}, count: {payout['payouts_count']}, threshold: {count_threshold}"
            )
        else:
            payout = remove_rare_compositions(payout, count_threshold=count_threshold)
            selected_payouts.append(payout)
    stats["Nonzero_Payouts"] = selected_payouts
    return stats


def remove_rare_compositions(payout: Dict, count_threshold: Optional[float] = None):
    if count_threshold is None:
        count_threshold = int(payout["payouts_count"] * REMOVE_RARE_COMPOSITION_THRESHOLD)

    selected_compositions = []
    for comp in payout["Compositions"]:
        if comp["composition_count"] < count_threshold:
            logger.info(
                f"remove rare composition: {comp['payout']}, count: {comp['composition_count']}, threshold: {count_threshold}"
            )
        else:
            selected_compositions.append(comp)
    payout["Compositions"] = selected_compositions
    return payout


def get_max_level(stats) -> int:
    max_level = 0
    for payout in stats["Nonzero_Payouts"]:
        for comp in payout["Compositions"]:
            max_level = max(max_level, comp["levels"])
    return max_level


def get_game_stats(files, output_path):

    logger.info("get_trigger_stats start...")
    os.makedirs(f"{output_path}/SS03_Trigger_Stats", exist_ok=True)
    os.makedirs(f"{output_path}/SS03_BG_Items", exist_ok=True)
    os.makedirs(f"{output_path}/SS03_FG_Items", exist_ok=True)

    stats_file = f"{output_path}/SS03_Trigger_Stats.json"
    stats_trigger = {
        "Total_Game_Rounds": 0,  # total number of bet rounds without considering free game.
        "Hit_Count": 0,  # the number of bet round with at least one payouts > 0
        "Hit_Count_BG": 0,  # the number of bet round with at least one payouts > 0
        "Hit_Count_FG": 0,  # the number of bet round with at least one payouts > 0
        "Free_Game_Triggered": 0,  # number of bet rounds with free game triggered.
        "Free_Trigger_Rounds": defaultdict(int),
    }

    stats_file_bg = f"{output_path}/SS03_BG_Items.json"
    stats_bg = {
        "Total_Count": 0,
        "Zero_Count": 0,
        "max_level": 0,
        "Nonzero_Payouts": [],
    }

    stats_file_fg = f"{output_path}/SS03_FG_Items.json"
    stats_fg = {
        "Total_Count": 0,
        "Zero_Count": 0,
        "max_level": 0,
        "Nonzero_Payouts": [],
    }

    for f in files:

        logger.info(f"process file: {f}")
        basename = Path(f).stem

        if GET_STATS["Trigger"]:
            trigger_stats_file_f = stats_file.replace(".json", f"/{basename}.json")
            stats_f = check_existing_file(trigger_stats_file_f)
        else:
            stats_f = {}

        if GET_STATS["BG"]:
            trigger_stats_file_bg_f = stats_file_bg.replace(".json", f"/{basename}.json")
            stats_f_bg = check_existing_file(trigger_stats_file_bg_f)
        else:
            stats_f_bg = {}

        if GET_STATS["FG"]:
            trigger_stats_file_fg_f = stats_file_fg.replace(".json", f"/{basename}.json")
            stats_f_fg = check_existing_file(trigger_stats_file_fg_f)
        else:
            stats_f_fg = {}

        if stats_f is None or stats_f_bg is None or stats_f_fg is None:
            data = read_files(
                f,
                columns=[
                    "loginname",
                    "bet_num",
                    "payout",
                    "account",
                    "game_type",
                    "elimination_num",
                    "free_elimination_num",
                ],
            )
            # remove the first and last bet which could be incompelete.
            data = remove_first_and_last_n_bets(data)
            data = sample_bet_rounds(data)
            data = remove_bet_rounds_with_short_free_game(data)
            data = add_base_account(data)

            if stats_f is None:
                stats_f = get_trigger_stats(data)
                json.dump(stats_f, open(trigger_stats_file_f, "w"), indent=4, cls=NumpyEncoder)

            if stats_f_bg is None:
                stats_f_bg = get_item_stats(data, game_type="BG")
                json.dump(stats_f_bg, open(trigger_stats_file_bg_f, "w"), indent=4, cls=NumpyEncoder)

            if stats_f_fg is None:
                stats_f_fg = get_item_stats(data, game_type="FG")
                json.dump(stats_f_fg, open(trigger_stats_file_fg_f, "w"), indent=4, cls=NumpyEncoder)

        update_trigger_stats(stats_trigger, stats_f)
        update_item_stats(stats_bg, stats_f_bg)
        update_item_stats(stats_fg, stats_f_fg)

    if stats_trigger:
        stats_trigger["Free_Trigger_Rounds"] = dict(sorted(stats_trigger["Free_Trigger_Rounds"].items()))
        json.dump(stats_trigger, open(stats_file, "w"), indent=4, cls=NumpyEncoder)

    if stats_bg:
        stats_bg = remove_rare_payouts(stats_bg)
        stats_bg["max_level"] = get_max_level(stats_bg)
        json.dump(stats_bg, open(stats_file_bg, "w"), indent=4, cls=NumpyEncoder)

    if stats_fg:
        stats_fg = remove_rare_payouts(stats_fg)
        stats_fg["max_level"] = get_max_level(stats_fg)
        json.dump(stats_fg, open(stats_file_fg, "w"), indent=4, cls=NumpyEncoder)

    stats_trigger


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=False, default=str(LOCAL_ROOT / "jobs/output_mahjiang_streak"))
    parser.add_argument("--log_output", required=False, default=str(LOCAL_ROOT / "jobs/log"))
    args = parser.parse_args()

    setup_logging(output_path=args.log_output, log_filename="analysis_mahjiang_streak_stats.log")
    files = list_s3_files(bucket="bituslabs-team-ai", prefix="processed_parquet", pattern=r".*/.*.parquet")
    stats = get_game_stats(files, output_path=args.output)
