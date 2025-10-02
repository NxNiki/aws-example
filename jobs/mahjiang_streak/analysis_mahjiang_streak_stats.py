import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.s3_utils import list_s3_files, read_files

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# TODO: separate process each single file and combine all results to enable parallel processing.

# Issue: for the file A01_202501, we found the number count for free game with 20 rounds is less (by 1) in the result
# with short bet rounds (<10) removed. Compare SS03_Trigger_StatsA01_202501_short_fg.json and SS03_Trigger_StatsA01_202501.json
# no similar issue found for other files so far.

# Issue: elimination number seems not increase (in some cases) winthin a single bet round. need to double check with Kailun.

# Issue: the ratio of hit (payout > 0) over total bets (only for base game) is much higher than expected (actual > .7, expected ~ .3)


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


def get_hit_count(data, game_type=None):

    # if game_type:
    #     max_payout_per_bet = data.loc[data["game_type"] == game_type].groupby("round_num")["payout"].max()
    # else:
    #     max_payout_per_bet = data.groupby("round_num")["payout"].max()

    # return int((max_payout_per_bet > 0).sum())

    if game_type:
        data_first_round = data[
            (data["game_type"] == game_type) & (data["elimination_num"] == 1) & (data["payout"] > 0)
        ]
    else:
        data_first_round = data[(data["elimination_num"] == 1) & (data["payout"] > 0)]

    return len(data_first_round)


def get_trigger_stats(data):

    stats = {
        "Total_Game_Rounds": 0,  # total number of bet rounds without considering free game.
        "Hit_Count": 0,  # the number of bet round with at least one payouts > 0
        "Hit_Count_BG": 0,  # the number of bet round with at least one payouts > 0
        "Hit_Count_FG": 0,  # the number of bet round with at least one payouts > 0
        "Free_Game_Triggered": 0,  # number of bet rounds with free game triggered.
        "Free_Trigger_Rounds": defaultdict(int),
    }

    # update Trigger_Stats:
    game_rounds_by_bet = len(data[["bet_num", "loginname"]].drop_duplicates())
    game_rounds = len(data[data["elimination_num"] == 1])

    if game_rounds != game_rounds_by_bet:
        logger.warning(f"miss match game round calculated by elimination_num and (bet_num, loginname)")

    stats["Total_Game_Rounds"] = game_rounds

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
        logger.info(f"add free game round: {max_rounds}, count: {count}")
        stats["Free_Trigger_Rounds"][str(max_rounds)] = count
        free_game_counts += count

    if free_game_counts != free_game_triggered:
        logger.critical("total number of free game triggered is not equal to sum of count for each round length!")

    return stats


def get_payout_stats(data):

    stats = {
        "Total_Count": 0,  # BG total number of payouts = zero_count + sum (payout_count)
        "zero_Count": 0,  # BG total number of zeroes
        "Nonzero_Payouts": [],
    }


def check_existing_file(file_name):

    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            stats_f = json.load(f)

        logger.info(f"update existing stats: {stats_f}")
    else:
        stats_f = None

    return stats_f


def get_game_stats(files, output_path):

    logger.info("get_trigger_stats start...")
    os.makedirs(output_path, exist_ok=True)

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
        "Compositions": [],
    }

    stats_file_fg = f"{output_path}/SS03_FG_Items.json"
    stats_fg = {
        "Total_Count": 0,
        "Zero_Count": 0,
        "Compositions": [],
    }

    for f in files:

        logger.info(f"process file: {f}")
        basename = Path(f).stem
        trigger_stats_file_f = stats_file.replace(".json", f"{basename}.json")
        trigger_stats_file_bg_f = stats_file_bg.replace(".json", f"{basename}.json")
        trigger_stats_file_fg_f = stats_file_fg.replace(".json", f"{basename}.json")

        stats_f = check_existing_file(trigger_stats_file_f)
        stats_f_bg = check_existing_file(trigger_stats_file_bg_f)
        stats_f_fg = check_existing_file(trigger_stats_file_bg_f)

        # if stats_f is None or stats_f_bg is None or stats_f_fg is None:
        if stats_f is None:

            data = read_files(
                f, columns=["loginname", "bet_num", "payout", "game_type", "elimination_num", "free_elimination_num"]
            )
            data = remove_bet_rounds_with_short_free_game(data)
            # remove the first and last bet which could be incompelete.
            min_bet_num, max_bet_num = min(data["bet_num"]), max(data["bet_num"])
            data = data[(data["bet_num"] > min_bet_num) & (data["bet_num"] < max_bet_num)]
            logger.info(f"min_bet_num: {min_bet_num}, max_bet_num: {max_bet_num}")

            if stats_f is None:
                stats_f = get_trigger_stats(data)
                json.dump(stats_f, open(trigger_stats_file_f, "w"), indent=4)

            # if stats_f_bg is None:
            #     stats_f_bg = get_payout_stats(data[data])

        stats_trigger["Total_Game_Rounds"] += stats_f["Total_Game_Rounds"]
        stats_trigger["Hit_Count"] += stats_f["Hit_Count"]
        stats_trigger["Hit_Count_BG"] += stats_f["Hit_Count_BG"]
        stats_trigger["Hit_Count_FG"] += stats_f["Hit_Count_FG"]
        stats_trigger["Free_Game_Triggered"] += stats_f["Free_Game_Triggered"]

        for key, value in stats_f["Free_Trigger_Rounds"].items():
            stats_trigger["Free_Trigger_Rounds"][key] += value

        # update base game items:
        stats_bg["Total_Count"]

    json.dump(stats_trigger, open(stats_file, "w"), indent=4)

    stats_trigger


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=False, default=str(LOCAL_ROOT / "jobs/output_mahjiang_streak"))
    parser.add_argument("--log_output", required=False, default=str(LOCAL_ROOT / "jobs/log"))
    args = parser.parse_args()

    setup_logging(output_path=args.log_output, log_filename="analysis_mahjiang_streak_stats.log")
    files = list_s3_files(bucket="bituslabs-team-ai", prefix="processed_parquet", pattern=r".*/.*.parquet")
    stats = get_game_stats(files, output_path=args.output)
