import json
import logging
import os
from collections import defaultdict

from bituslabs_ds.config import LOCAL_ROOT, setup_logging
from bituslabs_ds.s3_utils import list_s3_files, read_files

setup_logging(output_path=str(LOCAL_ROOT / "jobs/log"), log_filename="analysis_mahjiang_streak_math_infererence.log")
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_trigger_stats(files, output_path):

    logger.info("get_trigger_stats start...")

    stats = {
        "Total_Game_Rounds": 0,  # total number of bet rounds without considering free game.
        "Hit_Count": 0,  # the number of bet round with at least one payouts > 0
        "Free_Game_Triggered": 0,  # number of bet rounds with free game triggered.
        "Free_Trigger_Rounds": defaultdict(int),
    }

    for f in files:
        data = read_files(f)

        # remove the first and last bet which could be incompelete.
        min_bet_num, max_bet_num = min(data["bet_num"]), max(data["bet_num"])
        data = data[(data["bet_num"] > min_bet_num) & (data["bet_num"] < max_bet_num)]

        game_rounds = len(data["bet_num"].unique())
        stats["Total_Game_Rounds"] += game_rounds

        hit_count = len(data[(data["elimination_num"] == 1) & (data["payout"] > 0)])
        stats["Hit_Count"] += hit_count

        free_game_triggered = len(data.loc[data["game_type"] == "FG", "bet_num"].unique())
        stats["Free_Game_Triggered"] += free_game_triggered

        logger.info(f"process file: {f}")
        logger.info(
            f"game rounds: {game_rounds}, min_bet_num: {min_bet_num}, max_bet_num: {max_bet_num}, hit count: {hit_count}, free game triggered: {free_game_triggered}"
        )

        free_game_rounds = (
            data.loc[data["game_type"] == "FG", ["free_elimination_num", "bet_num"]]
            .groupby("bet_num")
            .agg("max")
            .value_counts()
        )
        free_game_counts = 0
        for value, count in free_game_rounds.items():
            logger.info(f"add free game round: {value}, count: {count}")
            stats["Free_Trigger_Rounds"][str(value[0])] += count
            free_game_counts += count

        if free_game_counts != free_game_triggered:
            logger.critical("total number of free game triggered is not equal to sum of count for each round length!")

    os.makedirs(output_path, exist_ok=True)
    json.dump(stats, open(f"{output_path}/SS03_Trigger_Stats.json", "w"), indent=4)

    stats


if __name__ == "__main__":

    files = list_s3_files(bucket="bituslabs-team-ai", prefix="processed_parquet", pattern=r".*/.*.parquet")
    stats = get_trigger_stats(files, output_path=str(LOCAL_ROOT / "jobs/output_mahjiang_streak"))
