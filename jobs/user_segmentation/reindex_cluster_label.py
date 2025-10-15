"""
For wucaishen data, we re-run cluster analysis and deployed the new model. The issue is the cluster index from the updated model
is not consistent with that used to train GAIL model and thus in simulation. Re-train GAIL model and re-run simulation takes 
huge time and computing resources (money).

This script is used the reindex the cluster label in
older clustering model based on the centriods. This is not a perfect solution but hopefully acceptable work around.
"""

reference_cluster_result = ""
reindex_cluster_result = ""
